"""Bounded, clause-scoped preview intent with explicit conversational transitions.

This is a lexical controller, not a general semantic parser. Cancellation,
retention, denial, questions, and reference-only followups are distinct states.
Only user turns and a fresh owned catalog establish a preview target; the result
never authorizes a business write. v4/v5/v6 continue using their old resolvers.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re
import unicodedata

from .grounded_target import mentions
from .storage import ServiceError

VERSION = 'scoped_preview_intent_v1'
CANCEL = re.compile(r'取消|撤销|撤掉|撤单|退订|作废|终止(?:这笔)?交易|撤回购买|不需要了|不想要了|不想买了|不打算买了|不买了|不要了|不用了')
ORDER_CLOSE = re.compile(r'(?:订单|交易|购买).{0,10}(?:关掉|关闭|结束|撤回|终止)|(?:关掉|关闭|终止|撤回).{0,10}(?:订单|交易|购买)')
KEEP = re.compile(r'保留|留下|留着|只要|继续购买|照常')
NEGATED_ACTION = re.compile(r'(?:不要|不用|不必|别|暂不|先不|先别|不想|不打算|不准备|不能|不可以|不准|无须|停止|暂停).{0,6}(?:取消|撤销|撤掉|撤单|退订|作废|终止|操作|处理)|不取消')
DENIAL = re.compile(r'(?:不是|并非|不代表).{0,8}(?:我|本人).{0,8}(?:想|要|意思|要求|同意|授权)|'
                    r'(?:不是真的|并不是真的|不是要).{0,6}(?:想|要|取消|撤单)|'
                    r'(?:我|本人).{0,4}(?:没有|没|未|从未|并未|并不).{0,6}(?:提出|发起|要求|同意|授权|说过|表示|打算|想)|'
                    r'(?:尚未|还没|没有).{0,6}(?:决定|确定|考虑好)|'
                    r'(?:取消|撤单|退订).{0,30}(?:是(?:我(?:的)?)?(?:别人|他人|同事|朋友|客服)|不是我|并非我)|'
                    r'(?:别人|他人|朋友|同事|客服).{0,6}(?:说|要求|提出).{0,18}(?:取消|撤单|退订)')
PAUSED_ACTION = re.compile(r'(?:取消|撤单|退订|取消申请|操作).{0,8}(?:先停|暂停|停止|先别)')
QUOTED_OR_HYPOTHETICAL = re.compile(r'''[“”"'‘’「」『』]|复述|转述|引用|翻译|举例|照着|模仿|假设|如果|假如|有人说''')
POLICY_QUESTION = re.compile(r'怎么|如何|教程|规则|政策|规定|流程|条件|含义|什么意思|是否已经|是不是|是否取消|'
                             r'已经.{0,12}取消|取消(?:进度|了吗|了没|过)|(?:能|可以)取消吗')
READ_ONLY = re.compile(r'只(?:是|想)?(?:查询|查|看|了解)|查询|查一下|查查|看看|什么状态|处理结果|到账|能不能|是否可以|聊点别的')
MONEY = re.compile(r'(?<![\d.])(\d{1,8}(?:\.\d{1,2})?)\s*(?:元|块)(?![\d.])')
CONFIRM_VIEW = re.compile(r'(?:看|显示|展示|打开|给).{0,10}(?:确认页面|确认卡|确认申请|确认按钮|预览)')
FOLLOWUP = re.compile(r'(?:(?:我|现在|已经|还是|再|再次|就|明确|真的|确定|确认|同意|决定|希望|要|想|请|帮我|麻烦|给我|继续|'
                      r'取消|撤销|撤单|不需要了|不想要了|不想买了|不买了|不要了|不用了|操作|这|那|原来|刚才|一|笔|单|张|份|金额|价格|订单|购买|交易|的|了|吧|好的|在聊天里)|[\s，,。.!！])*')
SELECTION = re.compile(r'(?:选|选择|就|是|要|金额|价格|那|这|一笔|笔|订单|的|请|我|了|吧|[\s，,。.!！])*')


@dataclass(frozen=True)
class IntentFrame:
    kind: str
    ids: tuple[str, ...] = ()
    retained: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()
    reason: str = 'target_needs_clarification'
    inherited: bool = False


def normalize(text):
    return unicodedata.normalize('NFKC', text).strip()


def transition(text, previous, orders):
    text = normalize(text)
    if DENIAL.search(text) or QUOTED_OR_HYPOTHETICAL.search(text) or PAUSED_ACTION.search(text):
        return IntentFrame('blocked_scope')
    if POLICY_QUESTION.search(text):
        return IntentFrame('question')
    all_ids, all_anchors = mentions(text, orders, catalog_aliases=True)
    clauses = [part.strip() for part in re.split(r'[，,。；;！？!?\n]|但是|不过|而是', text) if part.strip()]
    positive, retained, anchors = set(), set(), set()
    action = False
    for clause in clauses:
        ids, found = mentions(clause, orders, catalog_aliases=True)
        # "不想要了" negates purchasing, whereas "不想取消" negates cancellation.
        negative_scope = re.sub(r'不需要了|不想要了|不想买了|不打算买了|不买了|不要了|不用了', '', clause)
        if NEGATED_ACTION.search(negative_scope) or KEEP.search(clause):
            if not ids:
                if KEEP.search(clause) and re.search(r'其他|其它|别的', clause) and action:
                    continue
                return IntentFrame('withdrawn')
            retained.update(ids)
            # A clause with both a keep and cancel target is deliberately ambiguous.
            if KEEP.search(clause) and CANCEL.search(clause) and not NEGATED_ACTION.search(negative_scope):
                return IntentFrame('ambiguous', reason='target_ambiguous')
        elif CANCEL.search(clause) or ORDER_CLOSE.search(clause):
            remaining = clause
            for anchor in found: remaining = remaining.replace(anchor, '')
            if re.search(r'和|以及|还有|或者|或|、', remaining):
                return IntentFrame('ambiguous', reason='target_ambiguous')
            action = True; positive.update(ids); anchors.update(found)
    if positive & retained:
        return IntentFrame('contradictory')
    amounts = {int(Decimal(m.group(1))*100) for m in MONEY.finditer(text)}
    if len(amounts) > 1:
        return IntentFrame('ambiguous', reason='target_ambiguous')
    previous_active = previous is not None and previous.kind == 'cancel_preview'
    inherited = False
    if action:
        ids = positive or (all_ids - retained)
        if not ids and previous_active:
            plain = re.sub(r'[，,。；;！？!?\n]', '', text)
            # Only a bounded reference/confirmation can inherit; never an unknown new product.
            first = next((c for c in clauses if CANCEL.search(c)), '')
            tail_is_view = all(CONFIRM_VIEW.search(c) or c == first for c in clauses)
            if FOLLOWUP.fullmatch(plain) or (FOLLOWUP.fullmatch(first) and tail_is_view):
                ids = set(previous.ids); anchors.update(previous.anchors); inherited = True
            elif amounts and not all_ids:
                remainder = MONEY.sub('', text)
                if FOLLOWUP.fullmatch(remainder):
                    ids = set(previous.ids); anchors.update(previous.anchors); inherited = True
        if not ids:
            return IntentFrame('missing_target', reason='target_not_found')
    else:
        if retained or READ_ONLY.search(text) or not previous_active:
            return IntentFrame('query_or_idle' if not retained else 'withdrawn')
        if FOLLOWUP.fullmatch(text) and text:
            ids = set(previous.ids); anchors.update(previous.anchors); inherited = True
        else:
            remainder = MONEY.sub('', text)
            for anchor in all_anchors: remainder = remainder.replace(anchor, '')
            if not (all_ids or amounts) or not SELECTION.fullmatch(remainder):
                return IntentFrame('unrelated')
            ids = set(previous.ids) & all_ids if all_ids else set(previous.ids)
            anchors.update(all_anchors or previous.anchors); inherited = True
    ids -= retained
    if amounts: ids &= {r['id'] for r in orders if r['paid_cents'] in amounts}
    return IntentFrame('cancel_preview', tuple(sorted(ids)), tuple(sorted(retained)), tuple(sorted(anchors)),
                       'target_not_found', inherited)


def interpret(text, previous_users, orders):
    if len(previous_users) > 256: return IntentFrame('history_limit')
    previous = None
    for old in previous_users: previous = transition(old, previous, orders)
    return transition(text, previous, orders)


def resolve(text, previous_users, orders):
    frame = interpret(text, previous_users, orders)
    if frame.kind != 'cancel_preview': raise ServiceError(frame.reason, 409)
    if not frame.ids: raise ServiceError('target_not_found', 409)
    if len(frame.ids) != 1: raise ServiceError('target_ambiguous', 409)
    row = next(row for row in orders if row['id'] == frame.ids[0])
    return dict(order_id=row['id'], order_version=row['version'], candidate_count=1,
                source='user_request_and_fresh_owned_orders', matching_version=VERSION,
                intent='cancel_preview_only', user_anchors=list(frame.anchors),
                retained_order_ids=list(frame.retained), inherited_target=frame.inherited)


def guidance(text, previous_users, orders):
    """Read-only tool data that helps the model plan; never a write authorization."""
    frame = interpret(text, previous_users, orders)
    return dict(version=VERSION, intent=frame.kind, candidate_order_ids=list(frame.ids),
                retained_order_ids=list(frame.retained), unique_target=len(frame.ids)==1 and frame.kind=='cancel_preview',
                inherited_target=frame.inherited,
                next_action='get_order_then_preview_cancel' if len(frame.ids)==1 and frame.kind=='cancel_preview' else 'clarify_or_answer_only',
                authorization_for_write=False, source='current_user_and_fresh_owned_catalog')
