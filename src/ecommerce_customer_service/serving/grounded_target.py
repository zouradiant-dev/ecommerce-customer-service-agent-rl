"""Catalog-grounded cancellation intent, separate from irreversible authorization.

This is a deterministic lexical resolver, not unrestricted language understanding.
Only user text and fresh owned catalog entries can establish a target. Ambiguous,
quoted, hypothetical and unsupported requests fail closed. A result authorizes
only a preview: ownership, version and UI confirmation remain separate checks.
"""
from __future__ import annotations

from decimal import Decimal
import re
import unicodedata

from .storage import ServiceError
from .target_binding import ALIASES, subject as legacy_subject

VERSION = 'catalog_grounded_intent_v1'
ACTION = re.compile(r'取消|撤销|撤掉|撤单|退订|作废|不要了|不想要了|不想买了|不用了')
NEGATIVE = re.compile(r'(?:不要|不用|不必|别|不|停止|暂停|暂不|先不|先别).{0,8}(?:取消|撤销|撤掉|撤单|退订|作废|操作|处理)|保留')
UNSAFE_SCOPE = re.compile(r'''[“”"'‘’「」『』]|例子|举例|照着|模仿|复述|引用|转述|假设|如果|有人说|怎么|如何|规则|教程|说明|是否已经|已经.{0,8}取消|取消过|取消(?:条件|流程|进度|了吗|了没)|(?:能|可以)取消吗|是否取消|是不是取消|没(?:有)?(?:说|要求)''')
READ_ONLY = re.compile(r'只(?:是|想)?(?:查询|查|看|了解)|查询|查一下|看看|多少钱|什么状态|处理结果|到账|能不能|是否可以')
MONEY = re.compile(r'(?<![\d.])(\d{1,8}(?:\.\d{1,2})?)\s*(?:元|块)(?![\d.])')
CONFIRM = re.compile(r'^(?:好的?[，,]?|那就|现在)?(?:确认|确定)(?:取消|撤销)?(?:吧|了)?[。.!！]?$')


def normalize(text):
    return unicodedata.normalize('NFKC', text).strip()


def mentions(text, orders, *, catalog_aliases=False):
    matches = []
    bounded_subject = legacy_subject(text)
    for row in orders:
        for anchor, kind in [(row['id'], 'id'), (row['product'], 'product'),
                             *[(alias, 'alias') for alias in ALIASES.get(row['product'], ())]]:
            for hit in re.finditer(re.escape(anchor), text):
                left, right = text[:hit.start()], text[hit.end():]
                if kind == 'id' and (re.search(r'[A-Za-z0-9_]$', left) or re.match(r'[A-Za-z0-9_]', right)):
                    continue
                if kind == 'alias' and bounded_subject != anchor:
                    # Do not let "有线耳机" silently mean the catalog's "无线耳机".
                    left_function = catalog_aliases and re.search(r'(?:给|把|将|替|帮|为|的|这|那|该|一|笔|张|份|件|只|款|个|购买|订购|关于|针对|办理)$', left)
                    right_function = catalog_aliases and re.match(r'(?:的|订单|这|那|办理|进行|申请|请|帮|麻烦|退订|撤单|作废|取消|撤掉)', right)
                    if left and re.search(r'[\u4e00-\u9fff]$', left) and not left_function and not left.endswith(
                            ('取消', '撤销', '撤掉', '撤单', '退订', '作废', '把', '将', '的', '那只', '那个', '这款', '和', '及', '或', '或者')):
                        continue
                    if re.match(r'[\u4e00-\u9fff]', right) and not right_function and not right.startswith(
                            ('订单', '的', '这', '那', '不用了', '不要了', '不想要了', '不想买了', '先保留', '保留', '和', '及', '或', '、')):
                        continue
                matches.append((hit.start(), hit.end(), row['id'], anchor))
    # Prefer the longest catalog anchor at an overlapping location. Same-name
    # orders retain all IDs; no price/status preference silently selects one.
    kept = [m for m in matches if not any(n[0] <= m[0] and n[1] >= m[1] and n[1]-n[0] > m[1]-m[0] for n in matches)]
    return {m[2] for m in kept}, sorted({m[3] for m in kept})


def frame(text, previous, orders, *, catalog_aliases=False):
    text = normalize(text)
    if UNSAFE_SCOPE.search(text):
        return None, 'target_needs_clarification'
    if CONFIRM.fullmatch(text):
        return previous, 'target_needs_clarification'
    clauses = [part.strip() for part in re.split(r'[，,。；;！？!?\n]|但是|但|而是', text) if part.strip()]
    positive, negative, anchors = set(), set(), []
    action = False
    all_ids, all_anchors = mentions(text, orders, catalog_aliases=catalog_aliases)
    for clause in clauses:
        ids, found = mentions(clause, orders, catalog_aliases=catalog_aliases)
        negative_scope = re.sub(r'不用了|不要了|不想要了|不想买了', '', clause)
        if NEGATIVE.search(negative_scope):
            if not ids:
                return None, 'target_needs_clarification'
            negative.update(ids)
        elif ACTION.search(clause):
            remaining = clause
            for anchor in found:
                remaining = remaining.replace(anchor, '')
            if re.search(r'和|以及|还有|或者|或|、', remaining):
                return None, 'target_ambiguous'
            action = True
            positive.update(ids)
            anchors.extend(found)
    if positive & negative:
        return None, 'target_needs_clarification'
    amounts = {int(Decimal(hit.group(1)) * 100) for hit in MONEY.finditer(text)}
    if len(amounts) > 1:
        return None, 'target_ambiguous'
    if not action:
        # A fresh query, withdrawal, or unrelated turn never inherits old intent.
        if negative or READ_ONLY.search(text) or not previous:
            return None, 'target_needs_clarification'
        remainder = MONEY.sub('', text)
        for anchor in all_anchors:
            remainder = remainder.replace(anchor, '')
        if not (all_ids or amounts) or not re.fullmatch(r'[\s,，。.!！]*(?:选|选择|就|是|要|金额|价格|那|这|一笔|笔|订单|的|请|我|了|吧|\s|[,，。.!！])*', remainder):
            return None, 'target_needs_clarification'
        ids = set(previous['ids']) & all_ids if all_ids else set(previous['ids'])
        anchors = all_anchors or previous['anchors']
    else:
        if positive:
            ids = positive
        elif all_ids - negative:
            ids = all_ids - negative
            anchors = all_anchors
        elif amounts and previous:
            ids = set(previous['ids'])
            anchors = previous['anchors']
        else:
            return None, 'target_not_found' if re.search(r'取消|撤销|撤掉|退订|作废', text) else 'target_needs_clarification'
    if amounts:
        ids &= {row['id'] for row in orders if row['paid_cents'] in amounts}
    ids -= negative
    return {'ids': sorted(ids), 'anchors': anchors, 'intent': 'cancel_preview_only'}, 'target_not_found'


def resolve(text, previous_users, orders, *, catalog_aliases=False):
    if len(previous_users) > 256:
        raise ServiceError('target_needs_clarification', 409)
    current = None
    for old in previous_users:
        current, _ = frame(old, current, orders, catalog_aliases=catalog_aliases)
    current, error = frame(text, current, orders, catalog_aliases=catalog_aliases)
    if current is None:
        raise ServiceError(error, 409)
    if not current['ids']:
        raise ServiceError('target_not_found', 409)
    if len(current['ids']) != 1:
        raise ServiceError('target_ambiguous', 409)
    row = next(row for row in orders if row['id'] == current['ids'][0])
    return {'order_id': row['id'], 'order_version': row['version'], 'candidate_count': 1,
            'source': 'user_request_and_fresh_owned_orders', 'matching_version': VERSION + ('_catalog_aliases' if catalog_aliases else ''),
            'user_anchors': current['anchors'], 'intent': current['intent']}
