"""A bounded request parser, catalog resolver and persisted dialogue frame.

This is deterministic engineering, not learned semantic understanding. Clause
roles and source spans are kept separately from catalog resolution and preview
eligibility. No frame grants permission to cancel or refund an order.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
import re

from .intent_target import DENIAL, QUOTED_OR_HYPOTHETICAL, PAUSED_ACTION, POLICY_QUESTION, MONEY
from .storage import ServiceError, encode
from .target_binding import ALIASES

PROFILE = 'binding_state_v8'
VERSION = 'request_frame_v1'
ACTION = re.compile(r'取消|撤销|撤掉|撤单|退订|作废|终止(?:这笔)?交易|撤回购买|不需要了|不想要了|不想买了|不打算买了|不买了|不要了|不用了|不再购买|不再买')
CLOSE = re.compile(r'关掉|关闭|关了|结束|销掉|终止|撤回')
PURCHASE = re.compile(r'订单|购买|交易|这单|那单|这笔|那笔|这一单|那一单')
KEEP = re.compile(r'保留|留下|留着|照常|我需要|继续(?:购买|收货|寄|发|要)|只要(?!取消|撤|退订)')
NEGATE = re.compile(r'(?:请勿|不要|不用|不必|别|暂不|先不|先别|不想|不打算|不准备|不能|不可以|不准|无须|停止|暂停).{0,6}(?:取消|撤销|撤掉|撤单|退订|作废|操作|处理)|不取消|(?:反对|拒绝|打消|收回).{0,40}(?:取消|撤单|退订)')
DISAVOW = re.compile(r'(?:我|本人).{0,6}(?:没|未|没有|从未|从来没|并未).{0,4}(?:叫|让|要求|同意|授权).{0,16}(?:取消|撤单|退订)')
POST_VETO = re.compile(r'(?:取消|撤单|退订|申请).{0,8}(?:别|不要|不必).{0,4}(?:建|创建|办理|处理|提交)')
STATUS_QUESTION = re.compile(r'(?:有没有|是否|是不是).{0,8}(?:被)?取消|取消.{0,4}(?:了吗|了没)')
QUERY = re.compile(r'查询|查一下|查查|看看|多少钱|金额|价格|状态|配送时间|花了多少|查一查|查价格')
OTHER = re.compile(r'其他|其它|别的|另一(?:张|笔|单|件)|其余')
VIEW = re.compile(r'(?:看|显示|展示|打开|给|按|点|点击).{0,16}(?:确认|按钮|预览)|(?:确认项|确认卡|确认页面|确认按钮|预览).{0,8}(?:看|显示|展示|打开)')
DEICTIC = re.compile(r'刚才|原来|这(?:笔|单|张|份|件|个订单|次购买)|那(?:笔|单|张|份|件|个订单)|该(?:订单|购买|交易|商品|笔)|原订单')
CONFIRM = re.compile(r'确认|确定|同意|继续')
FUNCTION = re.compile(r'请帮忙|请给我|帮忙|办一下|走一下|就行|只要|其中|这边|先列出|出来|让我|申请|看看|一下|订单号|订单|购买|交易|刚才|原来|现在|明确|真的|确定|确认|同意|决定|希望|继续|金额|价格|办理|处理|麻烦|请帮我|帮我|给我|选择|选|是|请|我|要|想|这|那|哪|该|一|两|笔|单|张|份|件|个|的|了|吧|吗|能|把|将|替|好的|在聊天里|还是|再|操作|就|已经|退订')
LEFT = re.compile(r'(?:取消|撤销|撤掉|撤单|退订|作废|把|将|的|这|那|该|一|笔|张|份|件|只|款|个|购买|订购|关于|针对|办理|处理|和|及|或|要|留着|保留|留下|查询|查一下|查查|看看|只要)$')
RIGHT = re.compile(r'^(?:的|订单|购买|交易|这|那|本体|本机|办理|进行|申请|请|帮|麻烦|退订|撤单|作废|取消|撤掉|不用|不要|不想|不买|不再|不需要|不取消|不撤销|不退订|先|保留|留下|留着|照常|继续|我|要|想|暂时|已经|目前|应该|需要|可以|能|还|预算|订|重复|买|多|多少|才|是|有|也|和|及|或|、)')


@dataclass(frozen=True)
class Clause:
    text: str
    role: str
    mentions: tuple[dict, ...]
    amounts: tuple[int, ...]
    other: bool = False


def catalog_mentions(text, orders):
    matches=[]
    for row in orders:
        product=row['product']
        # Bounded suffix abbreviations, never arbitrary fuzzy string matching.
        # Boundaries reject "有线耳机" as an alias of "无线耳机". Identical
        # suffixes retain all candidate IDs rather than selecting the first row.
        anchors={product,*ALIASES.get(product,()),*(product[i:] for i in range(1,len(product)-2))}
        for anchor in {row['id'],*anchors}:
            for hit in re.finditer(re.escape(anchor),text):
                left,right=text[:hit.start()],text[hit.end():]
                if anchor==row['id']:
                    if re.search(r'[A-Za-z0-9_]$',left) or re.match(r'[A-Za-z0-9_]',right):continue
                else:
                    if left and re.search(r'[\u4e00-\u9fff]$',left) and not LEFT.search(left):continue
                    if right and re.match(r'[\u4e00-\u9fff]',right) and not RIGHT.search(right):continue
                matches.append(dict(order_id=row['id'],anchor=anchor,start=hit.start(),end=hit.end(),
                                    match='id' if anchor==row['id'] else 'product' if anchor==product else 'bounded_alias'))
    return tuple(m for m in matches if not any(n['start']<=m['start'] and n['end']>=m['end'] and
                  n['end']-n['start']>m['end']-m['start'] for n in matches))


def noun_free(text):
    remainder=MONEY.sub('',text)
    remainder=ACTION.sub('',remainder)
    remainder=CLOSE.sub('',remainder)
    remainder=DEICTIC.sub('',remainder)
    remainder=re.sub(r'先?列出(?:来)?|让(?:我)?(?:选择|挑选)|哪(?:一)?笔','',remainder)
    remainder=FUNCTION.sub('',remainder)
    return not re.sub(r'[\s，,。.!！?？;；:：]','',remainder)


def parse(text,orders):
    parts=[p.strip() for p in re.split(r'[，,。；;！？!?\n]|但是|不过|而是|但',text) if p.strip()]
    purchasing=bool(PURCHASE.search(text))
    clauses=[]
    for part in parts:
        scope=re.sub(r'不需要了|不想要了|不想买了|不打算买了|不买了|不要了|不用了','',part)
        negative=bool(NEGATE.search(scope) or POST_VETO.search(part))
        stop_purchase=bool(re.search(r'(?:订单|购买|交易).{0,8}(?:不要|不用|不再|别).{0,4}继续',part))
        role=('keep' if negative else 'cancel' if stop_purchase else
              'keep' if KEEP.search(part) or re.search(r'(?:订单|购买).{0,4}继续',part) else
              'query' if STATUS_QUESTION.search(part) else
              'cancel' if ACTION.search(part) or (purchasing and CLOSE.search(part)) else
              'display' if VIEW.search(part) else 'query' if QUERY.search(part) else 'mention')
        clauses.append(Clause(part,role,catalog_mentions(part,orders),
                              tuple(sorted({int(Decimal(m.group(1))*100) for m in MONEY.finditer(part)})),bool(OTHER.search(part))))
    return clauses


def analyze(text,previous,orders):
    """Resolve a new utterance using only persisted user-derived frame + catalog."""
    import unicodedata
    text=unicodedata.normalize('NFKC',text).strip()
    clauses=parse(text,orders)
    def frame(kind,ids=(),retained=(),inherited=False,reason='target_needs_clarification'):
        ids=sorted(set(ids));retained=sorted(set(retained))
        return dict(version=VERSION,intent=kind,candidate_order_ids=ids,retained_order_ids=retained,
            unique_target=kind=='cancel_preview' and len(ids)==1,inherited_target=inherited,
            dialogue_state='awaiting_choice' if kind=='cancel_preview' and len(ids)>1 else
                'awaiting_button' if kind=='cancel_preview' and len(ids)==1 else kind,
            next_action='get_order_then_preview_cancel' if kind=='cancel_preview' and len(ids)==1 else 'clarify_or_answer_only',
            reason=reason,authorization_for_write=False,source='user_clauses_and_live_owned_catalog',
            clauses=[dict(text=c.text,role=c.role,mentions=list(c.mentions),amounts=list(c.amounts),other=c.other) for c in clauses])
    if DENIAL.search(text) or DISAVOW.search(text) or QUOTED_OR_HYPOTHETICAL.search(text) or PAUSED_ACTION.search(text):return frame('blocked_scope')
    if POLICY_QUESTION.search(text):return frame('question')
    cancel=[c for c in clauses if c.role=='cancel']
    keep=[c for c in clauses if c.role=='keep']
    retained={m['order_id'] for c in keep for m in c.mentions}
    positive={m['order_id'] for c in cancel for m in c.mentions}
    all_ids={m['order_id'] for c in clauses for m in c.mentions}
    for clause in keep:
        if not clause.mentions and not clause.other and (DEICTIC.search(clause.text) or NEGATE.search(clause.text)):
            retained.update(positive or all_ids or (previous or {}).get('candidate_order_ids',[]))
    amounts={a for c in clauses if c.role!='keep' for a in c.amounts}
    if len(amounts)>1:return frame('ambiguous',reason='target_ambiguous')
    active=previous is not None and previous.get('intent')=='cancel_preview'
    prior_ids=set(previous['candidate_order_ids']) if active else set()
    inherited=False
    if cancel:
        ungrounded=[c for c in cancel if not c.mentions]
        if any(not noun_free(c.text) for c in ungrounded):
            return frame('missing_target',retained=retained,reason='target_not_found')
        subjects={m['order_id'] for c in clauses if c.role=='mention' for m in c.mentions}
        # Only a noun-free cancellation clause can use a separately mentioned
        # subject. Never select another catalog mention when the cancel clause
        # itself contains an unresolved new noun.
        ids=positive or (subjects-retained)
        # Unknown new product names cannot borrow the old request. A deictic
        # alone is insufficient: its cancel clause must contain no new noun.
        if not ids and active and all(noun_free(c.text) for c in cancel):
            ids=prior_ids;inherited=True
        if not ids:return frame('missing_target',retained=retained,reason='target_not_found')
    else:
        if keep:return frame('withdrawn',retained=retained)
        if any(c.role=='query' for c in clauses) and not amounts:return frame('query_or_idle',all_ids)
        reference_only=all(c.role=='display' or noun_free(c.text) for c in clauses)
        if active and amounts and not all_ids:
            reference_only=all(c.role=='display' or noun_free(c.text) for c in clauses)
        if active and (CONFIRM.search(text) or DEICTIC.search(text) or amounts) and reference_only:
            ids=prior_ids;inherited=True
        elif active and all_ids and not any(c.role=='query' for c in clauses):
            remainder=text
            for c in clauses:
                for m in sorted(c.mentions,key=lambda m:len(m['anchor']),reverse=True):remainder=remainder.replace(m['anchor'],'')
            if not noun_free(remainder):return frame('query_or_idle',all_ids)
            ids=prior_ids & all_ids;inherited=True
        else:return frame('query_or_idle',all_ids)
    if ids & retained:return frame('contradictory',retained=retained)
    ids-=retained
    if amounts:ids &= {r['id'] for r in orders if r['paid_cents'] in amounts}
    ids &= {r['id'] for r in orders}
    return frame('cancel_preview',ids,retained,inherited,'target_ambiguous' if len(ids)>1 else 'target_not_found')


def previous_frame(store,customer,conversation,request):
    with store.transaction() as db:
        store.require_owner(db,customer,conversation)
        row=db.execute('SELECT t.status,i.payload FROM turns t LEFT JOIN request_intents i '
            'ON t.conversation_id=i.conversation_id AND t.request_id=i.request_id '
            'WHERE t.conversation_id=? AND t.request_id!=? ORDER BY t.rowid DESC LIMIT 1',(conversation,request)).fetchone()
    if not row or row['status']!='completed' or row['payload'] is None:return None
    payload=json.loads(row['payload'])
    return payload['binding'] if payload.get('profile')==PROFILE else None


def validated_orders(listing,customer):
    rows=listing.get('orders') if isinstance(listing,dict) else None
    fields={'id','product','customer_id','paid_cents','status','version'}
    if not isinstance(rows,list) or len(rows)>1000 or any(
        not isinstance(r,dict) or not fields<=set(r) or r['customer_id']!=customer or
        not isinstance(r['id'],str) or not 1<=len(r['id'])<=128 or
        not isinstance(r['product'],str) or not 1<=len(r['product'])<=300 or
        type(r['paid_cents']) is not int or r['paid_cents']<0 or type(r['version']) is not int or r['version']<1 or
        r['status'] not in {'pending','paid','shipped','cancelled'} for r in rows):
        raise ServiceError('invalid_owned_order_evidence',502)
    if len({r['id'] for r in rows})!=len(rows):raise ServiceError('invalid_owned_order_evidence',502)
    return rows


def prepare(store,customer,conversation,request,text,orders):
    previous=previous_frame(store,customer,conversation,request)
    binding=analyze(text,previous,orders)
    payload=dict(profile=PROFILE,text_sha256=sha256(text.encode()).hexdigest(),previous=previous,binding=binding,
                 catalog=orders)
    with store.transaction() as db:
        store.require_owner(db,customer,conversation)
        db.execute('INSERT INTO request_intents(conversation_id,request_id,payload) VALUES (?,?,?)',
                   (conversation,request,encode(payload)))
    return binding


def current(store,customer,conversation,request):
    with store.transaction() as db:
        store.require_owner(db,customer,conversation)
        row=db.execute('SELECT i.payload,t.text,t.status FROM request_intents i JOIN turns t '
            'ON t.conversation_id=i.conversation_id AND t.request_id=i.request_id '
            'WHERE i.conversation_id=? AND i.request_id=?',(conversation,request)).fetchone()
    if row is None or row['status']!='running':raise ServiceError('turn_not_active',409)
    payload=json.loads(row['payload'])
    if payload['profile']!=PROFILE or payload['text_sha256']!=sha256(row['text'].encode()).hexdigest():
        raise ServiceError('target_needs_clarification',409)
    return payload,row['text']


def require_target(store,backend,customer,conversation,request,order):
    payload,text=current(store,customer,conversation,request)
    rows=validated_orders(backend.list_orders(customer),customer)
    binding=analyze(text,payload['previous'],rows)
    if binding['intent']!='cancel_preview':raise ServiceError(binding['reason'],409)
    ids=binding['candidate_order_ids']
    if not ids:raise ServiceError('target_not_found',409)
    if len(ids)!=1:raise ServiceError('target_ambiguous',409)
    if ids[0]!=order:raise ServiceError('target_order_mismatch',409)
    before=next((r for r in payload['catalog'] if r['id']==order),None)
    live=next(r for r in rows if r['id']==order)
    if before is None or any(before[k]!=live[k] for k in ('version','product','paid_cents','status')):
        raise ServiceError('target_order_changed',409)
    evidence=dict(order_id=order,order_version=live['version'],candidate_count=1,matching_version=VERSION,
                  source='persisted_user_frame_revalidated_against_live_catalog',intent='cancel_preview_only')
    store.record(conversation,request,'target_binding',{'order_id':order},evidence)
    return evidence
