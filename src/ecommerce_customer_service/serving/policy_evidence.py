"""Broad concept routing plus separately observable retrieval/filter decisions.

Only the three configured demonstration policy topics are supported. This does
not infer an enterprise policy from a loosely related retrieved document.
"""
import re
from .policy_answer import QUESTION,HANDOFF_REQUEST,extractive_answer

CONCEPTS=(('cancel-preview',re.compile(r'取消|撤单|退订|预览|确认(?:按钮|卡|项|页面)|发货')),
          ('refund-state',re.compile(r'退款|pending|到账|支付机构|资金')),
          ('handoff',re.compile(r'人工|真人|接管|工单|队列')))
UNSUPPORTED=re.compile(r'延保|保修|收费|关税|安装费|运费|退货|换货|保价')


def handoff_requested(text):
    if re.search(r'(?:不要|不用|不必|别|暂不|先不).{0,12}(?:转|接|人工|真人|客服)',text):return False
    if re.search(r'解释|说明|区别|政策|只(?:是|想)?(?:了解|咨询)',text):return False
    return bool(HANDOFF_REQUEST.search(text) or re.search(
        r'(?:转|找|联系|交给|安排|需要|我要|我想|请|麻烦|让).{0,12}(?:人工|真人|客服|工作人员)',text))


def route(text,binding):
    if not QUESTION.search(text):return None
    if HANDOFF_REQUEST.search(text) and not re.search(r'只(?:是|想)?(?:了解|咨询)|含义|解释|说明|区别|不要转|不需要转',text):return None
    if binding['intent']=='cancel_preview':return None
    if binding['intent']=='query_or_idle' and binding['candidate_order_ids'] and not re.search(r'规则|政策|演示|含义|流程',text):return None
    # A specific operation result is live business data, not a FAQ.
    if re.search(r'刚才|已经点击|点过|点了',text) and not re.search(r'演示|规则|流程|含义|说明|区别',text):return None
    if UNSUPPORTED.search(text):return []
    return [name for name,pattern in CONCEPTS if pattern.search(text)]


def select(result,topics):
    hits=[h for h in result['hits'] if h['id'] in topics]
    raw=[h['citation'] for h in result['hits']];selected=[h['citation'] for h in hits]
    diagnostics=dict(topic_ids=topics,raw_citations=raw,selected_citations=selected,
        discarded_citations=[c for c in raw if c not in selected],
        no_evidence_reason='unsupported_topic' if not topics else 'retrieval_miss' if not raw else
                           'requested_topic_not_retrieved' if not hits else None)
    return {**result,'hits':hits,'status':'evidence_found' if hits else 'no_evidence',
            'evidence_diagnostics':diagnostics,'scope':'three_configured_demo_policy_topics',
            'answer_mode':'evidence_extracts','authorization':False}
