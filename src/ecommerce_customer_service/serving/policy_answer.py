"""Bounded routing and extractive answers for the three configured demo policies.

This is an explicit engineering route, not a learned router or universal RAG.
Evidence extracts avoid replacing a source's policy with an unsupported model
paraphrase. Business queries/actions remain on the normal tool loop.
"""
import re

from .intent_target import CANCEL, ORDER_CLOSE

DOMAINS = (
    ('cancel-preview', re.compile(r'预览|确认(?:按钮|卡|页面|操作)|取消(?:流程|规则|政策|规定|条件|步骤|资格)|按钮')),
    ('refund-state', re.compile(r'退款|pending|资金到账|钱.*(?:到|收到)|银行卡')),
    ('handoff', re.compile(r'人工|接管|工单|队列')),
)
QUESTION = re.compile(r'什么|怎么|如何|吗|么|？|\?|是否|含义|意思|解释|说明|了解|区别|等于|规则|流程|价格|收费|费用|保修|延保|依据')
ACTION_REQUEST = re.compile(r'帮我|帮忙|请.{0,8}(?:取消|撤|退订|作废)|(?:取消|撤单|退订|作废).{0,12}(?:订单|这单|购买)|不用了|不要了|不需要了|不买了|终止交易')
ORDER_RESULT = re.compile(r'刚才|已经点击|点过|点了|这笔|这单|我的|订单号')
HANDOFF_REQUEST = re.compile(r'(?:请|希望|我要|我想|可以|能否|能不能|帮我|麻烦).{0,16}(?:人工|真人|工作人员).{0,12}(?:接手|接管|继续处理|处理|协助|沟通)|(?:转给|转接|安排).{0,8}人工')


def route(text):
    if not QUESTION.search(text): return None
    if HANDOFF_REQUEST.search(text) and not re.search(r'只(?:是|想)?(?:了解|咨询)|含义|解释|说明|区别|不要转|不需要转', text): return None
    if (CANCEL.search(text) or ORDER_CLOSE.search(text)) and ACTION_REQUEST.search(text) and not re.search(r'只(?:是|想)?(?:了解|咨询)|不要操作|别.*操作', text):
        return None
    # A particular completed order's status must come from get_order, not a FAQ.
    if ORDER_RESULT.search(text) and not re.search(r'演示|规则|流程|含义|说明|区别', text): return None
    topics = [name for name, pattern in DOMAINS if pattern.search(text)]
    return topics if topics else []


def scoped_result(result, topics):
    hits = [hit for hit in result['hits'] if hit['id'] in topics]
    return {**result, 'hits': hits, 'status': 'evidence_found' if hits else 'no_evidence',
            'scope': 'configured_demo_policy_topics_only', 'answer_mode': 'evidence_extracts',
            'authorization': False}


def extractive_answer(result):
    hits = result['hits']
    if not hits:
        return '没有检索到有效说明，暂时无法给出有依据的政策回答，请联系人工核实。'
    label = '以下为演示说明，不能当作实际商家政策：' if any(hit['demo_only'] for hit in hits) else '检索到以下说明：'
    return label + '\n' + '\n'.join('['+hit['citation']+'] '+hit['text'] for hit in hits)
