"""Executed synthetic five-tool repair references, separate from held-out cases."""
from hashlib import sha256
from .repair_sft import KINDS, episode
from .online_cases import say, tool

SEED = 20261014
TARGET_KINDS = {'unique_exact', 'change_target', 'chat_confirm'}
REQUESTS = ('{product}不用了，帮我取消。', '那只{product}不想要了，取消订单。')
KNOWLEDGE_QUERIES = {
    'train': (
        ('预览申请和界面确认在演示系统中有何区别？', 'cancel-preview@1'),
        ('请说明演示后台退款 pending 的含义和真实支付的关系。', 'refund-state@1'),
        ('请解释本地待人工队列的说明，不要帮我转人工。', 'handoff@1'),
        ('赠品保修年限？', None),
    ),
    'dev': (
        ('演示后台未发货时的取消预览与按钮授权规则怎么理解？', 'cancel-preview@1'),
        ('演示中的退款处理中和确认到账之间有什么区别？', 'refund-state@1'),
        ('我只想了解演示人工接管队列，不需要转接。', 'handoff@1'),
        ('安装维修费用？', None),
    ),
}
ANSWERS = {
    'cancel-preview@1': '这是演示说明：未发货订单可申请预览，预览不是取消成功；须核对订单和金额后点击界面确认。已发货不能直接取消，订单状态须查询后台。[cancel-preview@1]',
    'refund-state@1': '这是演示说明：退款 pending 表示处理中，尚未确认到账。系统不调用支付机构；结果未知时查询原操作，不能重复提交。[refund-state@1]',
    'handoff@1': '这是演示说明：用户要求人工才进入本地待人工队列；尚未接入企业工单系统，不保证人工实际收到。此次仅解释，没有转接。[handoff@1]',
    None: '未找到这项政策的证据，无法给出有依据的承诺。',
}


def business_case(split, kind, variant):
    case = episode(split, kind, variant, SEED)
    case['case_id'] = case['case_id'].replace('repair_', 'closeout_')
    case['provenance'] = 'new_executed_synthetic_reference_not_customer_logs'
    if split == 'train':
        # Novel requests are within the already declared finite grammar. Do not
        # expand that grammar from sealed test requests.
        first = next(step for step in case['steps'] if step['kind'] == 'message')
        if kind not in {'query_only', 'cross_owner', 'query_then_cancel', 'handoff'}:
            product = case['initial_orders'][0]['product']
            if kind == 'no_match': product = '储物盒'
            if kind == 'unique_alias':
                from .repair_sft import PRODUCTS
                product = dict(PRODUCTS['train'])[product]
            first['text'] = REQUESTS[variant % 2].format(product=product)
    return case


def knowledge_case(split, variant):
    pool = KNOWLEDGE_QUERIES[split]
    query, citation = pool[variant % len(pool)]
    # A question switch must retrieve evidence for the current question again.
    chain = [(query, citation)]
    if variant % len(pool) == 1:
        chain.append(pool[0])
    steps, responses, checks = [], [], []
    for index, (text, cite) in enumerate(chain):
        if index: steps.append({'kind': 'event', 'name': 'restart'})
        checks.append({'step': len(steps), 'required_citations': [cite] if cite else [], 'no_sources': cite is None})
        steps.append({'kind': 'message', 'text': text})
        responses.extend([tool('search_policies', query=text), say(ANSWERS[cite])])
    fingerprint = sha256(f'{SEED}/{split}/knowledge/{variant}'.encode()).hexdigest()[:10]
    order_id = 'EK' + fingerprint.upper()
    return {'case_id': f'closeout_{split}_knowledge_{variant:03d}', 'split': split,
            'category': 'knowledge', 'customer_id': 'customer_alice',
            'initial_orders': [{'id': order_id, 'customer_id': 'customer_alice',
                                'product': '示例商品', 'status': 'paid', 'paid_cents': 8900, 'version': 1}],
            'steps': steps, 'smoke_responses': responses,
            'expected': {'order_statuses': {order_id: 'paid'}, 'refund_count': 0,
                         'cancellation_count': 0, 'pending_count': 0, 'conversation_status': 'active',
                         'knowledge_lookup_required': True, 'knowledge_checks': checks},
            'provenance': 'executed_demo_policy_reference_not_real_merchant_policy'}


def cases(split):
    if split not in {'train', 'dev'}: raise ValueError('no_test_supervision')
    output = []
    for kind in KINDS:
        count = (24 if kind in TARGET_KINDS else 8) if split == 'train' else 2
        output.extend(business_case(split, kind, index) for index in range(count))
    output.extend(knowledge_case(split, index) for index in range(32 if split == 'train' else 8))
    return output
