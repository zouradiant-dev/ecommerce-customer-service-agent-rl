"""Small synthetic knowledge dev suite; fixtures prove wiring, not LLM quality."""
import json


def cases():
    specifications = [
        ('refund_pending', ['退款 pending 处理中是否代表到账？'], ['refund-state@1']),
        ('shipped_policy', ['已发货订单的取消预览政策是什么？'], ['cancel-preview@1']),
        ('handoff_policy', ['人工接管队列的演示说明是什么？请解释政策，不要转接。'], ['handoff@1']),
        ('pickup_unknown', ['是否支持自提？'], []),
        ('exchange_unknown', ['满月包换期限？'], []),
        ('restart_policy', ['退款 pending 说明是什么？', '已发货订单的取消政策是什么？'],
         ['refund-state@1', 'cancel-preview@1']),
    ]
    output = []
    for name, queries, citations in specifications:
        steps, responses, checks = [], [], []
        for index, query in enumerate(queries):
            if index:
                steps.append({'kind': 'event', 'name': 'restart'})
            checks.append({'step': len(steps), 'required_citations': [citations[index]] if citations else [],
                           'no_sources': not citations})
            steps.append({'kind': 'message', 'text': query})
            responses.extend([
                {'tool_calls': [{'id': 'fixture', 'type': 'function', 'function': {
                    'name': 'search_policies', 'arguments': json.dumps({'query': query}, ensure_ascii=False)}}]},
                {'content': ('这是演示说明 [' + citations[index] + ']') if citations else '未找到证据。'},
            ])
        output.append({'schema_version': 'online_knowledge_dev_v1', 'case_id': 'knowledge_' + name,
            'category': 'knowledge', 'split': 'dev', 'customer_id': 'customer_alice',
            'provenance': 'synthetic_demo_policy_dev_not_merchant_policy',
            'steps': steps, 'smoke_responses': responses,
            'expected': {'order_statuses': {'order_A': 'paid', 'order_B': 'paid', 'order_C': 'shipped', 'order_D': 'paid'},
                         'refund_count': 0, 'cancellation_count': 0, 'pending_count': 0,
                         'conversation_status': 'active', 'knowledge_lookup_required': True,
                         'knowledge_checks': checks}})
    return output
