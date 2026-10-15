"""Server-rendered money and owned-order facts; never model arithmetic."""
import re


def yuan(cents):
    if type(cents) is not int or cents<0:raise ValueError('invalid_money_cents')
    return str(cents//100)+'.'+str(cents%100).zfill(2)


def enrich(order):
    return {**order,'paid_amount_yuan':yuan(order['paid_cents']),'money_source':'business_backend_cents'}


def reply(text,binding,orders):
    if binding['intent']=='cancel_preview' and len(binding['candidate_order_ids'])>1:
        selected=set(binding['candidate_order_ids']);prefix='有多笔匹配订单，请按订单号或金额选择一笔：'
    elif binding['intent']=='query_or_idle' and re.search(r'金额|价格|多少钱|花了多少|状态|订单',text):
        selected=set(binding['candidate_order_ids']) or {r['id'] for r in orders};prefix='查询到以下本人订单：'
    else:return None
    names={'paid':'已付款','pending':'待付款','shipped':'已发货','cancelled':'已取消'}
    return prefix+'\n'+'\n'.join(f"- {r['product']}；订单 {r['id']}；已付 {yuan(r['paid_cents'])} 元；{names[r['status']]}"
                                  for r in orders if r['id'] in selected)
