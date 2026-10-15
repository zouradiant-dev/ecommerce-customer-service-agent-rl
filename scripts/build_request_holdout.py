"""Author the third synthetic challenge only after v8 development selection.

Only fixture/opaque-ID bookkeeping is shared. Labels are business specifications,
not outputs from the model, intent resolver, or training episode generator.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse
import json

from build_scoped_development import case,message,checkpoint,confirm
from evaluate_scoped_workflow import sha,write,assert_frozen


def cases():
    rows=[]
    def add(category,product,steps,**kw):
        row=case(len(rows)+1,category,product,steps,split='test',namespace='third-request-holdout-20261015-r1',**kw)
        row['case_id']='request_test_'+str(len(rows)+1).zfill(3)
        row['provenance']='manually_authored_business_spec_after_v8_selection_not_customer_logs'
        rows.append(row);return row
    for product,text in (
        ('电动捆扎机','电动捆扎机买重复了，这件的订单帮我撤销一下。'),
        ('台式清洗槽','台式清洗槽暂时用不上，请取消购买。'),
        ('便携读卡仪','便携读卡仪这张订单现在退订，另一件照常。'),
        ('黄铜香插','黄铜香插请取消订单，我已经有一件了。'),
    ):add('new_catalog',product,[message(text),checkpoint(1),confirm()],cancel='a')
    for product,second,text in (
        ('电动卷线盘','卷线盘防雨罩','电动卷线盘留下，把卷线盘防雨罩取消。'),
        ('手持电阻箱','电阻箱测试线','手持电阻箱我仍要，电阻箱测试线这项撤单。'),
        ('玻璃醒酒壶','醒酒壶清洁球','取消醒酒壶清洁球的订单，玻璃醒酒壶继续寄。'),
        ('高脚花架','花架防滑垫','花架防滑垫这次不要了，高脚花架正常安排。'),
    ):add('new_catalog',product,[message(text),checkpoint(1),confirm()],second=second,cancel='b')
    for product,text in (
        ('线圈绕制器','线圈绕制器我不再购买了，把这次交易结束吧。'),
        ('手动压汁杯','手动压汁杯订错，这单先撤了，我不买了。'),
        ('微型平衡仪','微型平衡仪这笔不继续买了，请替我办一下。'),
        ('陶瓷蒸汽盘','陶瓷蒸汽盘预算改了，订单给我关掉。'),
        ('台式磨钻机','台式磨钻机这次不需要了，购买请作废。'),
        ('木工定位尺','木工定位尺我决定退订，先展示申请让我核对。'),
        ('吸盘搬运器','吸盘搬运器不再购买，这笔订单销掉吧。'),
        ('手摇打孔器','手摇打孔器的交易终止，我以后需要再买。'),
    ):add('language_shift',product,[message(text),checkpoint(1),confirm()],cancel='a')
    for product,text in (
        ('桌面拉丝轮','我没有授权取消桌面拉丝轮，仍按原来的订单安排。'),
        ('迷你电泳槽','迷你电泳槽的退订是同事提出的，我从未同意。'),
        ('便携灯光测量仪','我没让你取消便携灯光测量仪，请保留它。'),
        ('可调焊接夹具','可调焊接夹具请勿撤单，我还需要这一件。'),
        ('超声测厚探头','别急着取消超声测厚探头，先让我考虑清楚。'),
        ('定时滴油器','定时滴油器先不撤销，继续寄给我。'),
        ('小型滤油机','我只查询小型滤油机是不是已经取消，不要创建新申请。'),
        ('电子角度规','取消电子角度规只是他人的意见，不代表本人要求。'),
        ('手动装订夹','关于手动装订夹，我还没决定是否取消，先保持原样。'),
        ('铝制冷却台','我只是转述“取消铝制冷却台”，别把它当成我的指令。'),
        ('便携磁吸台灯','假如便携磁吸台灯取消了会怎样？我只是举例。'),
        ('铜制雕花锤','我反对取消铜制雕花锤，这笔先留着。'),
        ('粉末分装漏斗','粉末分装漏斗不要取消，也不要帮我转人工。'),
        ('电动抛光鼓','请暂停取消电动抛光鼓的操作，我需要重新考虑。'),
    ):add('denial_query',product,[message(text),checkpoint(0)])
    add('target_roles','便携高压气泵',[message('高压气泵我需要，取消气泵收纳架才对。'),confirm()],second='气泵收纳架',cancel='b')
    add('target_roles','硅胶隔热罩',[message('隔热罩订单撤销，罩子固定环保留。'),confirm()],second='罩子固定环',cancel='a')
    row=add('target_roles','电子手轮',[message('电子手轮我买了两笔，只取消一笔；请列出金额让我选。'),checkpoint(0),
        message('选金额{amount}元那张办理取消，另外一张继续寄。'),confirm()],second='电子手轮',cancel='b')
    a,b=row['initial_orders'];a['paid_cents']=64307;b['paid_cents']=912
    row['steps'][2]['text']=row['steps'][2]['text'].replace('{amount}','9.12')
    row['expected']['reply_checks']=[dict(step=0,required=['643.07 元','9.12 元'],forbidden=[])]
    add('target_roles','桌面晶体收音器',[message('订单{b}请帮我取消，桌面晶体收音器留下。'),confirm()],second='收音器天线座',cancel='b')
    add('target_roles','红色防水工具袋',[message('工具袋我想取消，先确认是哪一个。'),checkpoint(0)],second='蓝色防水工具袋')
    add('target_roles','便携吸尘枪',[message('我要求取消的是电动除蜡仪，不是便携吸尘枪，请先核实目录。'),checkpoint(0)])
    add('multi_turn','台式电子负载',[message('台式电子负载帮我申请取消。'),checkpoint(1),
        message('对，就取消刚才那笔，确认页面再发我一下。'),checkpoint(1),confirm()],cancel='a')
    add('multi_turn','便携花边裁剪器',[message('便携花边裁剪器这笔订单退订。'),checkpoint(1),{'kind':'event','name':'restart'},
        message('我确定继续退订该订单，还没有点击确认按钮。'),checkpoint(1)],pending=1)
    add('multi_turn','桌面热压封边器',[message('桌面热压封边器这单取消。','old'),checkpoint(1),
        message('桌面热压封边器还是保留，不取消了。'),checkpoint(0),confirm('old','confirmation_invalidated'),
        message('我确定刚才的保留决定。'),checkpoint(0)])
    add('multi_turn','陶艺压泥板',[message('陶艺压泥板请撤销购买。','old'),checkpoint(1),
        message('压泥板留下，取消压泥板清洁布。','new'),confirm('old','confirmation_invalidated'),confirm('new')],second='压泥板清洁布',cancel='b')
    add('multi_turn','木质线轴架',[message('木质线轴架的订单多少钱？'),checkpoint(0),
        message('那现在取消刚才查的这笔订单。'),checkpoint(1),confirm()],cancel='a')
    add('multi_turn','光纤检查镜',[message('光纤检查镜申请退订。','old'),checkpoint(1),
        message('先只查光纤检查镜现在什么状态，不处理取消。'),checkpoint(0),confirm('old','confirmation_invalidated')])
    add('multi_turn','便携折弯器',[message('便携折弯器这次购买取消。'),checkpoint(1),{'kind':'event','name':'restart'},confirm()],cancel='a')
    add('multi_turn','铝制测量平台',[message('铝制测量平台买重了，申请取消。'),checkpoint(1),
        message('改一下，平台留着，取消平台调平脚。'),confirm()],second='平台调平脚',cancel='b')
    add('multi_turn','电子湿度记录卡',[message('电子湿度记录卡请取消。','old'),checkpoint(1),
        message('要取消的换成无线中继盒那单。'),checkpoint(0),confirm('old','confirmation_invalidated')])
    add('multi_turn','手持压线轮',[message('手持压线轮这张购买请退订。'),checkpoint(1),
        message('刚才那笔我同意取消，请给我看新的确认卡。'),checkpoint(1)],pending=1)
    add('recovery','便携喷砂笔',[message('便携喷砂笔订单帮我取消。'),confirm(),confirm(),{'kind':'query'}],cancel='a')
    add('recovery','光学滤镜转盘',[message('光学滤镜转盘请撤销购买。'),{'kind':'event','name':'lose_ack'},confirm(),
        {'kind':'event','name':'restart'},confirm()],cancel='a')
    add('recovery','桌面电热压板',[message('桌面电热压板我决定取消订单。'),{'kind':'event','name':'expire_confirmation'},
        confirm(error='confirmation_expired')],extra={'operation_status':'expired'})
    row=add('recovery','手动冲压座',[message('手动冲压座这件的订单撤单。'),
        {'kind':'event','name':'change_order','order_id':'{a}','status':'shipped'},confirm()],extra={'operation_status':'failed'})
    row['expected']['order_statuses'][row['initial_orders'][0]['id']]='shipped'
    row=add('recovery','黄铜放大镜架',[message('黄铜放大镜架还未付款，取消购买就行。'),confirm()],cancel='a')
    row['initial_orders'][0].update(status='pending',paid_cents=0);row['expected']['refund_count']=0;row['expected'].pop('refund_status')
    row=add('recovery','台式震动筛',[message('台式震动筛已经发货了，帮我看看还能不能取消。'),checkpoint(0)])
    row['initial_orders'][0]['status']='shipped';row['expected']['order_statuses'][row['initial_orders'][0]['id']]='shipped'
    for product,text,citations in (
        ('微型电动阀','想了解这套演示规则：已经发货的订单可否直接取消？',['cancel-preview@1']),
        ('手持防静电刷','演示里的退款pending代表什么，能证明真实资金到账吗？',['refund-state@1']),
        ('台式清洁风枪','演示人工接管队列的政策请解释一下，暂时不要转接。',['handoff@1']),
        ('小型冷裱机','请解释演示取消确认流程及退款处理中状态，我只了解规则。',['cancel-preview@1','refund-state@1']),
    ):add('knowledge_handoff',product,[message(text)],extra={'knowledge_lookup_required':True,'required_citations':citations})
    add('knowledge_handoff','便携绕纤盒',[message('演示商店对安装费用有什么收费政策？请给出依据。')],extra={'knowledge_lookup_required':True,'no_knowledge_sources':True})
    add('knowledge_handoff','手动压胶筒',[message('演示系统对保修期内换货运费有规定吗？')],extra={'knowledge_lookup_required':True,'no_knowledge_sources':True})
    add('knowledge_handoff','桌面微型砂带架',[message('请安排真人客服接手这次咨询，我需要人工协助。')],extra={'conversation_status':'handed_off'})
    add('knowledge_handoff','黄铜打磨底座',[message('这件事情请转人工，由工作人员继续处理。'),
        {'kind':'message','text':'刚转接后我又想到，要取消黄铜打磨底座。','expected_error':'conversation_handed_off'}],extra={'conversation_status':'handed_off'})
    assert len(rows)==60
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    selection=args.development/'selection.json';selected=json.loads(selection.read_text(encoding='utf-8'))
    if not selected['gate_passed'] or selected['profile']!='binding_state_v8':raise ValueError('v8_development_selection_required')
    protocol=json.loads((args.development/'protocol.json').read_text(encoding='utf-8'));assert_frozen(protocol['source_sha256'])
    args.output.mkdir(parents=True,exist_ok=False);rows=cases()
    (args.output/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    write(args.output/'authorship.json',dict(authored_at=datetime.now(timezone.utc).isoformat(),selection_sha256=sha(selection),
        author_source_sha256=sha(Path(__file__)),counts=dict(Counter(r['category'] for r in rows)),
        author_independent=False,labels_from_model_or_resolver=False,training_episode_generator_used=False))
    print('Authored 60 new specifications after v8 development selection.')


if __name__=='__main__':main()
