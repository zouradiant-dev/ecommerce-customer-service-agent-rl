"""Author 60 new synthetic specifications after development selection.

Only ID/fixture bookkeeping is reused. No model, resolver, episode generator or
development outputs determine these labels. The author knows the project; this
is not independent human or production sampling.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import json

from build_scoped_development import case, message, checkpoint, confirm
from evaluate_scoped_workflow import sha, write


def cases():
    rows=[]
    def add(category,product,steps,**kwargs):
        row=case(len(rows)+1,category,product,steps,split='test',
                 namespace='scoped-heldout-after-selection-20261015-r1',**kwargs)
        row['provenance']='manually_authored_after_development_selection_not_customer_logs'
        rows.append(row)
        return row

    # Eight new catalog fixtures; roles and ID ordering vary.
    for product,text in (
        ('行星搅拌机','行星搅拌机重复采购了，请帮我取消这一件的订单。'),
        ('自动喂鱼器','自动喂鱼器这单撤销购买，别的商品保留。'),
        ('便携洗眼器','便携洗眼器那项我决定退订，请处理取消申请。'),
        ('触控绘画屏','触控绘画屏这次先不买了，订单给我取消。'),
    ): add('new_catalog',product,[message(text),checkpoint(1),confirm()],cancel='a')
    for product,second,text in (
        ('香薰扩香器','香薰补充液','香薰补充液不要了，扩香器本体照常寄，取消补充液那笔。'),
        ('迷你投影幕','投影幕收纳筒','取消投影幕收纳筒，迷你投影幕继续要。'),
        ('电子秤砝码','校准托盘','校准托盘买多了，帮我取消；电子秤砝码留下。'),
        ('黄铜浇水壶','长嘴喷壶','黄铜浇水壶我需要，长嘴喷壶才是要撤单的那个。'),
    ): add('new_catalog',product,[message(text),checkpoint(1),confirm()],second=second,cancel='b')

    # Ordinary paraphrases, including expressions beyond explicit "取消".
    for product,text in (
        ('昆虫观察盒','昆虫观察盒这笔我已经不想再买了，麻烦把订单结束。'),
        ('桌面书籍扫描架','桌面书籍扫描架买错型号了，这单关了吧。'),
        ('便携调奶器','便携调奶器这笔购买作废吧，我不需要它了。'),
        ('铜管扩口器','铜管扩口器的交易请终止，这次不买。'),
        ('电动橡皮擦','电动橡皮擦我改主意了，撤掉这次购买。'),
        ('茶叶分装器','茶叶分装器不再购买，订单麻烦销掉。'),
        ('手工编织机','手工编织机我现在要退订，先给我确认一下对应购买。'),
        ('电容测试仪','电容测试仪暂时不用了，把这笔订单撤回吧。'),
    ): add('language_shift',product,[message(text),checkpoint(1),confirm()],cancel='a')

    # Fourteen requests with no cancellation authorization.
    for product,text in (
        ('热转印压烫机','我没叫你取消热转印压烫机，你弄错了；请保留这件商品。'),
        ('便携测色仪','我从来没让你退订便携测色仪，这笔购买继续。'),
        ('蜂蜡熔化锅','取消蜂蜡熔化锅是配送员提的，不代表我的意思。'),
        ('软陶切片机','关于软陶切片机撤单这事，我并没有提出要求，只是核实情况。'),
        ('桌面吸烟仪','朋友发了句“把桌面吸烟仪取消”，我转给你看看，不是我的操作指令。'),
        ('便携织物剃毛器','请勿取消便携织物剃毛器，我仍然等着收货。'),
        ('磁力搅拌台','磁力搅拌台我还没决定要不要取消，先不要办理。'),
        ('多肉补光架','多肉补光架的取消申请别建，我只是问订单现在的情况。'),
        ('激光转速计','查询激光转速计的订单有没有被取消就行，不要替我取消。'),
        ('木工开榫器','举个例子：我想取消木工开榫器。上面只是例句，不要照着操作。'),
        ('胶片晾片架','我不是要取消胶片晾片架，而是想确认预计配送时间。'),
        ('皮具打孔机','我反对把皮具打孔机取消，订单应该保持原状。'),
        ('桌面缠线机','桌面缠线机不用取消，继续按原订单安排。'),
        ('玻璃切圆器','我正在考虑取消玻璃切圆器，但还没有确定，请先不要动作。'),
    ): add('denial_query',product,[message(text),checkpoint(0)])

    # Six grounding / role disambiguation specifications.
    add('target_roles','显微镜载物台',[message('显微镜载物台保留，取消载物台保护盒这笔购买。'),confirm()],second='载物台保护盒',cancel='b')
    add('target_roles','太阳能充电板',[message('充电板支撑架照常发，太阳能充电板的订单撤掉。'),confirm()],second='充电板支撑架',cancel='a')
    add('target_roles','刺绣定位框',[message('刺绣定位框先别取消；定位框收纳袋不用了。'),confirm()],second='定位框收纳袋',cancel='b')
    row=add('target_roles','旋转展示底座',[message('旋转展示底座有两张订单，我只要取消一张，先让我看清是哪笔。'),checkpoint(0),
        message('请办理金额{amount}元那一笔的取消，另一张留下。'),confirm()],second='旋转展示底座',cancel='b')
    row['steps'][2]['text']=row['steps'][2]['text'].replace('{amount}',f"{row['initial_orders'][1]['paid_cents']/100:.2f}")
    add('target_roles','数码夜视仪',[message('请将{b}这张订单取消，我留着数码夜视仪。'),confirm()],second='夜视仪颈带',cancel='b')
    add('target_roles','小型压花机',[message('我想取消的是热风整形枪，但列表里似乎没有，请先核实。'),checkpoint(0)],second='压花模板')

    # Ten composed conversations: query, switch, withdraw, restart, confirmation.
    add('multi_turn','折叠摄影背景板',[message('折叠摄影背景板帮我走取消申请。','old'),checkpoint(1),
        message('刚才那笔我现在确定取消，请给我可点击的新确认项。'),checkpoint(1),confirm()],cancel='a')
    add('multi_turn','滴灌控制阀',[message('滴灌控制阀这张购买请取消。'),checkpoint(1),{'kind':'event','name':'restart'},
        message('还是刚才那笔，我决定继续取消。'),checkpoint(1)],pending=1)
    add('multi_turn','迷你拉坯工具箱',[message('迷你拉坯工具箱不要了，请申请取消。','old'),checkpoint(1),
        message('先收回刚才的取消请求，这套工具我留着。'),checkpoint(0),confirm('old','confirmation_invalidated'),
        message('我确认刚才说的保留安排。'),checkpoint(0)])
    add('multi_turn','硅胶揉面垫',[message('硅胶揉面垫请退订。','old'),checkpoint(1),
        message('讲错了，揉面垫要保留，取消擀面杖支架才对。','new'),confirm('old','confirmation_invalidated'),confirm('new')],second='擀面杖支架',cancel='b')
    add('multi_turn','镜头调焦环',[message('镜头调焦环订单目前是已付款还是已发货？'),checkpoint(0),
        message('既然还没发货，就把镜头调焦环订单取消掉。'),confirm()],cancel='a')
    add('multi_turn','盆景修枝套装',[message('盆景修枝套装撤掉订单。','old'),checkpoint(1),
        message('先放一下取消的事，我只想查盆景修枝套装花了多少钱。'),checkpoint(0),
        confirm('old','confirmation_invalidated'),message('价格看到了，多谢回复。'),checkpoint(0)])
    add('multi_turn','便携电磨笔',[message('便携电磨笔的购买麻烦申请取消。'),checkpoint(1),{'kind':'event','name':'restart'},confirm()],cancel='a')
    add('multi_turn','电动订书器',[message('电动订书器买多了，申请撤单。'),checkpoint(1),
        message('再改一下，保留电动订书器，取消订书钉补充装。'),checkpoint(1),confirm()],second='订书钉补充装',cancel='b')
    add('multi_turn','手持声级计',[message('手持声级计那笔先帮我撤单。','old'),checkpoint(1),
        message('现在要取消的是光照度仪，不是声级计。'),checkpoint(0),confirm('old','confirmation_invalidated')])
    add('multi_turn','黄铜篆刻印床',[message('黄铜篆刻印床这单请退订。','first'),checkpoint(1),
        message('我确定要退订该购买，但还没有按页面按钮。'),checkpoint(1)],pending=1)

    # Six recovery / business-state cases with explicit fault events.
    add('recovery','数码钳形表',[message('数码钳形表的购买我不需要，申请取消。'),confirm(),confirm(),{'kind':'query'}],cancel='a')
    add('recovery','户外滴水计时器',[message('户外滴水计时器这项请退订。'),{'kind':'event','name':'lose_ack'},confirm(),
        {'kind':'event','name':'restart'},confirm()],cancel='a')
    add('recovery','便携压面机',[message('便携压面机订单取消申请给我一下。'),{'kind':'event','name':'expire_confirmation'},
        confirm(error='confirmation_expired')],extra={'operation_status':'expired'})
    row=add('recovery','迷你砂轮机',[message('迷你砂轮机的订单麻烦取消。'),
        {'kind':'event','name':'change_order','order_id':'{a}','status':'shipped'},confirm()],extra={'operation_status':'failed'})
    row['expected']['order_statuses'][row['initial_orders'][0]['id']]='shipped'
    row=add('recovery','指针式罗盘',[message('指针式罗盘还没付款，替我取消这笔购买。'),confirm()],cancel='a')
    row['initial_orders'][0].update(status='pending',paid_cents=0)
    row['expected']['refund_count']=0;row['expected'].pop('refund_status')
    row=add('recovery','显影液量杯',[message('显影液量杯虽然已经发货，我还是想取消，你先核实能不能办。'),checkpoint(0)])
    row['initial_orders'][0]['status']='shipped'
    row['expected']['order_statuses'][row['initial_orders'][0]['id']]='shipped'

    # Eight knowledge / handoff cases. Labels are based on configured demo facts.
    for product,text,citation in (
        ('手持振动分析仪','订单取消的演示流程里，生成预览以后还要按确认按钮吗？','cancel-preview@1'),
        ('桌面抛光轮','这套演示退款显示pending，我可以把它理解成真实钱已退回吗？','refund-state@1'),
        ('线材剥皮钳','演示页面转人工之后，是进入真实公司的客服工单系统吗？','handoff@1'),
        ('空气取样泵','我只想了解演示规则：发货后还能取消吗？先不处理我的订单。','cancel-preview@1'),
    ): add('knowledge_handoff',product,[message(text)],extra={'knowledge_lookup_required':True,'required_citations':[citation]})
    add('knowledge_handoff','水槽过滤篮',[message('想查一下演示商店跨境关税的计算政策，有没有可引用的规定？')],
        extra={'knowledge_lookup_required':True,'no_knowledge_sources':True})
    add('knowledge_handoff','手持风速计',[message('请解释演示取消的确认步骤以及模拟退款状态，先不要帮我取消购买。')],
        extra={'knowledge_lookup_required':True,'required_citations':['cancel-preview@1','refund-state@1']})
    add('knowledge_handoff','车载逆变器',[message('请安排人工客服和我聊一下，我需要真人协助。')],extra={'conversation_status':'handed_off'})
    add('knowledge_handoff','陶艺雕刻刀套装',[message('麻烦将会话交给工作人员，由人工接着处理。'),
        {'kind':'message','text':'交接以后，顺带把陶艺雕刻刀套装取消。','expected_error':'conversation_handed_off'}],
        extra={'conversation_status':'handed_off'})
    assert len(rows)==60
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();selection=args.development/'selection.json'
    if not json.loads(selection.read_text(encoding='utf-8'))['gate_passed']:
        raise ValueError('development_selection_required')
    args.output.mkdir(parents=True,exist_ok=False);rows=cases()
    (args.output/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    write(args.output/'authorship.json',dict(authored_at=datetime.now(timezone.utc).isoformat(),
        development_selection_sha256=sha(selection),author_source_sha256=sha(Path(__file__)),
        counts=dict(Counter(r['category'] for r in rows)),model_or_resolver_used_for_labels=False,
        independent_author=False,customer_logs=False,shared_bookkeeping_only=True))
    print(json.dumps(dict(cases=len(rows),categories=dict(Counter(r['category'] for r in rows))),indent=2))


if __name__=='__main__':main()
