"""Separate v7 development compositions; the consumed test stays regression data."""
from pathlib import Path
from hashlib import sha256
import argparse
import json

ROOT=Path(__file__).resolve().parents[1]


def message(text, save='latest'): return {'kind':'message','text':text,'save_as':save}
def checkpoint(n): return {'kind':'checkpoint','assertions':{'pending_count':n,'cancellation_count':0}}
def confirm(ref='latest', error=None): return {'kind':'confirm','ref':ref,**({'expected_error':error} if error else {})}


def case(number, category, product, steps, *, second='木质笔筒', cancel=None, pending=0, extra=None, split='dev', namespace='scoped-development-v1'):
    # Opaque IDs, independent of labels; target is not always first in listing order.
    ids={key:'OX'+sha256(f'{namespace}/{number}/{key}'.encode()).hexdigest()[:11].upper() for key in ('a','b')}
    def replace(value):
        if isinstance(value,str): return value.replace('{a}',ids['a']).replace('{b}',ids['b'])
        if isinstance(value,list): return [replace(v) for v in value]
        if isinstance(value,dict): return {replace(k):replace(v) for k,v in value.items()}
        return value
    rows=[dict(id=ids['a'],product=product,customer_id='customer_alice',paid_cents=13745+number*113,status='paid',version=2),
          dict(id=ids['b'],product=second,customer_id='customer_alice',paid_cents=48820+number*73,status='paid',version=1)]
    expected=dict(order_statuses={r['id']:('cancelled' if cancel and r['id']==ids[cancel] else 'paid') for r in rows},
        cancellation_count=int(bool(cancel)),refund_count=int(bool(cancel)),pending_count=pending,conversation_status='active',
        forbidden_reply_phrases=['退款已到账','退款已经到账'])
    if cancel: expected['refund_status']='pending'
    expected.update(extra or {})
    return dict(case_id=f'scoped_{split}_{number:03d}',split=split,category=category,customer_id='customer_alice',
                initial_orders=rows,steps=replace(steps),expected=expected,
                provenance='separately_authored_synthetic_workflow_development_not_customer_logs')


def development():
    rows=[]
    def add(*args,**kwargs): rows.append(case(len(rows)+1,*args,**kwargs))
    for product,text in (
        ('光固化打印机','光固化打印机多订了一台，这笔订单关闭吧。'),
        ('桌面切割机','桌面切割机预算不够，不买了。'),
        ('电动磨甲器','电动磨甲器这次不需要了，麻烦处理购买。'),
        ('无线领夹麦克风','无线领夹麦克风的订单请帮我终止交易。'),
        ('壁挂式风扇','壁挂式风扇我想退订，另外那件正常寄。'),
        ('厨房计时器','厨房计时器的购买请撤回，我以后再买。'),
        ('桌面加湿器','桌面加湿器暂时不要了，谢谢。'),
        ('车载充气泵','车载充气泵这笔不想买了，取消申请给我看看。'),
    ): add('purchase_intent',product,[message(text),checkpoint(1),confirm()],cancel='a')
    for product,text in (
        ('手持金属探测器','取消手持金属探测器是同事的想法，我只是来问问。'),
        ('无线路由器','我从未同意取消无线路由器，别替我作决定。'),
        ('电动升降桌','并不是我要退订电动升降桌，商家误会了。'),
        ('便携水质仪','便携水质仪请别取消，我想继续收货。'),
        ('数码显微镜','数码显微镜的撤单先停一下，我还需要它。'),
        ('充电电烙铁','我要打消取消充电电烙铁的念头，这单保留。'),
        ('宠物烘干箱','问你怎么取消宠物烘干箱，并不是让你办。'),
        ('家用制冰机','家用制冰机不用了，这句话只是举例，请勿操作。'),
        ('墙面测距仪','别人说过取消墙面测距仪，这不是我的要求。'),
        ('电子节拍器','请只查询电子节拍器是否已经被取消，不要再创建申请。'),
        ('衣物护理机','衣物护理机的订单先不要动，我还没决定是否取消。'),
        ('手机云台','我只是引用：取消手机云台，请不要照做。'),
    ): add('denial_query_contrast',product,[message(text),checkpoint(0)])
    add('target_roles','迷你洗衣机',[message('只留下洗衣机防尘罩，迷你洗衣机订单不需要了。'),confirm()],second='洗衣机防尘罩',cancel='a')
    add('target_roles','收音机',[message('取消天线延长线那笔，收音机保留。'),confirm()],second='天线延长线',cancel='b')
    add('target_roles','玻璃水壶',[message('玻璃水壶留下，只取消保温底座。'),confirm()],second='保温底座',cancel='b')
    add('target_roles','键帽拔取器',[message('取消蓝色键帽的购买，键帽拔取器不取消。'),confirm()],second='蓝色键帽',cancel='b')
    add('conversation','磁吸补光灯',[message('磁吸补光灯不买了。'),checkpoint(1),
        {'kind':'event','name':'restart'},message('我明确同意取消这笔订单。'),checkpoint(1)],pending=1)
    add('conversation','抽真空收纳机',[message('抽真空收纳机退订吧。'),checkpoint(1),
        message('我确认取消，请先展示确认卡。'),checkpoint(1)],pending=1)
    add('conversation','陶艺转盘',[message('陶艺转盘这一笔请取消。','old'),checkpoint(1),
        message('算了，保留陶艺转盘，暂时不处理。'),confirm('old','confirmation_invalidated'),
        message('我再确认一下。'),checkpoint(0)])
    add('conversation','专业调音器',[message('专业调音器我不买了。','old'),checkpoint(1),
        message('专业调音器先留着，取消乐器连接线。','new'),confirm('old','confirmation_invalidated'),confirm('new')],second='乐器连接线',cancel='b')
    add('conversation','电动打气筒',[message('电动打气筒请撤单。','old'),checkpoint(1),
        message('电动打气筒多少钱？我只查价格。'),checkpoint(0),confirm('old','confirmation_invalidated')])
    add('conversation','门窗报警器',[message('门窗报警器的订单状态麻烦查一查。'),checkpoint(0),
        message('了解了，这笔门窗报警器购买现在取消。'),confirm()],cancel='a')
    add('conversation','智能插座',[message('智能插座这张订单不要了。'),checkpoint(1),
        message('取消蓝牙适配器这一笔，智能插座继续寄。'),confirm()],second='蓝牙适配器',cancel='b')
    add('conversation','竹制鞋架',[message('竹制鞋架请取消。'),checkpoint(1),
        message('竹制鞋架先不取消了。'),message('确认。'),checkpoint(0)])
    for product,text,cite in (
        ('电动开瓶器','演示页面显示预览已生成，这能等于订单已经取消吗？','cancel-preview@1'),
        ('手持温度计','退款 pending 的含义是不是还没有到账？请解释演示状态。','refund-state@1'),
        ('智能水表','演示系统里的本地人工队列和实际客服工单有什么区别？','handoff@1'),
        ('陶瓷磨刀石','如果订单已经发货，演示取消规则是怎样规定的？','cancel-preview@1'),
    ): add('policy_facts',product,[message(text)],extra={'knowledge_lookup_required':True,'required_citations':[cite]})
    add('policy_facts','票据扫描笔',[message('演示商品三年延保的收费标准有明确依据吗？')],extra={'knowledge_lookup_required':True,'no_knowledge_sources':True})
    add('policy_facts','户外折叠梯',[message('请解释演示退款会不会真的对接支付机构。')],extra={'knowledge_lookup_required':True,'required_citations':['refund-state@1']})
    add('handoff','车载冰箱',[message('能否让一位人工客服继续处理我的事情？')],extra={'conversation_status':'handed_off'})
    add('handoff','机械数字时钟',[message('这次请客服工作人员来接手，我需要和真人沟通。')],extra={'conversation_status':'handed_off'})
    assert len(rows)==40
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    suites={'new-development':development()}
    old=json.loads((ROOT/'data/consumed-heldout-system-20261015-r1/summary.json').read_text(encoding='utf-8'))
    consumed=[json.loads(line) for line in (ROOT/'data/consumed-heldout-system-20261015-r1/cases.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in consumed: row.update(split='dev',evaluation_status='consumed_test_now_regression')
    suites['consumed-regression']=consumed
    origin=ROOT/'outputs/current-instance-closeout-result-20261014-r1/extracted/closeout-inputs'
    for name,filename in [('prior-business','business-dev-cases.jsonl'),('prior-knowledge','knowledge-dev-cases.jsonl')]:
        suites[name]=[json.loads(line) for line in (origin/filename).read_text(encoding='utf-8').splitlines()]
    for name,rows in suites.items():
        (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows),encoding='utf-8')
    manifest=dict(version='scoped_development_v1',consumed_test_baseline_passed=old['passed_cases'],
        consumed_test_is_not_new_test=True,counts={k:len(v) for k,v in suites.items()},
        sha256={path.name:sha256(path.read_bytes()).hexdigest() for path in args.output.glob('*.jsonl')},
        no_training=True,no_new_test_authored_yet=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
