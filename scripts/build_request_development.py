"""Contrastive v8 development specifications; historical tests are consumed."""
from pathlib import Path
import argparse
import json
import shutil

from build_scoped_development import case,message,checkpoint,confirm
from evaluate_scoped_workflow import ROOT,sha,write,read_rows
from review_scoped_development import validate_gold


def development():
    rows=[]
    def add(category,product,steps,**kwargs):
        row=case(len(rows)+1,category,product,steps,namespace='request-development-20261015-r1',**kwargs)
        row['case_id']='request_dev_'+str(len(rows)+1).zfill(3);rows.append(row);return row
    pairs=[('便携除锈笔','除锈笔备用头'),('木质首饰盒','首饰盒隔板'),
           ('手动磨粉机','磨粉机防尘罩'),('车载逆变电源','逆变电源固定架')]
    for product,second in pairs:
        add('role_contrast',product,[message(f'{product}不再购买，{second}保留。'),confirm()],second=second,cancel='a')
        add('role_contrast',product,[message(f'保留{product}，取消{second}。'),confirm()],second=second,cancel='b')
        add('denial_query',product,[message(f'{second}取消是别人说的，我没有授权。'),checkpoint(0)],second=second)
        add('denial_query',product,[message(f'不要取消{product}，{second}也留着。'),checkpoint(0)],second=second)
    for product,text in (
        ('手持雕刻针','手持雕刻针的订单关掉，购买作废。'),
        ('台式珐琅炉','台式珐琅炉多买了，这单关了吧。'),
        ('便携除胶器','便携除胶器不再购买，订单销掉吧。'),
        ('电动拉花笔','电动拉花笔预算不够，不买了。'),
    ):add('purchase_language',product,[message(text),checkpoint(1),confirm()],cancel='a')
    for product,text in (
        ('台式封蜡炉','我反对取消台式封蜡炉，按原样保留。'),
        ('光学观鸟镜','取消光学观鸟镜只是同事的建议，我还没有决定。'),
        ('手动裁皮器','我想知道手动裁皮器是否已经取消，不要另外申请。'),
        ('电子航海钟','假如我取消电子航海钟，这只是个假设。'),
    ):add('denial_query',product,[message(text),checkpoint(0)])
    add('conversation','桌面雕花仪',[message('桌面雕花仪这单退订。'),checkpoint(1),
        {'kind':'event','name':'restart'},message('我确定要退订该购买，不过我还没按按钮。'),checkpoint(1),confirm()],cancel='a')
    add('conversation','电动磨皮器',[message('电动磨皮器不要了。'),checkpoint(1),
        message('刚才那笔继续取消，请给我新的确认页面。'),checkpoint(1)],pending=1)
    add('conversation','户外天文支架',[message('户外天文支架先申请取消。','old'),checkpoint(1),
        message('现在保留天文支架，取消支架配重块。','new'),confirm('old','confirmation_invalidated'),confirm('new')],second='支架配重块',cancel='b')
    add('conversation','玻璃磨口机',[message('玻璃磨口机购买取消。','old'),checkpoint(1),
        message('撤回取消请求，玻璃磨口机留着。'),confirm('old','confirmation_invalidated'),message('我确认保留这一件。'),checkpoint(0)])
    add('conversation','电池内阻仪',[message('电池内阻仪请撤单。','old'),checkpoint(1),
        message('我只查电池内阻仪的金额，先不处理。'),checkpoint(0),confirm('old','confirmation_invalidated')])
    add('conversation','微型扭矩仪',[message('微型扭矩仪我现在要取消。','old'),checkpoint(1),
        message('改为取消桌面频谱仪这笔。'),checkpoint(0),confirm('old','confirmation_invalidated')])
    add('conversation','陶瓷拉坯刀',[message('陶瓷拉坯刀花了多少钱？'),checkpoint(0),
        message('陶瓷拉坯刀买错了，现在申请取消。'),confirm()],cancel='a')
    add('conversation','便携压痕机',[message('便携压痕机请退订。'),checkpoint(1),
        message('我想看刚才的确认卡。'),checkpoint(1)],pending=1)
    for index in range(4):
        row=add('money_selection','精密称重台'+str(index+1),[message('精密称重台'+str(index+1)+'有两笔，先列出来让我选择取消哪笔。'),checkpoint(0),
            message('取消金额{amount}元那一笔，另外一笔留着。'),confirm()],second='精密称重台'+str(index+1),cancel='b')
        a,b=row['initial_orders'];a['paid_cents']=51302+index;b['paid_cents']=507+index
        row['steps'][2]['text']=row['steps'][2]['text'].replace('{amount}',f"{b['paid_cents']/100:.2f}")
        row['expected']['reply_checks']=[dict(step=0,required=[f"{a['paid_cents']/100:.2f} 元",f"{b['paid_cents']/100:.2f} 元"],forbidden=[])]
    for product,text,citations in (
        ('小型套丝机','演示规则我想了解一下，发货以后能取消吗？',['cancel-preview@1']),
        ('手持磁场计','演示中的退款pending是否能证明资金已经到账？',['refund-state@1']),
        ('台式测厚仪','请讲讲演示里人工队列和真实工单的区别。',['handoff@1']),
        ('便携覆膜机','这套演示有没有关于运费计算的政策？',[]),
    ):add('policy_facts',product,[message(text)],extra={'knowledge_lookup_required':True,'required_citations':citations,'no_knowledge_sources':not citations})
    assert len(rows)==40
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    suites={'new-development':development()}
    old=ROOT/'outputs/scoped-development-cases-20261015-r3'
    for name,filename in [('old-development','new-development'),('consumed-first','consumed-regression'),
                           ('prior-business','prior-business'),('prior-knowledge','prior-knowledge')]:
        suites[name]=read_rows(old/(filename+'.jsonl'))
    rows=read_rows(ROOT/'data/consumed-scoped-heldout-20261015-r1/cases.jsonl')
    for row in rows:row.update(split='dev',evaluation_status='second_test_consumed_now_regression')
    suites['consumed-second']=rows
    for name,rows in suites.items():
        validate_gold(rows)
        (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    write(args.output/'manifest.json',dict(counts={k:len(v) for k,v in suites.items()},
        sha256={p.name:sha(p) for p in args.output.glob('*.jsonl')},tests_consumed=True,no_training=True))
    print(json.dumps({k:len(v) for k,v in suites.items()},indent=2))


if __name__=='__main__':main()
