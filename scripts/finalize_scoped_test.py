"""Review a completed paired test without inference or changes to the system."""
from pathlib import Path
import argparse
import json
import shutil

from evaluate_scoped_workflow import sha,write,read_rows,assert_frozen
from finalize_heldout_system_test import actual_input_audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases-dir',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--consumed-output',type=Path,required=True)
    args=p.parse_args()
    complete=json.loads((args.run/'completed.json').read_text(encoding='utf-8'))
    summary=json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
    protocol=json.loads((args.run/'protocol.json').read_text(encoding='utf-8'))
    if not complete['complete'] or complete['summary_sha256']!=sha(args.run/'summary.json'):
        raise ValueError('completed_unchanged_summary_required')
    assert_frozen(protocol['source_sha256'])
    cases=read_rows(args.cases_dir/'cases.jsonl')
    profiles={}
    for name in ('test-v6','test-v7'):
        stage=args.run/name
        audit=actual_input_audit(args.cases_dir,stage)
        if audit['cross_split_exact_canonical_input_collisions']:
            raise ValueError('actual_input_overlap_found')
        write(stage/'actual-input-audit.json',audit)
        profiles[name]={r['case_id']:r for r in (json.loads(path.read_text(encoding='utf-8'))
                                              for path in (stage/'cases').glob('*/result.json'))}
    args.consumed_output.mkdir(parents=True,exist_ok=False)
    for name in ('cases.jsonl','protocol.json','manifest.json','overlap-audit.json'):
        shutil.copy2(args.cases_dir/name,args.consumed_output/name)
    shutil.copy2(args.run/'summary.json',args.consumed_output/'summary.json')
    for name in profiles:
        shutil.copy2(args.run/name/'bad_cases.jsonl',args.consumed_output/(name+'-bad-cases.jsonl'))
        shutil.copy2(args.run/name/'actual-input-audit.json',args.consumed_output/(name+'-actual-input-audit.json'))
    paired=[dict(case_id=c['case_id'],category=c['category'],
                 baseline=profiles['test-v6'][c['case_id']],candidate=profiles['test-v7'][c['case_id']]) for c in cases]
    (args.consumed_output/'paired-results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in paired),encoding='utf-8')
    baseline,candidate=summary['baseline'],summary['candidate']
    report=['# v7 冻结后的合成留出 Test', '',
        f"同一权重、同一组 60 题：v6 **{baseline['passed_cases']}/60**，v7 **{candidate['passed_cases']}/60**。",
        f"预先登记的整体验收条件：**{'通过' if summary['predefined_criteria_met'] else '未通过'}**。",
        f"配对改善 {len(summary['paired_improvements'])} 题，回退 {len(summary['paired_regressions'])} 题。", '',
        '先完成开发选择，再另行编写和封存这 60 题。每个版本各执行一次，没有从本轮 Test 修系统或重测失败题。',
        '同一项目代理出题，业务状态为模拟、用户后续发言与按钮事件预先指定；不能称独立作者或真实客户分布上的认证。', '',
        '| 切片 | v6 | v7 | 总题数 |','|---|---:|---:|---:|']
    report += [f"| {name} | {baseline['by_category'][name]['passed']} | {count['passed']} | {count['total']} |"
               for name,count in candidate['by_category'].items()]
    report += ['', '## 业务边界与证据', '',
        '- v7 账本审计：'+json.dumps(candidate['ledger_audit'],ensure_ascii=False)+'。',
        f"- v7 否认/查询切片失败 {len(summary['denial_query_failures'])}；模型输出错误 {candidate['model_output_failures']}；基础设施错误 {candidate['infrastructure_failures']}。",
        f"- 实际模型调用：v6 {baseline['model_calls']}，v7 {candidate['model_calls']}。调用不等于题数；每题允许多次工具往返。",
        f"- v7 控制器动作 {candidate['controller_actions']}，其中受限预览重建 {candidate['controller_preview_rebuilds']}；单独计数，不作为模型自主调用能力。",
        '- 所有用户轮与可用历史语料归一化精确重复为 0；两个版本的实际请求监督前缀精确碰撞为 0。这不证明语义或预训练独立。',
        '- 通过预算与哈希核对的远程 BF16 推理使用同一 SFT 权重；提升属于控制器、工作流和证据摘录的系统提升。没有新增 SFT 或 GRPO。',
        '- 自动评分检查终态、检查点、引用和有限短语，不是全回答语义质量评分。随机会话 ID、预览 ID 和时间会改变后续输入，单次贪心解码也不保证整段交互完全确定。',
        '- 本轮 Test 已消费。未来依据这些失败继续改动时，这组题只能作为回归，需要另一份冻结验收。', '',
        '## 全部结果（保留失败）', '', '| ID | 切片 | 首轮发言 | v6 | v7 | v7 失败 |', '|---|---|---|---|---|---|']
    for case,row in zip(cases,paired):
        text=next(s['text'] for s in case['steps'] if s['kind']=='message')
        report.append(f"| {case['case_id']} | {case['category']} | {text.replace('|','/')} | {row['baseline']['passed']} | {row['candidate']['passed']} | {'; '.join(row['candidate']['failures'])} |")
    (args.run/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (args.consumed_output/'README.md').write_text(
        '# 已消费的 v7 配对合成 Test\n\n'
        f"v6 {baseline['passed_cases']}/60；v7 {candidate['passed_cases']}/60；预先登记的整体验收条件：{summary['predefined_criteria_met']}。\n\n"
        '这组数据在开发选型之后单独编写、冻结，每版各一次真实推理。结果已查看；从现在起属于已消费测试/回归证据，不再是未来调参的独立验收。\n\n'
        '同一项目代理出题，模拟后台、静态对话脚本。保留两个版本所有结果及失败；不自动加入训练，也不改写之前的52/60。完整原始调用、SQLite、源代码快照在本机校验归档中。\n',encoding='utf-8')
    receipt=dict(cases_sha256=sha(args.cases_dir/'cases.jsonl'),summary_sha256=sha(args.run/'summary.json'),
        completed_marker_sha256=sha(args.run/'completed.json'),test_consumed=True,model_calls_replayed=0,
        runtime_unchanged_since_freeze=True,no_training=True,production_promoted=False,
        source_result_directory=str(args.run.resolve()),files_sha256={p.name:sha(p) for p in args.consumed_output.iterdir() if p.is_file()})
    write(args.consumed_output/'review-receipt.json',receipt)
    print(json.dumps(dict(baseline_passed=baseline['passed_cases'],candidate_passed=candidate['passed_cases'],
        criteria_passed=summary['predefined_criteria_met'],inputs_audited=True),indent=2))


if __name__=='__main__':main()
