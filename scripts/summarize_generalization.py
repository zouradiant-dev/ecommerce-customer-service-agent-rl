"""Summarize saved, completed controller comparisons without model calls."""
from pathlib import Path
import argparse
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/grounded-generalization-real-20261015-r1')
    args=p.parse_args()
    complete=json.loads((args.output/'completed.json').read_text(encoding='utf-8'))
    names=('new-dev-binding_guard_v4','new-dev-binding_grounded_v5','known-regression-binding_grounded_v5',
           'prior-business-dev-binding_grounded_v5','acceptance-binding_grounded_v5')
    summaries={}
    failed={}
    for name in names:
        directory=args.output/name
        if not (directory/'summary.json').exists():continue
        report=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        summaries[name]={key:report[key] for key in ('passed_cases','evaluated_cases','case_pass_rate','model_calls',
                'tool_calls','tool_errors','model_output_failures','model_infrastructure_failures','complete')}
        failed[name]=[json.loads(line) for line in (directory/'bad_cases.jsonl').read_text(encoding='utf-8').splitlines()]
    result={'schema_version':'grounded_generalization_final_summary_v1','no_training':True,
            'old_test_now_known_regression':True,'model_identity_unchanged':True,
            'independent_human_distribution_claim':False,'production_promoted':False,
            'completed':complete,'summaries':summaries,'failed_cases':failed}
    (args.output/'final-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 固定模型的系统目标识别改进结果','',
           '同一 SFT 适配器、BF16 贪心推理、提示词和五工具集合；本轮未训练。', '',
           '| 集合与配置 | 通过 | 生成格式错误 | 推理基础设施错误 |', '|---|---:|---:|---:|']
    for name,report in summaries.items():
        lines.append(f"| {name} | {report['passed_cases']}/{report['evaluated_cases']} | {report['model_output_failures']} | {report['model_infrastructure_failures']} |")
    lines+=['','旧 30 题已知且参与工程诊断，只作回归。新验收是作者已知模板的词法/商品组合留出，存在共享协议短语，不是独立人工流量测试。',
            '解析器仍受有限词汇与别名约束；未支持的表达应澄清。实际写入仍须按钮确认；所有业务数据均为模拟。', '', '## 未通过案例','']
    for name,rows in failed.items():
        if 'binding_guard_v4' in name:continue
        for row in rows:lines.append('- '+name+' / '+row['case_id']+': '+', '.join(row['failures']))
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
