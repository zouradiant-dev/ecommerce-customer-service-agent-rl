"""Audit actual inputs, report failures, and verify a portable results archive."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecommerce_customer_service.training import canonicalize_tool_schemas, normalize_messages


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def canonical_input(messages, tools):
    text = json.dumps({'messages': normalize_messages(messages), 'tools': canonicalize_tool_schemas(tools)},
                      ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def actual_input_audit(cases_dir, run):
    audit = json.loads((cases_dir / 'overlap-audit.json').read_text(encoding='utf-8'))
    indices = defaultdict(set); prefix_count = 0; source_count = 0
    for name, source in audit['source_files'].items():
        path = ROOT / name
        if sha(path) != source['sha256']: raise ValueError('historical_audit_source_changed:' + name)
        file_prefixes = 0
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip(): continue
            row = json.loads(line); messages = row.get('messages')
            if not isinstance(messages, list) or not isinstance(row.get('tools'), list): continue
            mask = row.get('loss_mask')
            for i, message in enumerate(messages):
                if message.get('role') != 'assistant' or (isinstance(mask, list) and len(mask)==len(messages) and not mask[i]): continue
                key = canonical_input(messages[:i], row['tools'])
                indices[key].add(name); prefix_count += 1; file_prefixes += 1
        source_count += bool(file_prefixes)
    collisions = []; calls = 0; per_case = Counter(); groups = []
    raw_requests = (run / 'model-requests.jsonl').read_bytes()
    for line in raw_requests.decode('utf-8').splitlines():
        request = json.loads(line); calls += 1; per_case[request['case_id']] += 1
        if not groups or groups[-1] != request['case_id']: groups.append(request['case_id'])
        key = canonical_input(request['messages'], request['tools'])
        if key in indices: collisions.append({'case_id': request['case_id'], 'input_sha256': key, 'sources': sorted(indices[key])})
    cases = [json.loads(line) for line in (cases_dir / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    if groups != [case['case_id'] for case in cases]: raise ValueError('missing_or_repeated_case_execution_group')
    recovery = run / 'postprocessing-recovery.json'
    if recovery.exists():
        record = json.loads(recovery.read_text(encoding='utf-8'))
        count = record['first_case_already_completed_model_calls']
        original = b''.join(raw_requests.splitlines(keepends=True)[:count])
        if hashlib.sha256(original).hexdigest() != record['request_log_before_resume_sha256']:
            raise ValueError('original_first_case_request_log_changed')
        if per_case[cases[0]['case_id']] != count or record['model_calls_replayed'] != 0:
            raise ValueError('first_case_was_replayed')
    return {'scope': 'available_messages_tools_training_or_dev_assistant_prefixes_vs_all_actual_test_model_calls',
            'historical_files_with_prefixes': source_count, 'historical_prefixes': prefix_count,
            'historical_unique_prefixes': len(indices), 'actual_test_model_calls': calls,
            'actual_calls_by_case': dict(per_case), 'cross_split_exact_canonical_input_collisions': collisions,
            'all_60_case_call_groups_seen_once_in_order': True,
            'original_first_case_calls_hash_verified': recovery.exists(),
            'zero_user_text_collisions_in_preflight': not audit['exact_user_turn_collisions'],
            'model_calls_replayed_after_postprocessing_error': 0,
            'limitations': ['Canonical messages/tools equality, not tokenizer-rendered or embedding semantic equality.',
                            'Different prompts, state, IDs or tool schemas can prevent exact collisions; text audit remains necessary.',
                            'Available local lineage only; absence from pretraining is not verified.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases-dir', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.run / 'summary.json').read_text(encoding='utf-8'))
    complete = json.loads((args.run / 'completed.json').read_text(encoding='utf-8'))
    if not summary['complete'] or not complete['completed']: raise ValueError('incomplete_test_cannot_be_finalized')
    actual = actual_input_audit(args.cases_dir, args.run)
    write(args.run / 'actual-input-audit.json', actual)
    cases = [json.loads(line) for line in (args.cases_dir / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [json.loads((args.run / 'cases' / case['case_id'] / 'result.json').read_text(encoding='utf-8')) for case in cases]
    report = ['# 冻结系统的留出合成 Test（2026-10-15）', '',
        f"严格任务通过：**{summary['passed_cases']}/{summary['evaluated_cases']}**；预先约定的 90% 门槛：**{'通过' if summary['preregistered_criteria_met'] else '未通过'}**。", '',
        '系统固定为 binding_catalog_v6，Qwen2.5-7B + 0c9da9f74732 适配器。没有训练、修改提示词或业务控制器；测试结果不触发自动优化。', '',
        '这是本轮调优之外、单独编写并封存的 60 条合成挑战题，不是独立作者、真实客户或生产流量上的泛化估计。', '',
        '| 切片 | 通过 | 总数 |', '|---|---:|---:|']
    report += [f"| {category} | {counts['passed']} | {counts['total']} |" for category, counts in summary['by_category'].items()]
    report += ['', '## 安全、审计与执行边界', '',
        '- 账本检查：' + json.dumps(summary['ledger_audit'], ensure_ascii=False) + '。',
        f"- 模型输出错误 {summary['model_output_failures']}；基础设施错误 {summary['infrastructure_failures']}。",
        f"- 可用历史数据中的实际监督前缀 {actual['historical_prefixes']} 个；实际测试模型调用 {actual['actual_test_model_calls']} 次；精确规范输入碰撞 {len(actual['cross_split_exact_canonical_input_collisions'])} 个。",
        '- 去重包括归一化用户文本、字符二元组近似和实际请求的规范输入；这些都不等于完整语义去重或预训练污染检查。',
        '- 首条模型执行后，账本记录的相对路径转换失败；保留原始调用、轨迹和数据库，只修复评测记录程序，继续剩余 59 条。第一条模型请求没有重放。',
        '- 每题使用隔离模拟订单库，显式按钮事件由测试脚本触发；本轮不验证浏览器交互、并发压测或真实订单/支付。',
        '- 自动评分检查数据库终态、检查点及有限短语/引用，不等于对所有回答做语义质量评审。',
        '- 测试结果已经查看，这组题从现在起作为已消费测试/回归证据；不得再次用于模型选择后仍称为独立 Test。', '',
        '## 所有案例（包括失败）', '', '| ID | 切片 | 首轮用户输入 | 通过 | 失败 |', '|---|---|---|---|---|']
    for case, row in zip(cases, rows):
        first = next(s['text'] for s in case['steps'] if s['kind']=='message')
        report.append(f"| {case['case_id']} | {case['category']} | {first.replace('|', '/')} | {row['passed']} | {'; '.join(row['failures'])} |")
    (args.run / 'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    args.output.mkdir(parents=True, exist_ok=False)
    payload = args.output / 'verified-payload'; payload.mkdir()
    shutil.copytree(args.cases_dir, payload / 'cases-and-protocol')
    shutil.copytree(args.run, payload / 'results')
    shutil.copytree(ROOT / 'outputs/heldout-system-launch-20261015-r1', payload / 'launch-and-interruption-logs')
    for name in ('prepare_heldout_system_test.py', 'run_heldout_system_test.py', 'finalize_heldout_system_test.py'):
        (payload / 'final-evaluation-scripts').mkdir(exist_ok=True)
        shutil.copy2(ROOT / 'scripts' / name, payload / 'final-evaluation-scripts' / name)
    sqlite_count = 0
    for path in payload.rglob('*.sqlite'):
        db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok': raise ValueError('sqlite_failed:' + str(path))
        db.close(); sqlite_count += 1
    files = {path.relative_to(payload).as_posix(): sha(path) for path in payload.rglob('*') if path.is_file()}
    write(payload / 'files-sha256.json', files)
    archive = args.output / 'heldout-system-test-20261015-r1.tar.gz'
    with tarfile.open(archive, 'w:gz') as target:
        for path in sorted(payload.rglob('*')):
            if path.is_file(): target.add(path, arcname=path.relative_to(payload).as_posix())
    with tarfile.open(archive, 'r:gz') as check:
        for name, expected in files.items():
            if hashlib.sha256(check.extractfile(name).read()).hexdigest() != expected: raise ValueError('archive_mismatch:' + name)
    receipt = {'archive_sha256': sha(archive), 'verified_files': len(files), 'sqlite_databases_verified': sqlite_count,
               'test_cases': len(cases), 'passed_cases': summary['passed_cases'], 'all_raw_results_retained': True,
               'one_model_execution_per_case': True, 'postprocessing_interruption_disclosed': True,
               'weights_unchanged': True, 'weights_or_base_environment_in_archive': False,
               'no_training': True, 'no_production_promotion': True}
    write(args.output / 'verification-receipt.json', receipt)
    (args.output / (archive.name+'.sha256')).write_text(receipt['archive_sha256']+'  '+archive.name+'\n', encoding='ascii')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
