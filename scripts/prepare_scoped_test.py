"""Freeze a selected system and audit separately supplied, unexecuted test cases."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import shutil
import tarfile
from urllib.request import urlopen

from evaluate_scoped_workflow import ROOT,sha,write,read_rows,assert_frozen
from prepare_heldout_system_test import audit
from run_heldout_system_test import preflight
from review_scoped_development import validate_gold


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True);p.add_argument('--cases-input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    selection_path=args.development/'reviewed-selection.json'
    if not selection_path.exists():selection_path=args.development/'selection.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    dev_protocol=json.loads((args.development/'protocol.json').read_text(encoding='utf-8'))
    if not selection['gate_passed']:raise ValueError('select_on_development_before_preparing_test')
    assert_frozen(dev_protocol['source_sha256'])
    with urlopen('http://127.0.0.1:8647/healthz',timeout=10) as stream:health=json.load(stream)
    if not health['ready'] or any(health[k]!=dev_protocol['health'][k] for k in ('adapter_sha256','base_revision','model_id','precision')):
        raise ValueError('selected_inference_changed')
    args.output.mkdir(parents=True,exist_ok=False)
    source=dict(dev_protocol['source_sha256'])
    for name in ('prepare_scoped_test.py','run_scoped_test.py','build_scoped_development.py','review_scoped_development.py'):
        path=ROOT/'scripts'/name;source[path.relative_to(ROOT).as_posix()]=sha(path)
    protocol=dict(version='paired_scoped_synthetic_holdout_v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        selected_profile='binding_scoped_v7',baseline_profile='binding_catalog_v6',source_sha256=source,health=health,
        development_selection_sha256=sha(selection_path),development_selection_name=selection_path.name,knowledge_sha256=health['knowledge_index_sha256'],
        temperature=0,max_tokens=512,max_input_tokens=8192,recent_turns=6,trials_per_profile=1,
        criteria={'candidate_case_pass_rate_at_least':.90,'denial_query_failures':0,'ledger_errors':0,'model_output_failures':0,'infrastructure_failures':0},
        same_weights_both_profiles=True,no_training=True,author_knows_project=True,
        independent_human_or_production_distribution=False,scenario_semantics_shared=True,
        model_or_resolver_used_for_test_labels=False,one_execution_per_profile=True,
        test_policy='Do not change runtime from this test; after inspection this corpus becomes consumed regression.')
    write(args.output/'protocol.json',protocol)
    rows=read_rows(args.cases_input);preflight(rows);validate_gold(rows)
    shutil.copy2(args.cases_input,args.output/'cases.jsonl')
    extras=list((ROOT/'outputs').glob('scoped-development-cases-20261015-r*/*.jsonl'))
    overlap=audit(rows,extra_paths=extras);write(args.output/'overlap-audit.json',overlap)
    if overlap['exact_first_turn_collisions'] or overlap['exact_user_turn_collisions']:
        raise ValueError('new_test_has_exact_user_text_reuse')
    manifest=dict(version='scoped_holdout_manifest_v1',cases=60,frozen_before_test_model_calls=True,
        new_test_authored_after_development_selection=True,author_independent=False,semantic_independence_claim=False,
        sha256={name:sha(args.output/name) for name in ('cases.jsonl','protocol.json','overlap-audit.json')})
    write(args.output/'manifest.json',manifest)
    with tarfile.open(args.output/'frozen-source.tar.gz','w:gz') as archive:
        for name in source:archive.add(ROOT/name,arcname=name)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
