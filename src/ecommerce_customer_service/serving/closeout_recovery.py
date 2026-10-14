"""Accept only the preserved four-step smoke with the diagnosed double shuffle."""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import json
import math
import random

from .artifacts import file_sha256
from .closeout_experiment import CONFIG, ONLINE_SHA
from .repair_experiment import parameter_delta, rows

OLD_TRAINER_SHA='9a48b1e20e07b0cb1092e4d7c393d64aa881702b37022cfce0c82847cb253d11'
OLD_RUNNER_SHA='4b61f18b1d210a1d2cf47c7273d9787edbf8143a521163247c439fd28eadbe1d'


def order_digest(ids):
    return sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def validate_smoke_metrics(metrics,actual_order,source_rows,online_adapter):
    """All numerical, provenance and no-skips gates remain mandatory."""
    reasons=[]
    if metrics.get('optimizer_steps')!=4: reasons.append('optimizer_steps')
    if metrics.get('train_skipped')!=0 or metrics.get('dev_skipped')!=0: reasons.append('skipped_rows')
    if metrics.get('adapter_initialization')!='existing_adapter' or metrics.get('resume_checkpoint') is not None:
        reasons.append('initialization')
    if Path(metrics.get('initial_adapter','')).resolve()!=online_adapter.resolve(): reasons.append('initial_adapter')
    if set(metrics.get('trainable_parameter_elements_by_dtype',{}))!={'torch.float32'}: reasons.append('trainable_dtype')
    if metrics.get('training_rows_shuffled_once') is not True: reasons.append('missing_shuffle_metadata')
    if metrics.get('train_rows')!=len(source_rows) or len(actual_order.get('ordered_example_ids',[]))!=len(source_rows):
        reasons.append('training_row_count')
    if metrics.get('dev_rows')!=16 or metrics.get('dev_source_rows')!=148: reasons.append('dev_row_count')
    trace=metrics.get('trace') or []
    if len(trace)!=4 or [row.get('step') for row in trace]!=[1,2,3,4]: reasons.append('training_trace')
    if any(not isinstance(row.get(key),(int,float)) or not math.isfinite(row[key])
           for row in trace for key in ('train_loss','gradient_norm')): reasons.append('nonfinite_training')
    if any(not isinstance(metrics.get(key),(int,float)) or not math.isfinite(metrics[key])
           for key in ('initial_dev_loss','final_dev_loss')): reasons.append('nonfinite_dev_loss')
    once=[row['id'] for row in source_rows]
    random.Random(CONFIG['seed']).shuffle(once)
    twice=once.copy();random.Random(CONFIG['seed']).shuffle(twice)
    digest=order_digest(twice)
    if (actual_order.get('ordered_example_ids')!=twice or actual_order.get('order_sha256')!=digest
            or metrics.get('training_order_sha256')!=digest or actual_order.get('seed')!=CONFIG['seed']):
        reasons.append('not_the_diagnosed_double_shuffle')
    if reasons: raise ValueError('saved_smoke_rejected:'+','.join(reasons))
    categories={row['id']:row['category'] for row in source_rows}
    return {'smoke_steps_reused':4,'smoke_order_deviation':'seeded_shuffle_applied_twice',
            'recorded_order_sha256':digest,'planned_pilot_order_sha256':order_digest(once),
            'actual_smoke_prefix_category_rows':dict(Counter(categories[name] for name in twice[:32])),
            'pilot_optimizer_reinitialized':True,'pilot_initializes_from':ONLINE_SHA,
            'smoke_weights_not_used_for_pilot':True,'test_used_for_recovery':False}


def validate_saved_smoke(root,data,output,online_adapter):
    if file_sha256(root/'scripts/train_sft.py')!=OLD_TRAINER_SHA:
        raise ValueError('unexpected_failed_trainer_source_no_overwrite')
    if file_sha256(root/'scripts/run_current_instance_closeout.py')!=OLD_RUNNER_SHA:
        raise ValueError('unexpected_failed_runner_source_no_overwrite')
    failure=json.loads((output/'failure.json').read_text(encoding='utf-8'))
    if (failure.get('phase')!='smoke' or failure.get('exception_type')!='ValueError'
            or failure.get('message')!='training_gate_failed'):
        raise ValueError('unexpected_closeout_failure')
    forbidden=('pilot','candidate_dev','selection.json','sealed-test','sealed-test-claim.json',
               'completed.json','recovery-preflight.json')
    if any((output/name).exists() for name in forbidden): raise ValueError('later_stage_or_recovery_already_exists')
    for suite in ('business','knowledge'):
        baseline=output/(suite+'-baseline')
        report=json.loads((baseline/'summary.json').read_text(encoding='utf-8'))
        artifact=json.loads((baseline/'artifact.json').read_text(encoding='utf-8'))
        cases=rows(data.parent/(suite+'-dev-cases.jsonl'))
        expected_ids=[case['case_id'] for case in cases]
        # Use the runtime serializer for its byte-exact case receipt.
        from .storage import encode
        expected_digest=sha256(encode(cases).encode()).hexdigest()
        if (report.get('complete') is not True or report.get('real_model_execution') is not True
                or report.get('model_infrastructure_failures')!=0
                or report.get('case_ids')!=expected_ids or report.get('case_sha256')!=expected_digest
                or artifact.get('adapter_sha256')!=ONLINE_SHA):
            raise ValueError('cannot_reuse_incomplete_or_mismatched_baseline:'+suite)
        for name,expected in report['source_sha256'].items():
            if Path(name).name!=name or file_sha256(root/'src/ecommerce_customer_service/serving'/name)!=expected:
                raise ValueError('baseline_runtime_changed:'+name)
    smoke=output/'smoke'
    metrics=json.loads((smoke/'metrics.json').read_text(encoding='utf-8'))
    actual_order=json.loads((smoke/'training-order.json').read_text(encoding='utf-8'))
    receipt=validate_smoke_metrics(metrics,actual_order,rows(data/'train.jsonl'),online_adapter)
    delta=parameter_delta(online_adapter/'adapter_model.safetensors',smoke/'adapter_model.safetensors')
    receipt.update(weight_update=delta,smoke_model_sha256=file_sha256(smoke/'adapter_model.safetensors'),
        preserved_failure_sha256=file_sha256(output/'failure.json'),
        preserved_smoke_metrics_sha256=file_sha256(smoke/'metrics.json'),
        preserved_training_order_sha256=file_sha256(smoke/'training-order.json'))
    return receipt
