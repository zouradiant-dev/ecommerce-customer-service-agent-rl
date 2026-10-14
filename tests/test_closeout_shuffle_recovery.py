from pathlib import Path
from hashlib import sha256
from copy import deepcopy
import importlib.util
import json
import random
import unittest

from ecommerce_customer_service.serving.closeout_recovery import validate_smoke_metrics,order_digest

ROOT=Path(__file__).resolve().parents[1]


class ShuffleRecoveryTests(unittest.TestCase):
    def saved_smoke(self):
        source=[{'id':'sample_'+str(i),'category':'knowledge' if i%4==0 else 'business'} for i in range(40)]
        ids=[row['id'] for row in source]
        for _ in range(2): random.Random(20261014).shuffle(ids)
        digest=order_digest(ids)
        initial=ROOT/'fake-initial-online'
        metrics={'optimizer_steps':4,'train_skipped':0,'dev_skipped':0,'adapter_initialization':'existing_adapter',
            'resume_checkpoint':None,'initial_adapter':str(initial),'trainable_parameter_elements_by_dtype':{'torch.float32':10},
            'training_rows_shuffled_once':True,'train_rows':40,'dev_rows':16,'dev_source_rows':148,
            'trace':[{'step':i,'train_loss':.2,'gradient_norm':1.} for i in range(1,5)],
            'initial_dev_loss':.2,'final_dev_loss':.1,'training_order_sha256':digest}
        order={'ordered_example_ids':ids,'order_sha256':digest,'seed':20261014}
        return metrics,order,source,initial

    def test_only_diagnosed_order_deviation_is_accepted(self):
        args=self.saved_smoke()
        receipt=validate_smoke_metrics(*args)
        self.assertEqual(4,receipt['smoke_steps_reused'])
        self.assertTrue(receipt['smoke_weights_not_used_for_pilot'])
        self.assertNotEqual(receipt['recorded_order_sha256'],receipt['planned_pilot_order_sha256'])

    def test_recovery_does_not_relax_steps_skips_dtype_or_numerics(self):
        original,order,source,initial=self.saved_smoke()
        changes=({'optimizer_steps':3},{'train_skipped':1},{'dev_skipped':1},
                 {'trainable_parameter_elements_by_dtype':{'torch.float16':10}},
                 {'final_dev_loss':float('nan')},{'adapter_initialization':'new_lora'})
        for change in changes:
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'saved_smoke_rejected'):
                validate_smoke_metrics({**deepcopy(original),**change},order,source,initial)
        broken=deepcopy(original);broken['trace'][0]['gradient_norm']=float('inf')
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            validate_smoke_metrics(broken,order,source,initial)

    def test_recovery_rejects_unknown_permutation_or_seed(self):
        metrics,order,source,initial=self.saved_smoke()
        bad=deepcopy(order);bad['ordered_example_ids'].reverse()
        with self.assertRaisesRegex(ValueError,'diagnosed_double_shuffle'):
            validate_smoke_metrics(metrics,bad,source,initial)

    @unittest.skipUnless(importlib.util.find_spec('transformers'),'tokenizer integration requires CPU tokenizer environment')
    def test_actual_encoded_training_order_matches_frozen_single_shuffle_and_category_coverage(self):
        from transformers import AutoTokenizer
        spec=importlib.util.spec_from_file_location('fixed_train_sft',ROOT/'scripts/train_sft.py')
        trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
        data=ROOT/'outputs/current-instance-five-tool-sft-20261014-r3'
        plan=json.loads((ROOT/'outputs/current-instance-closeout-inputs-20261014-r1/closeout-plan.json').read_text(encoding='utf-8'))
        source=[json.loads(line) for line in (data/'train.jsonl').read_text(encoding='utf-8').splitlines()]
        dev=[json.loads(line) for line in (data/'dev.jsonl').read_text(encoding='utf-8').splitlines()]
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/v18-sft-verified',local_files_only=True)
        args=trainer.build_parser().parse_args(['--train',str(data/'train.jsonl'),'--output','unused',
            '--seed','20261014','--max-sequence-length','3072','--shuffle-training-rows'])
        encoded,skipped,encoded_dev,dev_skipped=trainer.prepare_training_examples(source,dev,tokenizer,args)
        self.assertEqual((880,0,148,0),(len(encoded),skipped,len(encoded_dev),dev_skipped))
        ids=[row['id'] for row in encoded]
        self.assertEqual(plan['training_order_sha256'],order_digest(ids))
        from collections import Counter
        category={row['id']:row['category'] for row in source}
        self.assertEqual(plan['pilot_prefix_category_rows'],dict(Counter(category[key] for key in ids[:512])))
        # Legacy invocations without the compatibility flag still shuffle once.
        args.shuffle_training_rows=False
        other,_,_,_=trainer.prepare_training_examples(source,[],tokenizer,args)
        self.assertEqual(ids,[row['id'] for row in other])


if __name__=='__main__': unittest.main()
