from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecommerce_customer_service.local_hf import ToolCallParseError
from ecommerce_customer_service.serving.diagnostics import DiagnosticStore
from ecommerce_customer_service.serving.model import ModelOutputError
from ecommerce_customer_service.serving.inference import InferenceServer
from ecommerce_customer_service.serving.repair_experiment import compare_runs, parameter_delta, training_arguments

ROOT=Path(__file__).resolve().parents[1]


class RepairExperimentTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('safetensors'),'tensor check runs in tokenizer environment')
    def test_weight_gate_rejects_unchanged_tensors_even_with_different_serialization(self):
        import numpy as np
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);tensor=np.array([1.,2.,3.],dtype=np.float32)
            save_file({'lora':tensor},str(root/'before'),metadata={'version':'old'})
            save_file({'lora':tensor},str(root/'same'),metadata={'version':'new'})
            with self.assertRaisesRegex(ValueError,'tensors_did_not_change'): parameter_delta(root/'before',root/'same')
            save_file({'lora':tensor+0.01},str(root/'after'))
            result=parameter_delta(root/'before',root/'after')
            self.assertEqual(1,result['changed_tensors']);self.assertGreater(result['delta_l2_norm'],0)

    def test_training_rejects_test_rows_before_loading_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'test.jsonl'
            for placement in ('top','metadata'):
                row={'id':'forbidden','messages':[{'role':'user','content':'query'},{'role':'assistant','content':'reply'}],
                     'loss_mask':[False,True]}
                row.update({'split':'test'} if placement=='top' else {'metadata':{'split':'test'}})
                path.write_text(json.dumps(row)+'\n',encoding='utf-8')
                process=subprocess.run([sys.executable,str(ROOT/'scripts/train_sft.py'),'--train',str(path),
                    '--output',str(Path(temporary)/'output'),'--validate-data-only'],capture_output=True,text=True)
                self.assertNotEqual(0,process.returncode)
                self.assertIn('test rows must not enter SFT',process.stderr)

    def test_pilot_initializes_original_weights_without_resuming_smoke_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            arguments=training_arguments(root/'data',root/'base',root/'original',root/'pilot',smoke=False)
            self.assertEqual(str(root/'original'),arguments[arguments.index('--init-adapter')+1])
            self.assertEqual('64',arguments[arguments.index('--max-steps')+1])
            self.assertIn('--local-files-only',arguments)
            self.assertNotIn('--resume-from',arguments)
            with self.assertRaisesRegex(ValueError,'cannot_write'):
                training_arguments(root/'data',root/'base',root/'original',root/'original'/'new',smoke=True)

    def runs(self,root):
        controls={'complete':True,'real_model_execution':True,'case_sha256':'fixed','case_ids':['a','b'],
            'context_counter':'tokenizer_chat_template','context_budget':8192,'context_policy':'structured','recent_turns':6,
            'system_prompt_sha256':'same','tools_sha256':'same','source_sha256':{'runtime':'same'},
            'passed_cases':1,'case_pass_rate':0.5,'model_output_failures':1}
        artifact={'base_revision':'pinned','precision':'bf16','template_sha256':'same','vocab_sha256':'same','generation':{'greedy':True}}
        for label,outcomes in (('baseline',(False,True)),('candidate',(True,False))):
            folder=root/label;folder.mkdir()
            (folder/'summary.json').write_text(json.dumps(controls),encoding='utf-8')
            (folder/'artifact.json').write_text(json.dumps({**artifact,'adapter_sha256':label}),encoding='utf-8')
            for name,passed in zip(('a','b'),outcomes):
                path=folder/'cases'/name;path.mkdir(parents=True)
                (path/'result.json').write_text(json.dumps({'passed':passed,'category':'fixture'}),encoding='utf-8')
        return root/'baseline',root/'candidate'

    def test_pairing_reports_regression_and_does_not_promote_by_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            before,after=self.runs(Path(temporary))
            report=compare_runs(before,after)
            self.assertEqual(['a'],report['improved_cases'])
            self.assertEqual(['b'],report['regressed_cases'])
            self.assertFalse(report['automatic_promotion'])
            self.assertFalse(report['old_offline_nine_tool_regression_performed'])

    def test_pairing_rejects_changed_precision_template_or_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            before,after=self.runs(Path(temporary))
            source=json.loads((after/'artifact.json').read_text(encoding='utf-8'))
            for field in ('precision','template_sha256'):
                value={**source,field:'different'}
                (after/'artifact.json').write_text(json.dumps(value),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'nonpaired'): compare_runs(before,after)
            (after/'artifact.json').write_text(json.dumps(source),encoding='utf-8')
            summary=json.loads((after/'summary.json').read_text(encoding='utf-8'));summary['complete']=False
            (after/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'incomplete'): compare_runs(before,after)

    def test_local_evaluation_matches_http_output_failure_classification(self):
        spec=importlib.util.spec_from_file_location('local_eval',ROOT/'scripts/evaluate_online_local_adapter.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        error=ToolCallParseError('unbalanced_tool_call_tags','PRIVATE')
        error.generation_metadata={'local_input_tokens':1026,'local_generated_tokens':22,'local_finish_reason':'stop'}
        def failing(*args): raise error
        with tempfile.TemporaryDirectory() as temporary:
            client=module.LocalChatModel(failing,'fixture',DiagnosticStore(Path(temporary),capture_raw=True))
            with self.assertRaises(ModelOutputError) as caught: client.complete([],[])
            self.assertEqual('invalid_tool_call_output',caught.exception.code)
            self.assertEqual(22,caught.exception.metadata['output_tokens'])
            self.assertNotIn('PRIVATE',str(caught.exception))

    def test_paired_knowledge_comparison_rejects_changed_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            before, after = self.runs(Path(temporary))
            summary = json.loads((after / 'summary.json').read_text(encoding='utf-8'))
            summary['knowledge_index_sha256'] = 'different'
            (after / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'nonpaired_knowledge'):
                compare_runs(before, after)

    def test_paired_comparison_rejects_different_reply_check_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            before, after = self.runs(Path(temporary))
            summary = json.loads((after / 'summary.json').read_text(encoding='utf-8'))
            summary['cancel_reply_check_version'] = 'negated_success_v2'
            (after / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'nonpaired_cancel_reply'):
                compare_runs(before, after)

    def test_completed_report_endpoint_is_read_only_and_uses_existing_auth(self):
        def forbidden(*args): raise AssertionError('report must not generate or train')
        report={'schema_version':'online_repair_pilot_v1','experiment_complete':True,'candidate_promoted':False}
        server=InferenceServer(('127.0.0.1',0),forbidden,api_key='fixture-key',repair_experiment_report=report)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}/v1/repair-experiment'
            with self.assertRaises(HTTPError) as denied: urlopen(url)
            self.assertEqual(401,denied.exception.code);denied.exception.close()
            with urlopen(Request(url,headers={'Authorization':'Bearer fixture-key'})) as response:
                self.assertEqual(report,json.load(response))
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__': unittest.main()
