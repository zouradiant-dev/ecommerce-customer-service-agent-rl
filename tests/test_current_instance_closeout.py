from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from ecommerce_customer_service.serving.closeout_experiment import choose_adapter, claim_test, training_arguments
from ecommerce_customer_service.serving.context_builder import ContextBuilder
from ecommerce_customer_service.serving.closeout_data import knowledge_case
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.online_evaluation import run_suite
from ecommerce_customer_service.serving.repair_sft import execute_reference
from ecommerce_customer_service.serving.inference import InferenceServer
from ecommerce_customer_service.serving.model import OpenAIChatModel
from ecommerce_customer_service.serving.online_cases import tool, say, prepare

ROOT=Path(__file__).resolve().parents[1]


class CurrentInstanceCloseoutTests(unittest.TestCase):
    def test_smoke_and_pilot_both_start_from_saved_online_weights(self):
        adapter=Path('/saved-online')
        for smoke,steps in ((True,'4'),(False,'64')):
            args=training_arguments(Path('/data'),Path('/base'),adapter,Path('/new-output'),smoke=smoke)
            self.assertEqual(str(adapter),args[args.index('--init-adapter')+1])
            self.assertEqual(steps,args[args.index('--max-steps')+1])
            self.assertNotIn('--resume-from',args)
            self.assertIn('--shuffle-training-rows',args)

    def comparisons(self):
        return {suite:{'improved_cases':[],'regressed_cases':[],'candidate_output_failures':0,
                       'baseline_output_failures':0} for suite in ('business','knowledge')}

    def test_selection_uses_task_results_not_loss_and_rejects_any_regression(self):
        reports=self.comparisons()
        self.assertEqual('old-online',choose_adapter(reports)['selected'])
        reports['knowledge']['improved_cases']=['evidence-refresh']
        self.assertEqual('pilot',choose_adapter(reports)['selected'])
        reports['business']['regressed_cases']=['cancel-other-order']
        self.assertEqual('old-online',choose_adapter(reports)['selected'])
        reports['business']['regressed_cases']=[]
        reports['knowledge']['candidate_output_failures']=1
        self.assertEqual('old-online',choose_adapter(reports)['selected'])

    def test_test_claim_is_exclusive_and_requires_frozen_dev_selection(self):
        from ecommerce_customer_service.serving.artifacts import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cases=root/'cases.jsonl'
            cases.write_text('\n'.join(json.dumps({'split':'test','case_id':str(i),'smoke_responses':[]}) for i in range(30)),encoding='utf-8')
            selected={'selected_by':'paired_dev_only','test_used_for_selection':False,'sealed_test_sha256':file_sha256(cases)}
            self.assertEqual(30,len(claim_test(selected,cases,root/'claim.json')))
            with self.assertRaises(FileExistsError): claim_test(selected,cases,root/'claim.json')
            selected['test_used_for_selection']=True
            with self.assertRaisesRegex(ValueError,'frozen_dev'): claim_test(selected,cases,root/'other.json')

    def test_ordinary_dev_entry_still_rejects_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'dev_only'):
                run_suite([{'split':'test','case_id':'sealed'}],Path(tmp)/'out',mode='smoke')

    def test_knowledge_reference_clock_respects_effective_interval_and_supervises_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,result=execute_reference(knowledge_case('train',1),Path(tmp)/'case',
                policy_profile='binding_guard_v4',knowledge=KnowledgeIndex(ROOT/'data/demo-policies.jsonl'))
            self.assertTrue(result['passed'])
            self.assertEqual(4,len(rows))
            calls=[row['messages'][-1]['tool_calls'][0]['function']['name'] for row in rows if row['messages'][-1].get('tool_calls')]
            self.assertEqual(['search_policies','search_policies'],calls)
            self.assertTrue(all(len(row['tools'])==5 for row in rows))

    def test_served_adapter_identity_matches_model_list_request_and_response(self):
        identity={'model_id':'verified-online-fixture','adapter_sha256':'a'*64,'training_protocol':'fixture'}
        def generator(messages,tools,max_tokens):
            return {'role':'assistant','content':'fixture','local_input_tokens':1,'local_generated_tokens':1}
        server=InferenceServer(('127.0.0.1',0),generator,artifact_identity=identity)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            base=f'http://127.0.0.1:{server.server_port}'
            with urlopen(base+'/healthz') as r: health=json.load(r)
            self.assertEqual(identity['adapter_sha256'],health['adapter_sha256'])
            with urlopen(base+'/v1/models') as r: self.assertEqual(identity['model_id'],json.load(r)['data'][0]['id'])
            reply=OpenAIChatModel(base+'/v1',identity['model_id']).complete([{'role':'user','content':'test'}],[])
            self.assertEqual(identity['model_id'],reply['_provider_metadata']['response_model'])
        finally:
            server.shutdown();server.server_close();worker.join()

    def test_browser_api_transport_restart_and_idempotency_with_explicit_fixture(self):
        spec=importlib.util.spec_from_file_location('check_closeout_http',ROOT/'scripts/check_closeout_http.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        responses=[tool('list_orders'),say('已查询，没有操作。')]+prepare('order_A')
        identity={'model_id':'http-integration-fixture','adapter_sha256':'b'*64}
        def generator(messages,tools,max_tokens):
            return {**responses.pop(0),'role':'assistant','local_input_tokens':1,'local_generated_tokens':1}
        server=InferenceServer(('127.0.0.1',0),generator,artifact_identity=identity)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                knowledge=ROOT/'data/demo-policies.jsonl'
                from ecommerce_customer_service.serving.artifacts import file_sha256
                selection=root/'selection.json'
                selection.write_text(json.dumps({'selected_adapter_dir':str(root),'selected_adapter_sha256':identity['adapter_sha256'],
                    'knowledge_sha256':file_sha256(knowledge)}),encoding='utf-8')
                argv=['check','--selection',str(selection),'--knowledge-file',str(knowledge),'--output',str(root/'output'),
                      '--api-base',f'http://127.0.0.1:{server.server_port}/v1']
                with patch.object(sys,'argv',argv),patch.object(module,'verify_online_adapter',return_value=identity),patch.object(module,'load_token_counter',return_value=None):
                    self.assertEqual(0,module.main())
                report=json.loads((root/'output/report.json').read_text(encoding='utf-8'))
                self.assertEqual({'cancellations':1,'refunds':1},report['evidence']['counts'])
                self.assertFalse(report['browser_visual_checked'])
                self.assertFalse(responses)
        finally:
            server.shutdown();server.server_close();worker.join()


if __name__=='__main__': unittest.main()
