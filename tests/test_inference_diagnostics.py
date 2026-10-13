from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json
import os
import subprocess
import sys

from ecommerce_customer_service.local_hf import ContextBudgetExceeded, ToolCallParseError
from ecommerce_customer_service.serving.diagnostics import DiagnosticStore, classify_failure
from ecommerce_customer_service.serving.inference import InferenceServer
from ecommerce_customer_service.serving.model import ModelOutputError, ModelServiceError, OpenAIChatModel
from ecommerce_customer_service.serving.online_evaluation import run_suite
from ecommerce_customer_service.serving.online_cases import defaults
from ecommerce_customer_service.serving.context_builder import ContextBuilder


class DiagnosticTests(unittest.TestCase):
    def test_capture_is_off_by_default_and_is_bounded_when_enabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = DiagnosticStore(Path(temporary))
            identifier = store.record(code='invalid_tool_call_output', exception_type='ToolCallParseError', raw='PRIVATE RAW')
            data = store.read(identifier)
            self.assertNotIn('raw_generation', data)
            self.assertIsNone(data['input_tokens'])
            capture = DiagnosticStore(Path(temporary), capture_raw=True)
            identifier = capture.record(code='invalid_tool_call_output', exception_type='ToolCallParseError', raw='x'*20000)
            data = capture.read(identifier)
            self.assertEqual(16384, len(data['raw_generation']))
            self.assertTrue(data['raw_capture_truncated'])

    def test_diagnostic_id_cannot_escape_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                DiagnosticStore(Path(temporary)).read('../secret')

    def test_parser_error_uses_safe_reason_and_truncation_precedes_parser_classification(self):
        error = ToolCallParseError('invalid_tool_call_json:duplicate_json_key:PRIVATE', 'raw')
        self.assertEqual((422, 'invalid_tool_call_output', 'invalid_tool_call_json'), classify_failure(error)[:3])
        error.generation_metadata = {'local_finish_reason':'length'}
        self.assertEqual('generation_truncated', classify_failure(error)[1])
        self.assertEqual('context_budget_exceeded', classify_failure(ContextBudgetExceeded(9000,8192))[1])
        self.assertEqual('cuda_out_of_memory', classify_failure(RuntimeError('CUDA out of memory. PRIVATE'))[1])
        self.assertEqual('inference_failed', classify_failure(RuntimeError('PRIVATE'))[1])

    def test_output_failure_counts_as_failed_sample_and_evaluation_continues(self):
        class BadModel:
            model = 'format-fixture-not-real-llm'
            def complete(self, messages, tools):
                raise ModelOutputError('invalid_tool_call_output',422,{'diagnostic_id':'a'*32})
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_suite(defaults()[:2],Path(temporary)/'run',mode='model',model=BadModel(),context=ContextBuilder())
            self.assertTrue(summary['complete'])
            self.assertEqual(0,summary['passed_cases'])
            self.assertGreater(summary['model_output_failures'],0)
            self.assertEqual(0,summary['model_infrastructure_failures'])

    def test_resource_failure_stops_without_claiming_complete_rate(self):
        class BadModel:
            model = 'resource-fixture-not-real-llm'
            def complete(self, messages, tools):
                raise ModelServiceError('cuda_out_of_memory',503,{})
        for size in (1,2):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temporary:
                summary = run_suite(defaults()[:size],Path(temporary)/'run',mode='model',model=BadModel(),context=ContextBuilder())
                self.assertFalse(summary['complete'])
                self.assertIsNone(summary['case_pass_rate'])


class DiagnosticHttpTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.failure = ToolCallParseError('tool_arguments_must_be_object', '<tool_call>PRIVATE INVALID</tool_call>')
        self.failure.generation_metadata={'local_input_tokens':100,'local_generated_tokens':12,'local_finish_reason':'stop'}
        def generate(*args): raise self.failure
        self.server=InferenceServer(('127.0.0.1',0),generate,api_key='test-key',diagnostics_dir=Path(self.temporary.name),capture_invalid_output=True)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}/v1'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.temporary.cleanup()

    def test_client_classifies_invalid_output_and_raw_requires_authenticated_retrieval(self):
        with self.assertRaises(ModelOutputError) as caught:
            OpenAIChatModel(self.base,'ecommerce-v18-sft','test-key').complete([{'role':'user','content':'query'}],[])
        error=caught.exception
        self.assertEqual('invalid_tool_call_output',error.code)
        self.assertEqual(100,error.metadata['input_tokens'])
        self.assertNotIn('PRIVATE',str(error))
        url=self.base+'/diagnostics/'+error.metadata['diagnostic_id']
        with self.assertRaises(HTTPError) as denied:
            urlopen(url)
        self.assertEqual(401,denied.exception.code)
        denied.exception.close()
        with urlopen(Request(url,headers={'Authorization':'Bearer test-key'})) as response:
            data=json.load(response)
        self.assertIn('PRIVATE INVALID',data['raw_generation'])

    def test_oom_is_infrastructure_error_without_private_exception_text(self):
        self.failure=RuntimeError('CUDA out of memory, PRIVATE')
        with self.assertRaises(ModelServiceError) as caught:
            OpenAIChatModel(self.base,'ecommerce-v18-sft','test-key').complete([{'role':'user','content':'query'}],[])
        self.assertNotIsInstance(caught.exception,ModelOutputError)
        self.assertEqual('cuda_out_of_memory',caught.exception.code)
        self.assertNotIn('PRIVATE',str(caught.exception))

    def test_replay_records_failed_output_but_never_executes_returned_tools(self):
        root=Path(__file__).resolve().parents[1]
        request=Path(self.temporary.name)/'request.json'
        request.write_text(json.dumps({'model':'ecommerce-v18-sft','messages':[{'role':'user','content':'query'}],
            'tools':[],'temperature':0,'max_tokens':512,'stream':False}),encoding='utf-8')
        output=Path(self.temporary.name)/'replay'
        process=subprocess.run([sys.executable,str(root/'scripts/replay_online_failure.py'),
            '--request-json',str(request),'--output',str(output),'--api-base',self.base],
            env={**os.environ,'MODEL_API_KEY':'test-key','PYTHONUTF8':'1'},capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(0,process.returncode,process.stderr)
        self.assertNotIn('PRIVATE INVALID',process.stdout)
        report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        self.assertFalse(report['business_tools_executed'])
        self.assertEqual(1,report['inference_requests'])
        self.assertEqual('invalid_tool_call_output',report['status'])
        self.assertIn('PRIVATE INVALID',(output/'diagnostic.json').read_text(encoding='utf-8'))


if __name__=='__main__': unittest.main()
