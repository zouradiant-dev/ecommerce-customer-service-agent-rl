from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.service import CustomerService
from ecommerce_customer_service.serving.backend import MockBusinessBackend
from ecommerce_customer_service.serving.storage import Store,encode


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.index=KnowledgeIndex(Path(__file__).resolve().parents[1]/'data/demo-policies.jsonl')
    def tearDown(self):self.temp.cleanup()

    def test_retrieval_returns_version_source_and_no_authorization(self):
        result=self.index.search('退款处理中是否到账？',as_of='2026-10-14')
        self.assertEqual('refund-state@1',result['hits'][0]['citation'])
        self.assertFalse(result['authorization']);self.assertTrue(result['hits'][0]['demo_only'])
        self.assertEqual('no_evidence',self.index.search('xyzabc987',as_of='2026-10-14')['status'])
        self.assertEqual('no_evidence',self.index.search('退款',as_of='2025-01-01')['status'])

    def test_conflicting_active_versions_are_rejected(self):
        docs=[dict(self.index.documents[0]),dict(self.index.documents[0])];docs[1]['version']='2'
        path=self.root/'conflicting.jsonl';path.write_text('\n'.join(json.dumps(d) for d in docs),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'overlapping'):KnowledgeIndex(path)

    def test_model_without_citations_gets_evidence_instead_of_unsupported_policy(self):
        responses=[{'tool_calls':[{'function':{'name':'search_policies','arguments':encode({'query':'退款处理中'})}}]},
                   {'content':'保证退款立即到账。'}]
        class Model:
            def complete(self,messages,tools):return responses.pop(0)
        service=CustomerService(Store(self.root/'s.sqlite'),MockBusinessBackend(self.root/'b.sqlite'),Model(),
                                knowledge=self.index,policy_profile='binding_guard_v4')
        conversation=service.create_conversation('customer_alice')['conversation_id']
        result=service.message('customer_alice',conversation,'query','退款处理中是什么意思？')
        self.assertIn('[refund-state@1]',result['message']);self.assertNotIn('保证退款',result['message'])
        self.assertTrue(result['sources']);self.assertIsNone(result['operation'])
        trace=service.telemetry.rows()
        self.assertTrue(any(row['kind']=='retrieval' for row in trace))
        self.assertNotIn('退款处理中',json.dumps(trace,ensure_ascii=False))


if __name__=='__main__':unittest.main()
