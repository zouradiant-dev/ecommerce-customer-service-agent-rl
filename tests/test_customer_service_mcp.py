import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

from ecommerce_customer_service.serving.api import OnlineServer
from ecommerce_customer_service.serving.backend import MockBusinessBackend
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.mcp_gateway import Gateway
from ecommerce_customer_service.serving.service import CustomerService
from ecommerce_customer_service.serving.storage import Store

try:
    from mcp import ClientSession,StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAVE_MCP=True
except ImportError:HAVE_MCP=False


class GatewayTests(unittest.TestCase):
    def test_external_origins_and_header_injection_are_rejected(self):
        for url in ('http://example.com','https://127.0.0.1','http://user:pass@localhost','http://localhost/?token=secret'):
            with self.assertRaises(ValueError):Gateway(url,'x'*20)
        with self.assertRaises(ValueError):Gateway('http://localhost','x'*20+'\r\n')

    @unittest.skipUnless(HAVE_MCP,'Install optional mcp dependency for actual stdio protocol test')
    def test_real_stdio_discovery_calls_identity_and_denied_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);repo=Path(__file__).resolve().parents[1]
            service=CustomerService(Store(root/'s.sqlite'),MockBusinessBackend(root/'b.sqlite'),
                knowledge=KnowledgeIndex(repo/'data/demo-policies.jsonl'))
            token='private-test-token-123456789'
            server=OnlineServer(('127.0.0.1',0),service,{hashlib.sha256(token.encode()).hexdigest():'customer_alice'})
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            async def run():
                params=StdioServerParameters(command=sys.executable,args=[str(repo/'scripts/run_customer_service_mcp.py'),
                    '--api-base',f'http://127.0.0.1:{server.server_port}'],
                    env={**os.environ,'CUSTOMER_SERVICE_MCP_TOKEN':token})
                async with stdio_client(params) as (read,write):
                    async with ClientSession(read,write) as client:
                        await client.initialize()
                        tools=await client.list_tools()
                        self.assertEqual({'list_orders','get_order','search_policies'},{t.name for t in tools.tools})
                        self.assertFalse(any('customer_id' in t.inputSchema.get('properties',{}) for t in tools.tools))
                        def payload(result):return json.loads(result.content[0].text)
                        own=payload(await client.call_tool('list_orders',{}))
                        self.assertEqual({'order_A','order_B','order_C'},{o['id'] for o in own['orders']})
                        foreign=payload(await client.call_tool('get_order',{'order_id':'order_D'}))
                        self.assertEqual(404,foreign['http_status'])
                        knowledge=payload(await client.call_tool('search_policies',{'query':'退款处理中'}))
                        self.assertEqual('evidence_found',knowledge['status'])
                        denied=await client.call_tool('cancel_order',{'order_id':'order_A'})
                        self.assertTrue(denied.isError)
                        denied=await client.call_tool('list_orders',{'customer_id':'customer_bob'})
                        self.assertTrue(denied.isError)
            try:asyncio.run(run())
            finally:server.shutdown();server.server_close();thread.join(3)
            self.assertEqual('paid',service.backend.get_order('customer_alice','order_A')['order']['status'])


if __name__=='__main__':unittest.main()
