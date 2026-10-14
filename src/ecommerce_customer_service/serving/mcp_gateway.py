"""Read-only MCP gateway to the authenticated local service.

One stdio process belongs to one operator-configured principal. Identity is never
a model tool parameter. No cancellation, preview or confirmation tool is exposed.
"""
import json
import re
from urllib.error import HTTPError
from urllib.parse import urlparse,quote
from urllib.request import Request,build_opener,HTTPRedirectHandler,ProxyHandler


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


class Gateway:
    def __init__(self,base_url,token):
        parsed=urlparse(base_url)
        if parsed.scheme!='http' or parsed.hostname not in {'127.0.0.1','localhost'} or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {'','/'}:
            raise ValueError('mcp_gateway_requires_local_origin')
        if not isinstance(token,str) or len(token)<16 or any(c in token for c in '\r\n'):
            raise ValueError('mcp_gateway_requires_private_token')
        self.base_url=base_url.rstrip('/');self.token=token
        self.opener=build_opener(ProxyHandler({}),NoRedirect())

    def _request(self,path,body=None):
        payload=json.dumps(body,ensure_ascii=False).encode() if body is not None else None
        request=Request(self.base_url+path,data=payload,headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
        try:
            with self.opener.open(request,timeout=15) as response:
                raw=response.read(1024*1024+1)
                if len(raw)>1024*1024:raise ValueError('mcp_upstream_response_too_large')
                return json.loads(raw)
        except HTTPError as error:
            # Never echo requests, tokens or upstream arbitrary error bodies.
            return {'error':'upstream_rejected','http_status':error.code}

    def list_orders(self):return self._request('/api/orders')

    def get_order(self,order_id):
        if not isinstance(order_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',order_id):
            return {'error':'invalid_order_id'}
        return self._request('/api/orders/'+quote(order_id,safe=''))

    def search_policies(self,query):
        if not isinstance(query,str) or not query.strip() or len(query)>1000:return {'error':'invalid_query'}
        return self._request('/api/knowledge/search',{'query':query})


def create_server(gateway):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
    class StrictServer(FastMCP):
        async def call_tool(self,name,arguments):
            fields={'list_orders':set(),'get_order':{'order_id'},'search_policies':{'query'}}
            if name not in fields or not isinstance(arguments,dict) or set(arguments)!=fields[name]:
                raise ValueError('tool_not_allowed_or_invalid_arguments')
            return await super().call_tool(name,arguments)

        async def list_tools(self):
            tools=await super().list_tools()
            for tool in tools:tool.inputSchema['additionalProperties']=False
            return tools

    server=StrictServer('ecommerce-read-only')
    hints=ToolAnnotations(readOnlyHint=True,destructiveHint=False,openWorldHint=False)

    @server.tool(annotations=hints)
    def list_orders()->dict:
        """查询此 MCP 进程绑定账号的订单，不接受 customer_id。"""
        return gateway.list_orders()

    @server.tool(annotations=hints)
    def get_order(order_id:str)->dict:
        """查询本人订单，其他账号订单由后台拒绝。"""
        return gateway.get_order(order_id)

    @server.tool(annotations=hints)
    def search_policies(query:str)->dict:
        """检索有效知识版本和来源。知识不授权业务操作。"""
        return gateway.search_policies(query)

    return server
