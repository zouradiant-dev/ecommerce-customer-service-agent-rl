"""One-principal stdio MCP gateway. Run the authenticated web service first."""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.mcp_gateway import Gateway,create_server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-base',default='http://127.0.0.1:8766')
    args=parser.parse_args()
    server=create_server(Gateway(args.api_base,os.environ.get('CUSTOMER_SERVICE_MCP_TOKEN','')))
    server.run(transport='stdio')


if __name__=='__main__':main()
