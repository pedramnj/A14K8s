#!/usr/bin/env python3
import asyncio
import json
import subprocess
import sys
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPKubernetesClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.available_tools = {}
        self.server_process = None
        
    async def connect_to_server(self, server_script_path: str):
        try:
            server_params = StdioServerParameters(
                command='python3',
                args=[server_script_path],
                env=None
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            
            await self.session.initialize()
            
            response = await self.session.list_tools()
            self.available_tools = {tool.name: tool for tool in response.tools}
            print(f'✅ Connected to MCP server with {len(self.available_tools)} tools')
            return True
            
        except Exception as e:
            print(f'⚠️  Error connecting to MCP server: {e}')
            return False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self.session:
                return {'error': 'Not connected to MCP server'}
                
            result = await self.session.call_tool(tool_name, arguments)
            return {
                'success': True,
                'result': result.content
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def cleanup(self):
        await self.exit_stack.aclose()

# Global MCP client instance
mcp_client = None

async def get_mcp_client():
    global mcp_client
    if mcp_client is None:
        mcp_client = MCPKubernetesClient()
        await mcp_client.connect_to_server('kubernetes_mcp_server.py')
    return mcp_client

async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    client = await get_mcp_client()
    return await client.call_tool(tool_name, arguments)
