import asyncio
import subprocess
import json
from typing import Dict, Any

class MCPKubernetesSyncWrapper:
    def __init__(self):
        self.server_process = None
        
    def execute_kubectl_command(self, command: str) -> Dict[str, Any]:
        try:
            if not self.server_process:
                self.server_process = subprocess.Popen(
                    ['python3', 'kubernetes_mcp_server.py'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            mcp_request = {
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'tools/call',
                'params': {
                    'name': 'execute_kubectl_command',
                    'arguments': {
                        'command': command
                    }
                }
            }
            
            request_json = json.dumps(mcp_request) + '\n'
            self.server_process.stdin.write(request_json)
            self.server_process.stdin.flush()
            
            response_line = self.server_process.stdout.readline()
            if response_line:
                response = json.loads(response_line.strip())
                if 'result' in response:
                    return {
                        'success': True,
                        'result': response['result']
                    }
                else:
                    return {
                        'success': False,
                        'error': response.get('error', 'Unknown error')
                    }
            else:
                return {
                    'success': False,
                    'error': 'No response from MCP server'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'MCP communication error: {str(e)}'
            }
    
    def cleanup(self):
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None

mcp_wrapper = MCPKubernetesSyncWrapper()
