import subprocess
import json
from typing import Dict, Any

class SimpleKubectlExecutor:
    def __init__(self):
        pass
        
    def execute_kubectl_command(self, command: str) -> Dict[str, Any]:
        """Execute kubectl command directly"""
        try:
            # Remove 'kubectl' prefix if present
            if command.startswith('kubectl '):
                cmd = command[8:]  # Remove 'kubectl '
            else:
                cmd = command
                
            # Execute the command
            result = subprocess.run(
                ['kubectl'] + cmd.split(),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'result': result.stdout
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr or 'Command failed'
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Command timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Execution error: {str(e)}'
            }

# Global instance
kubectl_executor = SimpleKubectlExecutor()
