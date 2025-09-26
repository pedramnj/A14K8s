import subprocess
import json
from typing import Dict, Any

class SimpleKubectlExecutor:
    def __init__(self):
        pass
        
    def execute_kubectl_command(self, command: str) -> Dict[str, Any]:
        """Execute kubectl command directly with improved formatting"""
        try:
            # Remove 'kubectl' prefix if present
            if command.startswith('kubectl '):
                cmd = command[8:]  # Remove 'kubectl '
            else:
                cmd = command
            
            # Special handling for common commands
            if cmd.strip() == 'get pods':
                cmd = 'get pods --all-namespaces -o wide'
            elif cmd.strip() == 'get pods --all-namespaces':
                cmd = 'get pods --all-namespaces -o wide'
            elif cmd.strip() == 'get pods -n default':
                cmd = 'get pods -n default'
            elif cmd.strip() == 'get events':
                cmd = 'get events --all-namespaces'
            elif cmd.strip() == 'get events --all-namespaces':
                cmd = 'get events --all-namespaces'
                
            # Execute the command
            result = subprocess.run(
                ['kubectl'] + cmd.split(),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Format the output for better readability
                formatted_output = self._format_kubectl_output(result.stdout, cmd)
                return {
                    'success': True,
                    'result': formatted_output
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
    
    def _format_kubectl_output(self, output: str, command: str) -> str:
        """Format kubectl output for better readability in chat"""
        lines = output.strip().split('\n')
        if not lines:
            return output
            
        # Handle top pods output
        if 'top pods' in command:
            return self._format_top_pods_output(lines)
        
        # Handle get nodes output
        if 'get nodes' in command:
            return self._format_nodes_output(lines)
        
        # Handle get events output
        if 'get events' in command:
            return self._format_events_output(lines)
        
        # Handle get pods output
        if 'get pods' in command:
            if '--all-namespaces' in command:
                return self._format_pods_all_namespaces(lines)
            else:
                return self._format_pods_single_namespace(lines)
        
        # For other commands, return as-is
        return output
    
    def _format_top_pods_output(self, lines: list) -> str:
        """Format top pods output for better readability"""
        if len(lines) < 2:
            return "📊 **No resource usage data available**"
            
        header = lines[0]
        pod_lines = lines[1:]
        
        # Check if this is all-namespaces format (header starts with NAMESPACE)
        has_namespace = header.split()[0].upper().startswith('NAMESPACE')
        
        formatted_lines = []
        formatted_lines.append("📊 **Pod Resource Usage:**\n")
        
        for line in pod_lines:
            if line.strip():
                parts = line.split()
                # Handle both single-namespace and all-namespaces formats
                if len(parts) >= 3:
                    # Check if this is all-namespaces format (header has NAMESPACE column)
                    if has_namespace:
                        # All-namespaces format: NAMESPACE NAME CPU MEMORY
                        if len(parts) >= 4:
                            name = parts[1]
                            cpu = parts[2] 
                            memory = parts[3]
                        else:
                            continue
                    else:
                        # Single namespace format: NAME CPU MEMORY
                        name = parts[0]
                        cpu = parts[1]
                        memory = parts[2]
                    
                    # Add emoji based on resource usage
                    cpu_value = int(cpu.replace('m', '')) if cpu.endswith('m') else int(cpu)
                    memory_value = int(memory.replace('Mi', '')) if memory.endswith('Mi') else int(memory)
                    
                    # Determine usage level
                    if cpu_value > 100 or memory_value > 100:
                        usage_emoji = "🔥"  # High usage
                    elif cpu_value > 50 or memory_value > 50:
                        usage_emoji = "⚠️"  # Medium usage
                    else:
                        usage_emoji = "✅"  # Low usage
                    
                    formatted_lines.append(f"{usage_emoji} **{name}**")
                    formatted_lines.append(f"   CPU: {cpu}")
                    formatted_lines.append(f"   Memory: {memory}\n")
        
        return '\n'.join(formatted_lines)
    
    def _format_nodes_output(self, lines: list) -> str:
        """Format nodes output for better readability"""
        if len(lines) < 2:
            return "🖥️ **No nodes found**"
            
        header = lines[0]
        node_lines = lines[1:]
        
        formatted_lines = []
        formatted_lines.append("🖥️ **Cluster Nodes:**\n")
        
        for line in node_lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    name = parts[0]
                    status = parts[1]
                    roles = parts[2]
                    age = parts[3]
                    version = parts[4]
                    
                    # Add emoji based on status
                    if status == 'Ready':
                        status_emoji = "✅"
                    elif status == 'NotReady':
                        status_emoji = "❌"
                    else:
                        status_emoji = "⚠️"
                    
                    # Format roles
                    if 'control-plane' in roles:
                        role_emoji = "👑"
                        role_text = "Control Plane"
                    elif 'master' in roles:
                        role_emoji = "👑"
                        role_text = "Master"
                    else:
                        role_emoji = "💻"
                        role_text = "Worker"
                    
                    formatted_lines.append(f"{status_emoji} **{name}**")
                    formatted_lines.append(f"   Status: {status}")
                    formatted_lines.append(f"   Role: {role_emoji} {role_text}")
                    formatted_lines.append(f"   Age: {age}")
                    formatted_lines.append(f"   Version: {version}\n")
        
        return '\n'.join(formatted_lines)
    
    def _format_events_output(self, lines: list) -> str:
        """Format events output for better readability"""
        if len(lines) < 2:
            return "📅 **No recent events found**"
            
        header = lines[0]
        event_lines = lines[1:]
        
        formatted_lines = []
        formatted_lines.append("📅 **Recent Events:**\n")
        
        for line in event_lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    namespace = parts[0]
                    last_seen = parts[1]
                    event_type = parts[2]
                    reason = parts[3]
                    object_name = parts[4]
                    message = ' '.join(parts[5:]) if len(parts) > 5 else ''
                    
                    # Add emoji based on event type
                    if event_type == 'Warning':
                        emoji = "⚠️"
                    elif event_type == 'Normal':
                        emoji = "ℹ️"
                    else:
                        emoji = "📝"
                    
                    formatted_lines.append(f"{emoji} **{reason}** in {namespace}")
                    formatted_lines.append(f"   Object: {object_name}")
                    formatted_lines.append(f"   Message: {message}")
                    formatted_lines.append(f"   Last seen: {last_seen}\n")
        
        return '\n'.join(formatted_lines)
    
    def _format_pods_all_namespaces(self, lines: list) -> str:
        """Format pods output for all namespaces with better grouping"""
        if len(lines) < 2:
            return '\n'.join(lines)
            
        header = lines[0]
        pod_lines = lines[1:]
        
        # Group pods by namespace
        namespaces = {}
        for line in pod_lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    namespace = parts[0]
                    name = parts[1]
                    ready = parts[2]
                    status = parts[3]
                    restarts = parts[4]
                    age = parts[5]
                    
                    if namespace not in namespaces:
                        namespaces[namespace] = []
                    namespaces[namespace].append({
                        'name': name,
                        'ready': ready,
                        'status': status,
                        'restarts': restarts,
                        'age': age
                    })
        
        # Format output by namespace
        formatted_lines = []
        formatted_lines.append("📋 **Pods by Namespace:**\n")
        
        for namespace, pods in namespaces.items():
            if namespace == 'default':
                formatted_lines.append(f"**🏠 {namespace.upper()}** (User Applications)")
            elif namespace == 'kube-system':
                formatted_lines.append(f"**⚙️ {namespace.upper()}** (System Components)")
            else:
                formatted_lines.append(f"**📦 {namespace.upper()}**")
            
            for pod in pods:
                status_emoji = "✅" if pod['status'] == 'Running' else "⚠️" if pod['status'] == 'Pending' else "❌"
                formatted_lines.append(f"  {status_emoji} {pod['name']} - {pod['status']} ({pod['ready']}) - {pod['age']}")
            
            formatted_lines.append("")  # Empty line between namespaces
        
        return '\n'.join(formatted_lines)
    
    def _format_pods_single_namespace(self, lines: list) -> str:
        """Format pods output for single namespace"""
        if len(lines) < 2:
            return '\n'.join(lines)
            
        header = lines[0]
        pod_lines = lines[1:]
        
        # Check if this is all-namespaces format (header starts with NAMESPACE)
        has_namespace = header.split()[0].upper().startswith('NAMESPACE')
        
        formatted_lines = []
        formatted_lines.append("📋 **Pods:**\n")
        
        for line in pod_lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    name = parts[0]
                    ready = parts[1]
                    status = parts[2]
                    restarts = parts[3]
                    age = parts[4]
                    
                    status_emoji = "✅" if status == 'Running' else "⚠️" if status == 'Pending' else "❌"
                    formatted_lines.append(f"{status_emoji} **{name}** - {status} ({ready}) - {age}")
        
        return '\n'.join(formatted_lines)

# Global instance
kubectl_executor = SimpleKubectlExecutor()
