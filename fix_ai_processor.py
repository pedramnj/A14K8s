import re
import json

# Read the current file
with open('ai_processor.py', 'r') as f:
    content = f.read()

# Find and replace the _process_with_groq_nl method
old_method = '''    def _process_with_groq_nl(self, query: str, system_prompt: str) -> dict:
        """Process query using Groq natural language approach"""
        try:
            # Create a system prompt for Groq to determine tool and args
            groq_system_prompt = (
                "You are a Kubernetes expert. Analyze the user's query and determine:\n"
                "1. What Kubernetes operation they want to perform\n"
                "2. What tool to use and with what parameters\n\n"
                
                "Available tools:\n"
                "- get_pods: List pods (params: namespace)\n"
                "- get_pods: Get pod details (params: name, namespace)\n"
                "- run_container_in_pod: Create pod (params: name, image, namespace)\n"
                "- execute_kubectl: Delete pod (params: name, namespace)\n"
                "- get_pod_logs: Get pod logs (params: name, namespace)\n"
                "- get_pod_top: Get resource usage (params: pod_name, namespace)\n"
                "- exec_into_pod: Execute command (params: name, command, namespace)\n\n"
                
                "Respond with JSON format:\n"
                "{\n"
                '  "tool": "tool_name",\n'
                '  "args": {"param": "value"},\n'
                '  "explanation": "Brief explanation"\n'
                "}\n\n"
                
                "Use intelligent defaults: namespace=default, image=nginx for web pods."
            )
            
            response = self.anthropic.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": groq_system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            ai_response = response.choices[0].message.content
            
            # Try to extract JSON from response
            import json
            import re
            
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                tool_name = parsed.get('tool')
                tool_args = parsed.get('args', {})
                explanation = parsed.get('explanation', '')
                
                # Execute the tool call
                result = self._call_mcp_tool(tool_name, tool_args)
                
                # Post-process with AI for polished response
                if result['success']:
                    polished_result = self._post_process_with_ai(
                        query, tool_name, result['result']
                    )
                    return {
                        'command': f'Groq: {tool_name}',
                        'explanation': polished_result,
                        'ai_processed': True,
                        'tool_results': [{'tool_name': tool_name, 'result': result}],
                        'mcp_result': result
                    }
                else:
                    return {
                        'command': f'Groq: {tool_name}',
                        'explanation': f"❌ **Error executing {tool_name}:** {result['error']}",
                        'ai_processed': True,
                        'tool_results': [{'tool_name': tool_name, 'result': result}],
                        'mcp_result': result
                    }
            else:
                return {
                    'command': 'Groq: parse_error',
                    'explanation': f"❌ **Failed to parse AI response:** {ai_response}",
                    'ai_processed': False,
                    'tool_results': [],
                    'mcp_result': None
                }
                
        except Exception as e:
            return {
                'command': 'Groq: error',
                'explanation': f"❌ **Error in Groq processing:** {str(e)}",
                'ai_processed': False,
                'tool_results': [],
                'mcp_result': None
            }'''

new_method = '''    def _process_with_groq_nl(self, query: str, system_prompt: str) -> dict:
        """Process query using Groq natural language approach with real MCP data"""
        try:
            # Determine the appropriate tool based on query
            tool_name = None
            tool_args = {}
            
            # Simple pattern matching for common queries
            query_lower = query.lower()
            if 'pod' in query_lower and ('list' in query_lower or 'show' in query_lower or 'get' in query_lower):
                tool_name = 'get_pods'
                tool_args = {}
            elif 'pod' in query_lower and 'top' in query_lower:
                tool_name = 'get_pod_top'
                tool_args = {}
            elif 'service' in query_lower and ('list' in query_lower or 'show' in query_lower or 'get' in query_lower):
                tool_name = 'get_services'
                tool_args = {}
            elif 'deployment' in query_lower and ('list' in query_lower or 'show' in query_lower or 'get' in query_lower):
                tool_name = 'get_deployments'
                tool_args = {}
            elif 'node' in query_lower and ('list' in query_lower or 'show' in query_lower or 'get' in query_lower):
                tool_name = 'get_cluster_info'
                tool_args = {}
            else:
                # Default to get_pods for general queries
                tool_name = 'get_pods'
                tool_args = {}
            
            # Execute the tool call to get real data
            result = self._call_mcp_tool(tool_name, tool_args)
            
            if result['success']:
                # Use Groq to process the real data and create a polished response
                groq_system_prompt = (
                    "You are a Kubernetes expert. You will receive real data from a Kubernetes cluster. "\n"
                    "Your job is to analyze this data and create a polished, user-friendly response.\n\n"
                    
                    "**CRITICAL INSTRUCTIONS:**\n"
                    "- Use ONLY the real data provided\n"
                    "- Do NOT generate fake or mock data\n"
                    "- Analyze the actual pod statuses, counts, and details\n"
                    "- Create a clear, informative response based on the real data\n"
                    "- Use emojis and formatting to make it readable\n"
                    "- Be accurate and truthful about what you see\n\n"
                    
                    "**RESPONSE STYLE:**\n"
                    "- Start with a brief summary of what you found\n"
                    "- Use bullet points or sections for clarity\n"
                    "- Use appropriate emojis (✅ for running, ⚠️ for issues, ❌ for failures)\n"
                    "- Group related information logically\n"
                    "- Highlight important statuses and issues\n"
                )
                
                # Create the prompt with real data
                data_prompt = f"User Query: {query}\n\nReal Kubernetes Data: {json.dumps(result['result'], indent=2)}\n\nPlease analyze this real data and provide a polished response."
                
                response = self.anthropic.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": groq_system_prompt},
                        {"role": "user", "content": data_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                
                ai_response = response.choices[0].message.content
                
                return {
                    'command': f'Groq: {tool_name}',
                    'explanation': ai_response,
                    'ai_processed': True,
                    'tool_results': [{'tool_name': tool_name, 'result': result}],
                    'mcp_result': result
                }
            else:
                return {
                    'command': f'Groq: {tool_name}',
                    'explanation': f"❌ **Error executing {tool_name}:** {result['error']}",
                    'ai_processed': True,
                    'tool_results': [{'tool_name': tool_name, 'result': result}],
                    'mcp_result': result
                }
                
        except Exception as e:
            return {
                'command': 'Groq: error',
                'explanation': f"❌ **Error in Groq processing:** {str(e)}",
                'ai_processed': False,
                'tool_results': [],
                'mcp_result': None
            }'''

# Replace the method
content = content.replace(old_method, new_method)

# Write the fixed file
with open('ai_processor.py', 'w') as f:
    f.write(content)

print('✅ AI processor fixed to use real MCP data')
