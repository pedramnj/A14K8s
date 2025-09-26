#!/usr/bin/env python3
"""
Enhanced AI Processor for AI4K8s with Post-Processing
Implements polished responses by passing MCP results through AI.
"""

import json
import re
import requests
from typing import Dict, List, Any, Optional

class EnhancedAIProcessor:
    """Enhanced AI processor with post-processing for polished responses"""
    
    def __init__(self, mcp_server_url="http://localhost:5002/mcp"):
        self.mcp_server_url = mcp_server_url
        self.available_tools = None
        self.use_ai = False
        self.anthropic = None
        self._load_tools()
        self._initialize_ai()
    
    def _initialize_ai(self):
        """Initialize AI client (Groq primary, Anthropic fallback)"""
        import os
        
        # Try to load .env file from client directory
        self._load_env_file()
        
        # Try Groq first (free)
        if os.getenv('GROQ_API_KEY'):
            try:
                import groq
                self.anthropic = groq.Groq(api_key=os.getenv('GROQ_API_KEY'))
                self.use_ai = True
                self.ai_provider = "groq"
                print("🤖 Groq AI processing enabled (free tier)")
                return
            except ImportError:
                print("⚠️  Groq package not installed, trying Anthropic...")
            except Exception as e:
                print(f"⚠️  Groq initialization failed: {e}, trying Anthropic...")
        
        # Fallback to Anthropic (paid)
        if os.getenv('ANTHROPIC_API_KEY'):
            try:
                from anthropic import Anthropic
                os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
                self.anthropic = Anthropic(max_retries=0)
                self.use_ai = True
                self.ai_provider = "anthropic"
                print("🤖 Anthropic AI processing enabled (paid)")
                return
            except ImportError:
                print("⚠️  Anthropic package not installed")
            except Exception as e:
                print(f"⚠️  Anthropic initialization failed: {e}")
        
        # No AI available
        print("🔧 Regex-only processing (no AI keys)")
        self.ai_provider = None
    
    def _load_env_file(self):
        """Load environment variables from .env file"""
        import os
        
        env_paths = [
            'client/.env',
            '.env',
            os.path.expanduser('~/.anthropic_api_key')
        ]
        
        for env_path in env_paths:
            if os.path.exists(env_path):
                try:
                    with open(env_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key] = value
                    print(f"✅ Loaded environment from {env_path}")
                    return
                except Exception as e:
                    print(f"⚠️  Failed to load {env_path}: {e}")
        
        print("⚠️  No .env file found in expected locations")
    
    def _load_tools(self):
        """Load available MCP tools from the server"""
        try:
            response = requests.post(
                self.mcp_server_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                self.available_tools = data.get('result', {}).get('tools', [])
                print(f"✅ Loaded {len(self.available_tools)} MCP tools")
            else:
                print(f"⚠️  Failed to load MCP tools: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Error loading MCP tools: {e}")
    
    def _get_mcp_tools_for_ai(self) -> List[Dict]:
        """Convert MCP tools to Anthropic format"""
        if not self.available_tools:
            return []
        
        tools = []
        for tool in self.available_tools:
            if tool.get('name') in ['pods_list', 'pods_get', 'pods_run', 'pods_delete', 'pods_log', 'pods_top', 'pods_exec', 'resources_create_or_update']:
                tools.append({
                    "name": tool['name'],
                    "description": tool.get('description', ''),
                    "input_schema": tool.get('inputSchema', {})
                })
        return tools
    
    def _process_with_groq_nl(self, query: str, system_prompt: str) -> dict:
        """Process query using Groq natural language approach"""
        try:
            # Create a system prompt for Groq to determine tool and args
            groq_system_prompt = (
                "You are a Kubernetes expert. Analyze the user's query and determine:\n"
                "1. What Kubernetes operation they want to perform\n"
                "2. What tool to use and with what parameters\n\n"
                
                "Available tools:\n"
                "- pods_list: List pods (params: namespace)\n"
                "- pods_get: Get pod details (params: name, namespace)\n"
                "- pods_run: Create pod (params: name, image, namespace)\n"
                "- pods_delete: Delete pod (params: name, namespace)\n"
                "- pods_log: Get pod logs (params: name, namespace)\n"
                "- pods_top: Get resource usage (params: pod_name, namespace)\n"
                "- pods_exec: Execute command (params: name, command, namespace)\n\n"
                
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
                
                if tool_name:
                    # Execute the tool
                    result = self._call_mcp_tool(tool_name, tool_args)
                    
                    if result['success']:
                        # Post-process the result
                        polished_result = self._post_process_with_ai(query, tool_name, result['result'])
                        
                        return {
                            'command': f'Groq: {tool_name}',
                            'explanation': f"{explanation}\n\n{polished_result}",
                            'ai_processed': True,
                            'tool_results': [{'tool_name': tool_name, 'result': result}],
                            'mcp_result': result
                        }
                    else:
                        return {
                            'command': f'Groq: {tool_name} (failed)',
                            'explanation': f"❌ **Error:** {result['error']}",
                            'ai_processed': True,
                            'tool_results': [],
                            'mcp_result': None
                        }
            
            # If no JSON found, return the AI response directly
            return {
                'command': 'Groq: text_response',
                'explanation': ai_response,
                'ai_processed': True,
                'tool_results': [],
                'mcp_result': None
            }
            
        except Exception as e:
            # Fallback to direct response
            return {
                'command': 'Groq: text_response',
                'explanation': f"Groq processing failed: {str(e)}",
                'ai_processed': True,
                'tool_results': [],
                'mcp_result': None
            }
    
    def _call_mcp_tool(self, tool_name: str, args: Dict) -> Dict:
        """Call MCP tool and return result"""
        try:
            # Create tool call request
            tool_call = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": args
                }
            }
            
            response = requests.post(
                self.mcp_server_url,
                json=tool_call,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('result', {})
                return {
                    'success': True,
                    'result': result
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _post_process_with_ai(self, user_query: str, tool_name: str, raw_result: str) -> str:
        """Post-process MCP results with AI to create polished responses"""
        try:
            # Extract the actual content from MCP result
            if isinstance(raw_result, dict):
                content = raw_result.get('content', [{}])
                if isinstance(content, list) and len(content) > 0:
                    actual_content = content[0].get('text', '')
                else:
                    actual_content = str(raw_result)
            else:
                actual_content = str(raw_result)
            
            # Create a system prompt for post-processing
            system_prompt = (
                "You are a Kubernetes expert assistant. Your job is to take raw Kubernetes command output "
                "and transform it into a polished, user-friendly response that directly answers the user's question.\n\n"
                
                "**CRITICAL INSTRUCTIONS:**\n"
                "- Analyze the user's question to understand what they really want to know\n"
                "- Transform the raw output into a clean, conversational response\n"
                "- Use emojis and formatting to make it readable\n"
                "- Focus on the key information the user asked for\n"
                "- Be concise but informative\n"
                "- Group related information logically\n"
                "- Highlight important statuses and issues\n\n"
                
                "**RESPONSE STYLE:**\n"
                "- Start with a brief summary of what you found\n"
                "- Use bullet points or sections for clarity\n"
                "- Use appropriate emojis (✅ for running, ⚠️ for issues, ❌ for failures)\n"
                "- End with helpful next steps or suggestions\n"
                "- Keep it conversational and helpful\n\n"
                
                "**EXAMPLES:**\n"
                "- For pod lists: Group by namespace, show status with emojis\n"
                "- For pod details: Show key info like status, image, IP, labels clearly\n"
                "- For errors: Explain what went wrong and suggest fixes\n"
                "- For success: Confirm what was done and suggest next steps"
            )
            
            # Create the message for post-processing
            message = f"""User Question: "{user_query}"

Tool Used: {tool_name}

Raw Kubernetes Output:
{actual_content}

Please transform this raw output into a polished, user-friendly response that directly answers the user's question. Focus on what they asked for and make it easy to understand."""
            
            # Call AI for post-processing
            if self.ai_provider == "groq":
                response = self.anthropic.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
            else:
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": message}]
                )
            
            # Extract the polished response
            if self.ai_provider == "groq":
                return response.choices[0].message.content
            else:
                if response.content and len(response.content) > 0:
                    return response.content[0].text
                else:
                    return actual_content
                
        except Exception as e:
            print(f"⚠️  Post-processing failed: {e}")
            # Fallback to basic formatting
            return self._basic_format_response(tool_name, raw_result)
    
    def _basic_format_response(self, tool_name: str, raw_result: str) -> str:
        """Basic fallback formatting when AI post-processing fails"""
        if isinstance(raw_result, dict):
            content = raw_result.get('content', [{}])
            if isinstance(content, list) and len(content) > 0:
                text_content = content[0].get('text', '')
            else:
                text_content = str(raw_result)
        else:
            text_content = str(raw_result)
        
        if tool_name == 'pods_list':
            return f"📋 **Pod List Results:**\n\n{text_content}"
        elif tool_name == 'pods_get':
            return f"📋 **Pod Details:**\n\n{text_content}"
        elif tool_name == 'pods_run':
            return f"✅ **Pod Creation:**\n\n{text_content}"
        else:
            return f"**{tool_name} Result:**\n\n{text_content}"
    
    def _process_with_ai(self, query: str) -> dict:
        """Process query using Anthropic AI with post-processing"""
        try:
            import json
            
            # Get available tools for AI
            available_tools = self._get_mcp_tools_for_ai()
            
            # Create system prompt for tool selection and execution
            system_prompt = (
                "You are an intelligent Kubernetes AI assistant. Your job is to:\n"
                "1. Understand the user's intent\n"
                "2. Execute the appropriate Kubernetes command\n"
                "3. The results will be automatically polished for the user\n\n"
                
                "**CRITICAL INSTRUCTIONS:**\n"
                "- ALWAYS use intelligent defaults (nginx for web pods)\n"
                "- NEVER ask for clarification - use smart defaults\n"
                "- Execute tools immediately when you understand the intent\n"
                "- Keep your explanations brief - the results will be polished\n\n"
                
                "**POD OPERATIONS:**\n"
                "- 'show me all pods' → Use pods_list\n"
                "- 'list pods in default namespace' → Use pods_list with namespace=default\n"
                "- 'get pod named X' → Use pods_get with name=X\n"
                "- 'create pod named X' → Use pods_run with name=X, image=nginx\n"
                "- 'delete pod X' → Use pods_delete with name=X\n"
                "- 'logs from pod X' → Use pods_log with name=X\n"
                "- 'execute command Y in pod X' → Use pods_exec\n\n"
                
                f"Available tools: {json.dumps([t['name'] for t in available_tools], indent=2)}"
            )
            
            messages = [{"role": "user", "content": query}]
            
            # Call AI with tools (Groq or Anthropic)
            if self.ai_provider == "groq":
                # Groq doesn't support function calling, use natural language approach
                return self._process_with_groq_nl(query, system_prompt)
            else:
                # Anthropic with function calling
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=800,
                    system=system_prompt,
                    messages=messages,
                    tools=available_tools
                )
            
            # Process AI response and handle tool calls
            final_text = []
            tool_results = []
            
            print(f"🔧 AI Response Content: {[c.type for c in response.content]}")
            
            for content in response.content:
                if content.type == 'text':
                    print(f"🔧 AI Text: {content.text}")
                    final_text.append(content.text)
                elif content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input
                    print(f"🔧 AI Tool Call: {tool_name} with args: {tool_args}")
                    
                    # Execute the tool call
                    result = self._call_mcp_tool(tool_name, tool_args)
                    tool_results.append({
                        'tool_name': tool_name,
                        'tool_args': tool_args,
                        'result': result
                    })
            
            # Post-process results with AI for polished responses
            if tool_results:
                # Get the AI's brief explanation
                ai_explanation = "\n\n".join([t.strip() for t in final_text if t and t.strip()])
                
                # Post-process each tool result
                polished_results = []
                for tool_result in tool_results:
                    if tool_result['result']['success']:
                        polished_result = self._post_process_with_ai(
                            query,
                            tool_result['tool_name'],
                            tool_result['result']['result']
                        )
                        polished_results.append(polished_result)
                    else:
                        polished_results.append(f"❌ **Error executing {tool_result['tool_name']}:** {tool_result['result']['error']}")
                
                # Combine AI explanation with polished results
                if ai_explanation:
                    response_text = f"{ai_explanation}\n\n{chr(10).join(polished_results)}"
                else:
                    response_text = "\n\n".join(polished_results)
                
                return {
                    'command': f'AI: {", ".join([tr["tool_name"] for tr in tool_results])}',
                    'explanation': response_text,
                    'ai_processed': True,
                    'tool_results': tool_results,
                    'mcp_result': tool_results[0]['result'] if tool_results else None
                }
            else:
                # No tools called, just return AI text
                response_text = "\n\n".join([t.strip() for t in final_text if t and t.strip()])
                return {
                    'command': 'AI: text_response',
                    'explanation': response_text,
                    'ai_processed': True,
                    'tool_results': [],
                    'mcp_result': None
                }
                
        except Exception as e:
            return {
                'command': None,
                'explanation': f"AI processing failed: {str(e)}",
                'ai_processed': True,
                'tool_results': [],
                'mcp_result': None
            }
    
    def process_query(self, query: str) -> dict:
        """Process natural language query using enhanced AI with post-processing"""
        
        # Try AI processing first if available
        if self.use_ai and self.anthropic:
            try:
                ai_result = self._process_with_ai(query)
                print(f"🔧 Enhanced AI Result: {ai_result}")
                if ai_result.get('ai_processed') and ai_result.get('command'):
                    print(f"✅ Using enhanced AI result: {ai_result.get('command')}")
                    return ai_result
                else:
                    print(f"⚠️  AI didn't process properly")
            except Exception as e:
                print(f"⚠️  AI processing failed: {e}")
        
        # Fallback to basic response
        return {
            'command': 'fallback',
            'explanation': 'AI processing is temporarily unavailable due to rate limits. Please wait a moment and try again.',
            'ai_processed': False,
            'tool_results': [],
            'mcp_result': None
        }
