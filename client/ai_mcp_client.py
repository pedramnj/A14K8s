import asyncio
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.stdio = None
        self.write = None

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python3" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        system_prompt = (
            "You are an AI assistant operating an MCP client connected to a Kubernetes MCP server. "
            "When you need cluster data or to take action, choose the correct tool. "
            "Be concise and return clear, well-formatted answers."
        )

        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Call Claude and handle iterative tool use until there are no more tool calls
        final_text = []
        while True:
            assistant_message_content = []
            resp = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1200,
                system=system_prompt,
                messages=messages,
                tools=available_tools
            )

            tool_results_to_append = []
            has_tool_use = False

            for content in resp.content:
                if content.type == 'text':
                    final_text.append(content.text)
                    assistant_message_content.append(content)
                elif content.type == 'tool_use':
                    has_tool_use = True
                    tool_name = content.name
                    tool_args = content.input

                    # Execute tool call via MCP session
                    result = await self.session.call_tool(tool_name, tool_args)
                    assistant_message_content.append(content)

                    # Send tool result back to Claude in the next turn
                    tool_results_to_append.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content,
                            }
                        ],
                    })

            # Append assistant turn (including tool_use directives) to history
            if assistant_message_content:
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content,
                })

            # If there were tool calls, append their results and continue the loop
            if has_tool_use:
                messages.extend(tool_results_to_append)
                continue

            # No tool calls -> finalize with the latest assistant text already captured
            break

        # Join and lightly format
        return "\n\n".join([t.strip() for t in final_text if t and t.strip()])

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\n🤖 Kubernetes MCP Client Started!")
        print("Type your queries about your Kubernetes cluster or 'quit' to exit.")
        print("\nExample queries:")
        print("- 'What pods are running in my cluster?'")
        print("- 'Show me the status of my nginx deployment'")
        print("- 'Get information about my cluster nodes'")
        print("- 'What services are available?'")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                print("\n🤔 Processing your query...")
                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        print("Example: python client.py ../mcp_server.py")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
