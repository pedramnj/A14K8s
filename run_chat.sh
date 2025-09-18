#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER="$SCRIPT_DIR/mcp_server.py"
CLIENT="$SCRIPT_DIR/client/client.py"
CLIENT_DIR="$SCRIPT_DIR/client"

if [ ! -f "$SERVER" ]; then
  echo "Missing mcp_server.py at $SERVER" >&2
  exit 1
fi
if [ ! -f "$CLIENT" ]; then
  echo "Missing client.py at $CLIENT" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  RUN_SERVER=(uv -C "$CLIENT_DIR" run)
  RUN_CLIENT=(uv -C "$CLIENT_DIR" run)
else
  RUN_SERVER=(python)
  RUN_CLIENT=(python)
fi

# Ensure Anthropic key is present
if [ -f "$CLIENT_DIR/.env" ]; then
  export $(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$CLIENT_DIR/.env" | xargs -I{} echo {})
fi

# Start server and client
echo "Starting MCP server..."
"${RUN_SERVER[@]}" "$SERVER" | sed 's/^/[server] /' &
SERVER_PID=$!

cleanup() {
  kill $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 1

echo "Starting MCP client..."
"${RUN_CLIENT[@]}" "$CLIENT" "$SERVER"
