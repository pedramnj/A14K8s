#!/bin/bash

# NetPress Benchmark Runner for AI4K8s
# This script runs comprehensive benchmarks on the AI4K8s MCP agent

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/netpress-results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}=== AI4K8s NetPress Benchmark Runner ===${NC}"
echo "Timestamp: $TIMESTAMP"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Function to check if service is running
check_service() {
    local url=$1
    local name=$2
    
    echo -e "${YELLOW}Checking $name at $url...${NC}"
    
    if curl -s --connect-timeout 5 "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $name is running${NC}"
        return 0
    else
        echo -e "${RED}✗ $name is not accessible at $url${NC}"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}Waiting for $name to be ready...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 5 "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $name is ready${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}Attempt $attempt/$max_attempts - waiting for $name...${NC}"
        sleep 2
        ((attempt++))
    done
    
    echo -e "${RED}✗ $name failed to start within timeout${NC}"
    return 1
}

# Check prerequisites
echo -e "${BLUE}=== Checking Prerequisites ===${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 is available${NC}"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl is available${NC}"

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}✗ Kubernetes cluster is not accessible${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Kubernetes cluster is accessible${NC}"

echo ""

# Check services
echo -e "${BLUE}=== Checking Services ===${NC}"

# Check MCP Bridge
if ! check_service "http://localhost:5001/api/mcp-status" "MCP Bridge"; then
    echo -e "${YELLOW}MCP Bridge not running. Please start it with:${NC}"
    echo "kubectl apply -f $PROJECT_ROOT/mcp-bridge-deployment.yaml"
    echo "kubectl -n web port-forward service/mcp-bridge 5001:5001"
    exit 1
fi

# Check Web App
if ! check_service "http://localhost:8080" "Web App"; then
    echo -e "${YELLOW}Web App not running. Please start it with:${NC}"
    echo "kubectl apply -f $PROJECT_ROOT/web-app-iframe-solution.yaml"
    echo "kubectl -n web port-forward service/nginx-proxy 8080:80"
    exit 1
fi

echo ""

# Install Python dependencies
echo -e "${BLUE}=== Installing Dependencies ===${NC}"
cd "$SCRIPT_DIR"

if [ ! -f "requirements.txt" ]; then
    echo "Creating requirements.txt..."
    cat > requirements.txt << EOF
requests>=2.31.0
python-dotenv>=1.0.0
EOF
fi

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo ""

# Run benchmarks
echo -e "${BLUE}=== Running NetPress Benchmarks ===${NC}"

# Basic benchmark
echo -e "${YELLOW}Running basic benchmark...${NC}"
python3 benchmark_runner.py \
    --config config.json \
    --output "$OUTPUT_DIR/basic_benchmark_$TIMESTAMP.json"

# Advanced benchmark
echo -e "${YELLOW}Running advanced benchmark...${NC}"
python3 benchmark_runner.py \
    --config config.json \
    --advanced \
    --output "$OUTPUT_DIR/advanced_benchmark_$TIMESTAMP.json"

echo ""

# Generate summary report
echo -e "${BLUE}=== Generating Summary Report ===${NC}"

cat > "$OUTPUT_DIR/summary_$TIMESTAMP.md" << EOF
# AI4K8s NetPress Benchmark Results

**Timestamp:** $TIMESTAMP  
**Project:** AI4K8s - AI-Powered Kubernetes Management System  
**Benchmark Framework:** NetPress  

## Test Environment

- **MCP Bridge URL:** http://localhost:5001
- **Web App URL:** http://localhost:8080
- **Kubernetes Cluster:** $(kubectl config current-context)
- **Cluster Version:** $(kubectl version --short --client 2>/dev/null | head -1)

## Benchmark Results

### Basic Benchmark
- **Results File:** basic_benchmark_$TIMESTAMP.json
- **Query Types:** Basic Kubernetes operations (list, status, logs, etc.)

### Advanced Benchmark  
- **Results File:** advanced_benchmark_$TIMESTAMP.json
- **Query Types:** Complex scenarios, troubleshooting, security recommendations

## Files Generated

- \`basic_benchmark_$TIMESTAMP.json\` - Basic benchmark results
- \`advanced_benchmark_$TIMESTAMP.json\` - Advanced benchmark results
- \`summary_$TIMESTAMP.md\` - This summary report

## Next Steps

1. Review the JSON results files for detailed metrics
2. Compare results across different runs
3. Use results to improve the MCP agent
4. Share results with the NetPress community

## NetPress Integration

This benchmark demonstrates how AI4K8s MCP agent can be evaluated using the NetPress framework, providing:

- **Correctness:** How accurately the agent responds to Kubernetes queries
- **Safety:** How safely the agent handles potentially dangerous operations  
- **Latency:** How quickly the agent responds to queries

EOF

echo -e "${GREEN}✓ Summary report generated: $OUTPUT_DIR/summary_$TIMESTAMP.md${NC}"

echo ""
echo -e "${GREEN}=== Benchmark Complete ===${NC}"
echo -e "Results saved to: ${BLUE}$OUTPUT_DIR${NC}"
echo -e "Summary report: ${BLUE}$OUTPUT_DIR/summary_$TIMESTAMP.md${NC}"
echo ""
echo -e "${YELLOW}To view results:${NC}"
echo "cat $OUTPUT_DIR/basic_benchmark_$TIMESTAMP.json | jq ."
echo "cat $OUTPUT_DIR/advanced_benchmark_$TIMESTAMP.json | jq ."
echo ""
echo -e "${YELLOW}To run again:${NC}"
echo "bash $SCRIPT_DIR/run_benchmark.sh"
