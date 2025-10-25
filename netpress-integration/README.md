# NetPress Integration for AI4K8s

This directory contains the NetPress benchmarking integration for the AI4K8s MCP agent. NetPress is a dynamic benchmark generation framework for evaluating LLM agents in real-world network applications.

## Overview

The NetPress integration allows you to benchmark your AI4K8s MCP agent against standardized Kubernetes scenarios, measuring:

- **Correctness**: How accurately the agent responds to Kubernetes queries
- **Safety**: How safely the agent handles potentially dangerous operations
- **Latency**: How quickly the agent responds to queries

## Files

- `mcp_agent.py` - AI4K8s MCP Agent wrapper for NetPress
- `benchmark_runner.py` - Main benchmark runner with comprehensive testing
- `config.json` - Configuration file for benchmark settings
- `run_benchmark.sh` - Shell script to run complete benchmark suite
- `README.md` - This documentation

## Prerequisites

1. **AI4K8s System Running**:
   - MCP Bridge service running on port 5001
   - Web App running on port 8080
   - Kubernetes cluster accessible

2. **Python Dependencies**:
   ```bash
   pip install requests python-dotenv
   ```

3. **System Tools**:
   - `kubectl` - Kubernetes command-line tool
   - `curl` - For service health checks
   - `jq` - For JSON processing (optional)

## Quick Start

1. **Ensure AI4K8s is running**:
   ```bash
   # Start MCP Bridge
   kubectl apply -f ../mcp-bridge-deployment.yaml
   kubectl -n web port-forward service/mcp-bridge 5001:5001 &
   
   # Start Web App
   kubectl apply -f ../web-app-iframe-solution.yaml
   kubectl -n web port-forward service/nginx-proxy 8080:80 &
   ```

2. **Run the benchmark**:
   ```bash
   ./run_benchmark.sh
   ```

3. **View results**:
   ```bash
   # View basic benchmark results
   cat ../netpress-results/basic_benchmark_*.json | jq .
   
   # View advanced benchmark results
   cat ../netpress-results/advanced_benchmark_*.json | jq .
   ```

## Manual Usage

### Basic Benchmark

```bash
python3 benchmark_runner.py --config config.json --output results.json
```

### Advanced Benchmark

```bash
python3 benchmark_runner.py --config config.json --advanced --output results.json
```

### Custom Configuration

```bash
python3 benchmark_runner.py \
    --mcp-bridge http://localhost:5001 \
    --web-app http://localhost:8080 \
    --output custom_results.json
```

## Benchmark Queries

### Basic Queries
- List pods, services, deployments
- Get cluster status and version
- Show pod logs and resource usage
- Basic scaling operations

### Advanced Queries
- Complex troubleshooting scenarios
- Security best practices
- Monitoring setup
- Multi-step operations

## Results Format

The benchmark generates JSON results with the following structure:

```json
{
  "summary": {
    "total_queries": 10,
    "successful_queries": 9,
    "success_rate": 0.9,
    "average_correctness": 0.85,
    "average_safety": 0.92,
    "average_latency": 2.3
  },
  "detailed_results": [
    {
      "query": "List all pods in the cluster",
      "response": "Found 15 pods...",
      "correctness": 0.9,
      "safety": 1.0,
      "latency": 1.2,
      "success": true
    }
  ],
  "benchmark_info": {
    "total_time": 25.4,
    "queries_per_second": 0.39,
    "config": {...}
  }
}
```

## Customization

### Adding Custom Queries

Create a JSON file with your custom queries:

```json
[
  {
    "query": "Your custom query here",
    "type": "custom_type",
    "expected_result": "Expected response",
    "category": "custom_category"
  }
]
```

Run with custom queries:

```bash
python3 benchmark_runner.py --queries custom_queries.json --output results.json
```

### Modifying Evaluation Criteria

Edit `config.json` to adjust evaluation weights:

```json
{
  "evaluation_criteria": {
    "correctness_weights": {
      "keyword_match": 0.3,
      "command_execution": 0.4,
      "result_accuracy": 0.3
    },
    "safety_weights": {
      "dangerous_operations": -0.5,
      "safe_operations": 0.3,
      "error_handling": 0.2
    }
  }
}
```

## Integration with NetPress Framework

This integration follows NetPress principles:

1. **Dynamic Benchmark Generation**: Queries are generated based on real Kubernetes scenarios
2. **Real Environment Feedback**: Tests run against actual Kubernetes clusters
3. **Comprehensive Evaluation**: Measures correctness, safety, and latency
4. **Standardized Metrics**: Results can be compared with other NetPress benchmarks

## Contributing to NetPress

Your AI4K8s MCP agent results can contribute to the NetPress research:

1. **Share Results**: Submit benchmark results to the NetPress community
2. **Add Scenarios**: Contribute new Kubernetes benchmark scenarios
3. **Improve Evaluation**: Enhance evaluation criteria for Kubernetes operations
4. **Documentation**: Share integration patterns and best practices

## Troubleshooting

### Services Not Available

If services are not accessible:

```bash
# Check if pods are running
kubectl -n web get pods

# Check port forwards
kubectl -n web port-forward service/mcp-bridge 5001:5001
kubectl -n web port-forward service/nginx-proxy 8080:80
```

### Permission Issues

Ensure kubectl has proper permissions:

```bash
kubectl auth can-i get pods
kubectl auth can-i list services
```

### Network Issues

Check network connectivity:

```bash
curl -I http://localhost:5001/api/mcp-status
curl -I http://localhost:8080
```

## References

- [NetPress GitHub Repository](https://github.com/Froot-NetSys/NetPress)
- [NetPress Paper](https://arxiv.org/abs/2506.03231)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [AI4K8s Project](../README.md)

## License

This integration follows the same license as the main AI4K8s project.
