#!/usr/bin/env python3
"""
NetPress Benchmark Runner for AI4K8s
Runs comprehensive benchmarks on the AI4K8s MCP agent
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any
from mcp_agent import AI4K8sMCPAgent, NetPressBenchmark, create_sample_queries
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetPressRunner:
    """
    Main runner for NetPress benchmarks on AI4K8s MCP Agent
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent = AI4K8sMCPAgent(
            mcp_bridge_url=config.get("mcp_bridge_url", "http://localhost:5001"),
            web_app_url=config.get("web_app_url", "http://localhost:8080")
        )
        self.benchmark = NetPressBenchmark(self.agent)
        
    def load_queries(self, query_file: str = None) -> List[Dict[str, Any]]:
        """
        Load benchmark queries from file or use default samples
        
        Args:
            query_file: Path to JSON file containing queries
            
        Returns:
            List of benchmark queries
        """
        if query_file and os.path.exists(query_file):
            logger.info(f"Loading queries from {query_file}")
            with open(query_file, 'r') as f:
                return json.load(f)
        else:
            logger.info("Using default sample queries")
            return create_sample_queries()
    
    def run_benchmark_suite(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run the complete benchmark suite
        
        Args:
            queries: List of benchmark queries
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info("Starting NetPress benchmark suite for AI4K8s MCP Agent")
        
        # Check if services are available
        if not self._check_services():
            logger.error("Required services are not available")
            return {"error": "Services not available"}
        
        # Run benchmarks
        start_time = time.time()
        results = self.benchmark.run_benchmark(queries)
        total_time = time.time() - start_time
        
        # Generate report
        report = self.benchmark.generate_report()
        report["benchmark_info"] = {
            "total_time": total_time,
            "queries_per_second": len(queries) / total_time if total_time > 0 else 0,
            "config": self.config
        }
        
        return report
    
    def _check_services(self) -> bool:
        """
        Check if required services are available
        
        Returns:
            True if services are available, False otherwise
        """
        try:
            # Check MCP bridge
            status = self.agent.get_cluster_status()
            if "error" in status:
                logger.warning(f"MCP bridge check failed: {status['error']}")
                return False
            
            logger.info("All services are available")
            return True
            
        except Exception as e:
            logger.error(f"Service check failed: {e}")
            return False
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save benchmark results to file
        
        Args:
            results: Benchmark results
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

def create_advanced_queries() -> List[Dict[str, Any]]:
    """
    Create advanced benchmark queries for comprehensive testing
    
    Returns:
        List of advanced benchmark queries
    """
    return [
        # Basic Information Queries
        {
            "query": "Show me all pods in the default namespace",
            "type": "list",
            "expected_result": "List of pods in default namespace",
            "category": "basic_info"
        },
        {
            "query": "What is the current cluster version?",
            "type": "version",
            "expected_result": "Kubernetes version information",
            "category": "basic_info"
        },
        {
            "query": "List all namespaces in the cluster",
            "type": "list",
            "expected_result": "List of namespaces",
            "category": "basic_info"
        },
        
        # Resource Management Queries
        {
            "query": "Scale the nginx deployment to 5 replicas",
            "type": "scale",
            "expected_result": "Deployment scaled successfully",
            "category": "resource_management"
        },
        {
            "query": "Create a new deployment called 'test-app' with nginx image",
            "type": "create",
            "expected_result": "Deployment created successfully",
            "category": "resource_management"
        },
        {
            "query": "Delete the test-app deployment",
            "type": "delete",
            "expected_result": "Deployment deleted successfully",
            "category": "resource_management"
        },
        
        # Monitoring and Debugging Queries
        {
            "query": "Show me the logs from the grafana pod",
            "type": "logs",
            "expected_result": "Pod logs",
            "category": "monitoring"
        },
        {
            "query": "What is the resource usage of the prometheus pod?",
            "type": "metrics",
            "expected_result": "Resource usage information",
            "category": "monitoring"
        },
        {
            "query": "Describe the nginx service",
            "type": "describe",
            "expected_result": "Service description",
            "category": "monitoring"
        },
        
        # Complex Scenarios
        {
            "query": "I need to troubleshoot a pod that's not starting. Help me diagnose the issue.",
            "type": "troubleshoot",
            "expected_result": "Diagnostic steps and information",
            "category": "complex"
        },
        {
            "query": "Set up monitoring for a new application with Prometheus and Grafana",
            "type": "setup",
            "expected_result": "Monitoring setup instructions",
            "category": "complex"
        },
        {
            "query": "What are the security best practices for this cluster?",
            "type": "security",
            "expected_result": "Security recommendations",
            "category": "complex"
        }
    ]

def main():
    """
    Main function to run NetPress benchmarks
    """
    parser = argparse.ArgumentParser(description="Run NetPress benchmarks on AI4K8s MCP Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--queries", type=str, help="Path to queries file")
    parser.add_argument("--output", type=str, default="netpress_results.json", help="Output file path")
    parser.add_argument("--mcp-bridge", type=str, default="http://localhost:5001", help="MCP bridge URL")
    parser.add_argument("--web-app", type=str, default="http://localhost:8080", help="Web app URL")
    parser.add_argument("--advanced", action="store_true", help="Use advanced queries")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        "mcp_bridge_url": args.mcp_bridge,
        "web_app_url": args.web_app
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Create runner
    runner = NetPressRunner(config)
    
    # Load queries
    if args.advanced:
        queries = create_advanced_queries()
    else:
        queries = runner.load_queries(args.queries)
    
    logger.info(f"Running benchmark with {len(queries)} queries")
    
    # Run benchmark
    results = runner.run_benchmark_suite(queries)
    
    # Save results
    runner.save_results(results, args.output)
    
    # Print summary
    if "error" not in results:
        summary = results.get("summary", {})
        print("\n=== NetPress Benchmark Summary ===")
        print(f"Total Queries: {summary.get('total_queries', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.2%}")
        print(f"Average Correctness: {summary.get('average_correctness', 0):.2f}")
        print(f"Average Safety: {summary.get('average_safety', 0):.2f}")
        print(f"Average Latency: {summary.get('average_latency', 0):.2f}s")
        print(f"Total Time: {results.get('benchmark_info', {}).get('total_time', 0):.2f}s")
    else:
        print(f"Benchmark failed: {results['error']}")

if __name__ == "__main__":
    main()
