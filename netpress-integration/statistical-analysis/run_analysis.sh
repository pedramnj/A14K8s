#!/bin/bash

# NetPress Statistical Analysis Runner
# Generates comprehensive statistical analysis with confidence intervals

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_DIR="$PROJECT_ROOT/netpress-results"
OUTPUT_DIR="$SCRIPT_DIR/analysis_output"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${BLUE}=== NetPress Statistical Analysis ===${NC}"
echo "Timestamp: $TIMESTAMP"
echo "Results Directory: $RESULTS_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to find latest results file
find_latest_results() {
    local pattern=$1
    find "$RESULTS_DIR" -name "$pattern" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
}

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}✗ Results directory not found: $RESULTS_DIR${NC}"
    echo "Please run the NetPress benchmark first:"
    echo "cd $PROJECT_ROOT/netpress-integration && ./run_benchmark.sh"
    exit 1
fi

# Find latest results files
echo -e "${YELLOW}Looking for benchmark results...${NC}"

BASIC_RESULTS=$(find_latest_results "basic_benchmark_*.json")
ADVANCED_RESULTS=$(find_latest_results "advanced_benchmark_*.json")

if [ -z "$BASIC_RESULTS" ] && [ -z "$ADVANCED_RESULTS" ]; then
    echo -e "${RED}✗ No benchmark results found in $RESULTS_DIR${NC}"
    echo "Please run the NetPress benchmark first:"
    echo "cd $PROJECT_ROOT/netpress-integration && ./run_benchmark.sh"
    exit 1
fi

echo -e "${GREEN}✓ Found benchmark results${NC}"
if [ -n "$BASIC_RESULTS" ]; then
    echo "  Basic: $(basename "$BASIC_RESULTS")"
fi
if [ -n "$ADVANCED_RESULTS" ]; then
    echo "  Advanced: $(basename "$ADVANCED_RESULTS")"
fi

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip3 install -r "$SCRIPT_DIR/requirements.txt"

# Run statistical analysis for each results file
if [ -n "$BASIC_RESULTS" ]; then
    echo -e "${YELLOW}Running statistical analysis for basic benchmark...${NC}"
    python3 "$SCRIPT_DIR/statistical_analyzer.py" \
        --results "$BASIC_RESULTS" \
        --output-dir "$OUTPUT_DIR/basic_analysis_$TIMESTAMP" \
        --report "$OUTPUT_DIR/basic_report_$TIMESTAMP.md" \
        --confidence 0.95
    
    echo -e "${GREEN}✓ Basic benchmark analysis complete${NC}"
fi

if [ -n "$ADVANCED_RESULTS" ]; then
    echo -e "${YELLOW}Running statistical analysis for advanced benchmark...${NC}"
    python3 "$SCRIPT_DIR/statistical_analyzer.py" \
        --results "$ADVANCED_RESULTS" \
        --output-dir "$OUTPUT_DIR/advanced_analysis_$TIMESTAMP" \
        --report "$OUTPUT_DIR/advanced_report_$TIMESTAMP.md" \
        --confidence 0.95
    
    echo -e "${GREEN}✓ Advanced benchmark analysis complete${NC}"
fi

# Create comparison analysis if both results exist
if [ -n "$BASIC_RESULTS" ] && [ -n "$ADVANCED_RESULTS" ]; then
    echo -e "${YELLOW}Creating comparative analysis...${NC}"
    
    # Create comparison script
    cat > "$OUTPUT_DIR/compare_analysis.py" << 'EOF'
#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_metrics(basic_file, advanced_file, output_dir):
    basic_data = load_results(basic_file)
    advanced_data = load_results(advanced_file)
    
    # Extract metrics
    basic_results = basic_data.get('detailed_results', [])
    advanced_results = advanced_data.get('detailed_results', [])
    
    metrics = ['correctness', 'safety', 'latency']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        basic_values = [r[metric] for r in basic_results if metric in r]
        advanced_values = [r[metric] for r in advanced_results if metric in r]
        
        # Box plot comparison
        data_to_plot = [basic_values, advanced_values]
        labels = ['Basic', 'Advanced']
        
        axes[i].boxplot(data_to_plot, labels=labels, patch_artist=True)
        axes[i].set_title(f'{metric.title()} Comparison')
        axes[i].set_ylabel(metric.title())
        axes[i].grid(True, alpha=0.3)
        
        # Perform t-test
        if len(basic_values) > 1 and len(advanced_values) > 1:
            t_stat, p_value = stats.ttest_ind(basic_values, advanced_values)
            axes[i].text(0.5, 0.95, f't-test p-value: {p_value:.4f}', 
                        transform=axes[i].transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Basic vs Advanced Benchmark Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate comparison report
    report = "# Basic vs Advanced Benchmark Comparison\n\n"
    
    for metric in metrics:
        basic_values = [r[metric] for r in basic_results if metric in r]
        advanced_values = [r[metric] for r in advanced_results if metric in r]
        
        if len(basic_values) > 1 and len(advanced_values) > 1:
            basic_mean = np.mean(basic_values)
            advanced_mean = np.mean(advanced_values)
            t_stat, p_value = stats.ttest_ind(basic_values, advanced_values)
            
            report += f"## {metric.title()}\n\n"
            report += f"- **Basic Benchmark**: Mean = {basic_mean:.3f}\n"
            report += f"- **Advanced Benchmark**: Mean = {advanced_mean:.3f}\n"
            report += f"- **Difference**: {advanced_mean - basic_mean:.3f}\n"
            report += f"- **t-test p-value**: {p_value:.4f}\n"
            report += f"- **Significant**: {'Yes' if p_value < 0.05 else 'No'}\n\n"
    
    with open(f'{output_dir}/comparison_report.md', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    basic_file = sys.argv[1]
    advanced_file = sys.argv[2]
    output_dir = sys.argv[3]
    compare_metrics(basic_file, advanced_file, output_dir)
EOF
    
    python3 "$OUTPUT_DIR/compare_analysis.py" \
        "$BASIC_RESULTS" \
        "$ADVANCED_RESULTS" \
        "$OUTPUT_DIR"
    
    echo -e "${GREEN}✓ Comparative analysis complete${NC}"
fi

# Generate summary report
echo -e "${YELLOW}Generating summary report...${NC}"

cat > "$OUTPUT_DIR/summary_$TIMESTAMP.md" << EOF
# NetPress Statistical Analysis Summary

**Timestamp:** $TIMESTAMP  
**Analysis Date:** $(date)  
**Project:** AI4K8s - AI-Powered Kubernetes Management System  

## Analysis Overview

This statistical analysis provides comprehensive evaluation of the NetPress benchmark results for the AI4K8s MCP Agent, including:

- **Descriptive Statistics**: Mean, median, standard deviation, quartiles
- **Confidence Intervals**: 95% confidence intervals for all metrics
- **Normality Tests**: Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling
- **Visualizations**: Distribution plots, box plots, correlation matrices
- **Comparative Analysis**: Basic vs Advanced benchmark comparison

## Generated Files

EOF

if [ -n "$BASIC_RESULTS" ]; then
    echo "- \`basic_analysis_$TIMESTAMP/\` - Basic benchmark statistical analysis" >> "$OUTPUT_DIR/summary_$TIMESTAMP.md"
    echo "- \`basic_report_$TIMESTAMP.md\` - Basic benchmark statistical report" >> "$OUTPUT_DIR/summary_$TIMESTAMP.md"
fi

if [ -n "$ADVANCED_RESULTS" ]; then
    echo "- \`advanced_analysis_$TIMESTAMP/\` - Advanced benchmark statistical analysis" >> "$OUTPUT_DIR/summary_$TIMESTAMP.md"
    echo "- \`advanced_report_$TIMESTAMP.md\` - Advanced benchmark statistical report" >> "$OUTPUT_DIR/summary_$TIMESTAMP.md"
fi

if [ -n "$BASIC_RESULTS" ] && [ -n "$ADVANCED_RESULTS" ]; then
    echo "- \`comparison_analysis.png\` - Basic vs Advanced comparison visualization" >> "$OUTPUT_DIR/summary_$TIMESTAMP.md"
    echo "- \`comparison_report.md\` - Comparative analysis report" >> "$OUTPUT_DIR/summary_$TIMESTAMP.md"
fi

cat >> "$OUTPUT_DIR/summary_$TIMESTAMP.md" << EOF

## Key Metrics Analyzed

1. **Correctness**: How accurately the agent responds to queries
2. **Safety**: How safely the agent handles operations
3. **Latency**: Response time in seconds

## Statistical Methods

- **Confidence Intervals**: 95% confidence level using t-distribution
- **Normality Tests**: Multiple tests to assess data distribution
- **Comparative Analysis**: Independent t-tests for significance testing
- **Visualization**: Comprehensive plots for data exploration

## Usage

To view the analysis results:

\`\`\`bash
# View plots
open $OUTPUT_DIR/*/distributions.png
open $OUTPUT_DIR/*/confidence_intervals.png
open $OUTPUT_DIR/*/summary_dashboard.png

# View reports
cat $OUTPUT_DIR/*_report_$TIMESTAMP.md
\`\`\`

## Next Steps

1. Review the statistical reports for insights
2. Use confidence intervals for performance evaluation
3. Compare results across different benchmark runs
4. Incorporate findings into thesis documentation

---
*Analysis generated by NetPress Statistical Analyzer*
EOF

echo -e "${GREEN}✓ Summary report generated: $OUTPUT_DIR/summary_$TIMESTAMP.md${NC}"

echo ""
echo -e "${GREEN}=== Statistical Analysis Complete ===${NC}"
echo -e "Results saved to: ${BLUE}$OUTPUT_DIR${NC}"
echo -e "Summary report: ${BLUE}$OUTPUT_DIR/summary_$TIMESTAMP.md${NC}"
echo ""
echo -e "${YELLOW}To view results:${NC}"
echo "open $OUTPUT_DIR/*/summary_dashboard.png"
echo "cat $OUTPUT_DIR/*_report_$TIMESTAMP.md"
echo ""
echo -e "${YELLOW}To run analysis again:${NC}"
echo "bash $SCRIPT_DIR/run_analysis.sh"
