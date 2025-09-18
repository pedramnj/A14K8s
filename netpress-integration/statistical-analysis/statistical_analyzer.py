#!/usr/bin/env python3
"""
Statistical Analysis for NetPress Benchmark Results
Provides comprehensive statistical analysis including confidence intervals
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t, norm
import argparse
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NetPressStatisticalAnalyzer:
    """
    Comprehensive statistical analysis for NetPress benchmark results
    """
    
    def __init__(self, results_file: str):
        """
        Initialize analyzer with benchmark results
        
        Args:
            results_file: Path to JSON file containing benchmark results
        """
        self.results_file = results_file
        self.data = self._load_results()
        self.df = self._create_dataframe()
        
    def _load_results(self) -> Dict[str, Any]:
        """Load benchmark results from JSON file"""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from benchmark results"""
        results = self.data.get('detailed_results', [])
        df = pd.DataFrame(results)
        
        # Convert numeric columns
        numeric_cols = ['correctness', 'safety', 'latency']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def calculate_confidence_intervals(self, confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence intervals for key metrics
        
        Args:
            confidence_level: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Dictionary containing confidence intervals for each metric
        """
        metrics = ['correctness', 'safety', 'latency']
        ci_results = {}
        
        for metric in metrics:
            if metric in self.df.columns:
                values = self.df[metric].dropna()
                n = len(values)
                
                if n > 1:
                    # Calculate mean and standard error
                    mean = values.mean()
                    std_err = values.std() / np.sqrt(n)
                    
                    # Calculate degrees of freedom
                    df = n - 1
                    
                    # Calculate t-value for confidence interval
                    alpha = 1 - confidence_level
                    t_value = t.ppf(1 - alpha/2, df)
                    
                    # Calculate confidence interval
                    margin_error = t_value * std_err
                    ci_lower = mean - margin_error
                    ci_upper = mean + margin_error
                    
                    ci_results[metric] = {
                        'mean': mean,
                        'std': values.std(),
                        'std_error': std_err,
                        'n': n,
                        'confidence_level': confidence_level,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'margin_error': margin_error
                    }
        
        return ci_results
    
    def calculate_descriptive_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive descriptive statistics
        
        Returns:
            Dictionary containing descriptive statistics for each metric
        """
        metrics = ['correctness', 'safety', 'latency']
        stats_results = {}
        
        for metric in metrics:
            if metric in self.df.columns:
                values = self.df[metric].dropna()
                
                stats_results[metric] = {
                    'count': len(values),
                    'mean': values.mean(),
                    'median': values.median(),
                    'std': values.std(),
                    'var': values.var(),
                    'min': values.min(),
                    'max': values.max(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75),
                    'iqr': values.quantile(0.75) - values.quantile(0.25),
                    'skewness': values.skew(),
                    'kurtosis': values.kurtosis()
                }
        
        return stats_results
    
    def perform_normality_tests(self) -> Dict[str, Dict[str, float]]:
        """
        Perform normality tests on the data
        
        Returns:
            Dictionary containing normality test results
        """
        metrics = ['correctness', 'safety', 'latency']
        normality_results = {}
        
        for metric in metrics:
            if metric in self.df.columns:
                values = self.df[metric].dropna()
                
                # Shapiro-Wilk test (for small samples)
                if len(values) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(values)
                else:
                    shapiro_stat, shapiro_p = np.nan, np.nan
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(values, 'norm', args=(values.mean(), values.std()))
                
                # Anderson-Darling test
                ad_stat, ad_critical, ad_significance = stats.anderson(values, dist='norm')
                
                normality_results[metric] = {
                    'shapiro_stat': shapiro_stat,
                    'shapiro_p': shapiro_p,
                    'ks_stat': ks_stat,
                    'ks_p': ks_p,
                    'ad_stat': ad_stat,
                    'ad_critical': ad_critical[2] if len(ad_critical) > 2 else ad_critical[0],  # Use 5% significance level
                    'ad_significance': ad_significance
                }
        
        return normality_results
    
    def create_visualizations(self, output_dir: str = "plots"):
        """
        Create comprehensive visualizations
        
        Args:
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Distribution plots
        self._create_distribution_plots(output_dir)
        
        # 2. Confidence interval plots
        self._create_confidence_interval_plots(output_dir)
        
        # 3. Box plots
        self._create_box_plots(output_dir)
        
        # 4. Correlation matrix
        self._create_correlation_plot(output_dir)
        
        # 5. Time series analysis
        self._create_time_series_plots(output_dir)
        
        # 6. Summary dashboard
        self._create_summary_dashboard(output_dir)
    
    def _create_distribution_plots(self, output_dir: str):
        """Create distribution plots for each metric"""
        metrics = ['correctness', 'safety', 'latency']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if metric in self.df.columns:
                values = self.df[metric].dropna()
                
                # Histogram with KDE
                axes[i].hist(values, bins=20, alpha=0.7, density=True, color='skyblue', edgecolor='black')
                
                # Add KDE curve
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 100)
                axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                
                # Add mean line
                mean_val = values.mean()
                axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                
                axes[i].set_title(f'{metric.title()} Distribution')
                axes[i].set_xlabel(metric.title())
                axes[i].set_ylabel('Density')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(metrics) < 4:
            axes[3].remove()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_confidence_interval_plots(self, output_dir: str):
        """Create confidence interval plots"""
        ci_results = self.calculate_confidence_intervals()
        
        metrics = list(ci_results.keys())
        means = [ci_results[metric]['mean'] for metric in metrics]
        ci_lowers = [ci_results[metric]['ci_lower'] for metric in metrics]
        ci_uppers = [ci_results[metric]['ci_upper'] for metric in metrics]
        errors = [ci_results[metric]['margin_error'] for metric in metrics]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x_pos = np.arange(len(metrics))
        
        # Create error bars
        ax.errorbar(x_pos, means, yerr=errors, fmt='o', capsize=5, capthick=2, 
                   markersize=8, color='blue', ecolor='red', linewidth=2)
        
        # Add confidence interval bars
        for i, metric in enumerate(metrics):
            ax.plot([i, i], [ci_lowers[i], ci_uppers[i]], 'r-', linewidth=3, alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('95% Confidence Intervals for NetPress Benchmark Metrics')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.title() for m in metrics])
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (mean, error) in enumerate(zip(means, errors)):
            ax.annotate(f'{mean:.3f} ± {error:.3f}', 
                       xy=(i, mean), xytext=(0, 10), 
                       textcoords='offset points', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confidence_intervals.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_box_plots(self, output_dir: str):
        """Create box plots for metrics"""
        metrics = ['correctness', 'safety', 'latency']
        data_to_plot = []
        labels = []
        
        for metric in metrics:
            if metric in self.df.columns:
                values = self.df[metric].dropna()
                data_to_plot.append(values)
                labels.append(metric.title())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('Box Plots for NetPress Benchmark Metrics')
        ax.set_ylabel('Values')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/box_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_correlation_plot(self, output_dir: str):
        """Create correlation matrix heatmap"""
        metrics = ['correctness', 'safety', 'latency']
        correlation_data = self.df[metrics].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        ax.set_title('Correlation Matrix of NetPress Benchmark Metrics')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_time_series_plots(self, output_dir: str):
        """Create time series plots for latency"""
        if 'latency' in self.df.columns:
            latencies = self.df['latency'].dropna()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Time series plot
            ax1.plot(range(len(latencies)), latencies, 'b-', linewidth=2, marker='o', markersize=4)
            ax1.set_title('Latency Time Series')
            ax1.set_xlabel('Query Number')
            ax1.set_ylabel('Latency (seconds)')
            ax1.grid(True, alpha=0.3)
            
            # Moving average
            window_size = min(5, len(latencies) // 3)
            if window_size > 1:
                moving_avg = latencies.rolling(window=window_size).mean()
                ax1.plot(range(len(moving_avg)), moving_avg, 'r-', linewidth=3, 
                        label=f'Moving Average (window={window_size})')
                ax1.legend()
            
            # Cumulative average
            cumulative_avg = latencies.expanding().mean()
            ax2.plot(range(len(cumulative_avg)), cumulative_avg, 'g-', linewidth=2)
            ax2.set_title('Cumulative Average Latency')
            ax2.set_xlabel('Query Number')
            ax2.set_ylabel('Cumulative Average (seconds)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/time_series.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_summary_dashboard(self, output_dir: str):
        """Create a comprehensive summary dashboard"""
        fig = plt.figure(figsize=(20, 15))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Summary statistics table
        ax1 = fig.add_subplot(gs[0, :2])
        stats_results = self.calculate_descriptive_statistics()
        
        # Create summary table
        summary_data = []
        for metric, stats in stats_results.items():
            summary_data.append([
                metric.title(),
                f"{stats['mean']:.3f}",
                f"{stats['std']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}"
            ])
        
        table = ax1.table(cellText=summary_data,
                         colLabels=['Metric', 'Mean', 'Std Dev', 'Min', 'Max'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax1.set_title('Summary Statistics', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. Confidence intervals
        ax2 = fig.add_subplot(gs[0, 2:])
        ci_results = self.calculate_confidence_intervals()
        
        metrics = list(ci_results.keys())
        means = [ci_results[metric]['mean'] for metric in metrics]
        errors = [ci_results[metric]['margin_error'] for metric in metrics]
        
        x_pos = np.arange(len(metrics))
        ax2.errorbar(x_pos, means, yerr=errors, fmt='o', capsize=5, capthick=2, 
                    markersize=8, color='blue', ecolor='red', linewidth=2)
        ax2.set_title('95% Confidence Intervals', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m.title() for m in metrics])
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution plots
        for i, metric in enumerate(['correctness', 'safety', 'latency']):
            if metric in self.df.columns:
                ax = fig.add_subplot(gs[1, i])
                values = self.df[metric].dropna()
                ax.hist(values, bins=15, alpha=0.7, color=f'C{i}', edgecolor='black')
                ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2)
                ax.set_title(f'{metric.title()} Distribution')
                ax.set_xlabel(metric.title())
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        # 4. Success rate pie chart
        ax4 = fig.add_subplot(gs[1, 3])
        success_count = self.df['success'].sum() if 'success' in self.df.columns else 0
        total_count = len(self.df)
        failure_count = total_count - success_count
        
        ax4.pie([success_count, failure_count], 
               labels=['Success', 'Failure'], 
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%',
               startangle=90)
        ax4.set_title('Success Rate')
        
        # 5. Latency over time
        ax5 = fig.add_subplot(gs[2, :])
        if 'latency' in self.df.columns:
            latencies = self.df['latency'].dropna()
            ax5.plot(range(len(latencies)), latencies, 'b-', linewidth=2, marker='o', markersize=3)
            ax5.set_title('Latency Over Time')
            ax5.set_xlabel('Query Number')
            ax5.set_ylabel('Latency (seconds)')
            ax5.grid(True, alpha=0.3)
        
        plt.suptitle('NetPress Benchmark Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.savefig(f'{output_dir}/summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_file: str = "statistical_report.md"):
        """
        Generate a comprehensive statistical report
        
        Args:
            output_file: Output file path for the report
        """
        ci_results = self.calculate_confidence_intervals()
        desc_stats = self.calculate_descriptive_statistics()
        normality_results = self.perform_normality_tests()
        
        report = f"""# NetPress Statistical Analysis Report

## Executive Summary

This report provides a comprehensive statistical analysis of the NetPress benchmark results for the AI4K8s MCP Agent.

### Key Findings

"""
        
        # Add key findings
        for metric, stats in desc_stats.items():
            ci = ci_results.get(metric, {})
            report += f"- **{metric.title()}**: Mean = {stats['mean']:.3f} (95% CI: {ci.get('ci_lower', 0):.3f} - {ci.get('ci_upper', 0):.3f})\n"
        
        report += f"""
## Descriptive Statistics

"""
        
        # Add descriptive statistics table
        for metric, stats in desc_stats.items():
            report += f"""
### {metric.title()}

| Statistic | Value |
|-----------|-------|
| Count | {stats['count']} |
| Mean | {stats['mean']:.4f} |
| Median | {stats['median']:.4f} |
| Standard Deviation | {stats['std']:.4f} |
| Variance | {stats['var']:.4f} |
| Minimum | {stats['min']:.4f} |
| Maximum | {stats['max']:.4f} |
| 25th Percentile | {stats['q25']:.4f} |
| 75th Percentile | {stats['q75']:.4f} |
| Interquartile Range | {stats['iqr']:.4f} |
| Skewness | {stats['skewness']:.4f} |
| Kurtosis | {stats['kurtosis']:.4f} |

"""
        
        report += """
## Confidence Intervals (95%)

"""
        
        # Add confidence intervals
        for metric, ci in ci_results.items():
            report += f"""
### {metric.title()}

- **Mean**: {ci['mean']:.4f}
- **Standard Error**: {ci['std_error']:.4f}
- **Margin of Error**: {ci['margin_error']:.4f}
- **95% Confidence Interval**: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]
- **Sample Size**: {ci['n']}

"""
        
        report += """
## Normality Tests

"""
        
        # Add normality test results
        for metric, tests in normality_results.items():
            report += f"""
### {metric.title()}

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Shapiro-Wilk | {tests['shapiro_stat']:.4f} | {tests['shapiro_p']:.4f} | {'Normal' if tests['shapiro_p'] > 0.05 else 'Not Normal'} |
| Kolmogorov-Smirnov | {tests['ks_stat']:.4f} | {tests['ks_p']:.4f} | {'Normal' if tests['ks_p'] > 0.05 else 'Not Normal'} |
| Anderson-Darling | {tests['ad_stat']:.4f} | Critical: {tests['ad_critical']:.4f} | {'Normal' if tests['ad_stat'] < tests['ad_critical'] else 'Not Normal'} |

"""
        
        report += """
## Statistical Interpretation

### Correctness Analysis
- The correctness metric measures how accurately the AI4K8s MCP agent responds to Kubernetes queries
- Higher values indicate better performance

### Safety Analysis  
- The safety metric evaluates how safely the agent handles potentially dangerous operations
- Higher values indicate safer behavior

### Latency Analysis
- The latency metric measures response time in seconds
- Lower values indicate better performance

## Recommendations

Based on the statistical analysis:

1. **Performance Optimization**: Focus on reducing latency while maintaining correctness
2. **Safety Improvements**: Enhance safety mechanisms for better protection
3. **Reliability**: Monitor consistency across different query types
4. **Scalability**: Test with larger query sets for more robust statistics

## Methodology

- **Confidence Level**: 95%
- **Statistical Tests**: Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling
- **Visualization**: Distribution plots, confidence intervals, correlation analysis
- **Sample Size**: Based on available benchmark data

---
*Report generated by NetPress Statistical Analyzer*
"""
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Statistical report saved to: {output_file}")

def main():
    """Main function to run statistical analysis"""
    parser = argparse.ArgumentParser(description="Statistical Analysis for NetPress Results")
    parser.add_argument("--results", required=True, help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", default="statistical_analysis", help="Output directory for plots")
    parser.add_argument("--report", default="statistical_report.md", help="Output file for statistical report")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level (default: 0.95)")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = NetPressStatisticalAnalyzer(args.results)
    
    print("Generating statistical analysis...")
    
    # Create visualizations
    analyzer.create_visualizations(args.output_dir)
    print(f"Visualizations saved to: {args.output_dir}/")
    
    # Generate report
    analyzer.generate_report(args.report)
    
    # Print summary
    ci_results = analyzer.calculate_confidence_intervals(args.confidence)
    desc_stats = analyzer.calculate_descriptive_statistics()
    
    print("\n=== Statistical Summary ===")
    for metric in ['correctness', 'safety', 'latency']:
        if metric in desc_stats:
            stats = desc_stats[metric]
            ci = ci_results.get(metric, {})
            print(f"{metric.title()}:")
            print(f"  Mean: {stats['mean']:.3f} ± {ci.get('margin_error', 0):.3f}")
            print(f"  95% CI: [{ci.get('ci_lower', 0):.3f}, {ci.get('ci_upper', 0):.3f}]")
            print(f"  Std Dev: {stats['std']:.3f}")
            print()

if __name__ == "__main__":
    main()
