# NetPress Statistical Analysis

This directory contains comprehensive statistical analysis tools for NetPress benchmark results, including confidence intervals, normality tests, and advanced visualizations.

## Overview

The statistical analysis provides:

- **Confidence Intervals**: 95% confidence intervals for all metrics
- **Descriptive Statistics**: Mean, median, standard deviation, quartiles, skewness, kurtosis
- **Normality Tests**: Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling tests
- **Visualizations**: Distribution plots, box plots, correlation matrices, time series
- **Comparative Analysis**: Statistical comparison between different benchmark runs

## Files

- `statistical_analyzer.py` - Main statistical analysis tool
- `run_analysis.sh` - Automated analysis runner script
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Analysis on Existing Results

```bash
# Analyze a specific results file
python3 statistical_analyzer.py --results ../test_results.json --output-dir analysis_output

# Run automated analysis (finds latest results)
./run_analysis.sh
```

### 3. View Results

```bash
# View generated plots
open analysis_output/summary_dashboard.png
open analysis_output/confidence_intervals.png
open analysis_output/distributions.png

# View statistical report
cat statistical_report.md
```

## Statistical Methods

### Confidence Intervals

- **Method**: t-distribution based confidence intervals
- **Confidence Level**: 95% (configurable)
- **Formula**: CI = mean ± t(α/2, df) × (std_error)
- **Degrees of Freedom**: n - 1

### Normality Tests

1. **Shapiro-Wilk Test**: Best for small samples (n ≤ 5000)
2. **Kolmogorov-Smirnov Test**: Good for larger samples
3. **Anderson-Darling Test**: More sensitive to tail deviations

### Descriptive Statistics

- **Central Tendency**: Mean, median
- **Variability**: Standard deviation, variance, IQR
- **Distribution Shape**: Skewness, kurtosis
- **Range**: Min, max, quartiles

## Generated Visualizations

### 1. Summary Dashboard
- **File**: `summary_dashboard.png`
- **Content**: Comprehensive overview with statistics table, confidence intervals, distributions, and success rate

### 2. Confidence Intervals Plot
- **File**: `confidence_intervals.png`
- **Content**: 95% confidence intervals for correctness, safety, and latency metrics

### 3. Distribution Plots
- **File**: `distributions.png`
- **Content**: Histograms with KDE curves and mean lines for each metric

### 4. Box Plots
- **File**: `box_plots.png`
- **Content**: Box plots showing quartiles, outliers, and distribution shape

### 5. Correlation Matrix
- **File**: `correlation_matrix.png`
- **Content**: Heatmap showing correlations between metrics

### 6. Time Series Analysis
- **File**: `time_series.png`
- **Content**: Latency over time with moving average and cumulative average

## Example Results

### AI4K8s MCP Agent Performance

Based on the latest benchmark results:

```
=== Statistical Summary ===
Correctness:
  Mean: 0.610 ± 0.163
  95% CI: [0.447, 0.773]
  Std Dev: 0.228

Safety:
  Mean: 0.460 ± 0.244
  95% CI: [0.216, 0.704]
  Std Dev: 0.341

Latency:
  Mean: 4.717 ± 1.266
  95% CI: [3.452, 5.983]
  Std Dev: 1.769
```

### Interpretation

- **Correctness (0.610)**: Moderate accuracy in responses
- **Safety (0.460)**: Room for improvement in safe operation handling
- **Latency (4.717s)**: Reasonable response time for complex queries

## Advanced Usage

### Custom Confidence Level

```bash
python3 statistical_analyzer.py --results results.json --confidence 0.99
```

### Batch Analysis

```bash
# Analyze multiple result files
for file in ../netpress-results/*.json; do
    python3 statistical_analyzer.py --results "$file" --output-dir "analysis_$(basename "$file" .json)"
done
```

### Comparative Analysis

The `run_analysis.sh` script automatically creates comparative analysis when both basic and advanced benchmark results are available.

## Statistical Report Format

The generated report includes:

1. **Executive Summary**: Key findings and metrics
2. **Descriptive Statistics**: Detailed statistics tables
3. **Confidence Intervals**: 95% CI with interpretation
4. **Normality Tests**: Test results and interpretations
5. **Statistical Interpretation**: Analysis of each metric
6. **Recommendations**: Based on statistical findings

## Academic Use

This statistical analysis is designed for academic research and thesis documentation:

- **Rigorous Methodology**: Uses established statistical methods
- **Comprehensive Metrics**: Multiple evaluation dimensions
- **Visual Documentation**: Publication-ready plots and charts
- **Reproducible Results**: Automated analysis pipeline

## Integration with NetPress

The statistical analysis integrates seamlessly with the NetPress benchmarking framework:

- **Standardized Input**: Accepts NetPress JSON result format
- **Research-Grade Output**: Academic-quality analysis and visualizations
- **Comparative Studies**: Enables comparison with other NetPress benchmarks
- **Publication Ready**: Results suitable for academic papers

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install requirements.txt
2. **File Not Found**: Ensure benchmark results exist
3. **Permission Errors**: Check file permissions
4. **Memory Issues**: Reduce sample size for large datasets

### Performance Tips

- Use smaller confidence levels for faster computation
- Process results in batches for large datasets
- Use headless matplotlib for server environments

## References

- NetPress Framework: https://github.com/Froot-NetSys/NetPress
- Statistical Methods: Scipy Documentation
- Visualization: Matplotlib and Seaborn Documentation
- Confidence Intervals: Statistical Inference Theory

## License

This statistical analysis tool follows the same license as the main AI4K8s project.
