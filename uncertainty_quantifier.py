#!/usr/bin/env python3
"""
AI4K8s Uncertainty Quantification Module
=========================================

Provides proper uncertainty estimates for:
1. Time series forecasts (prediction intervals that grow with horizon)
2. Anomaly detection (calibrated probabilities instead of binary flags)
3. Scaling decisions (replica confidence intervals + overload probabilities)

Addresses two critical limitations:
- Constant-width confidence intervals → horizon-dependent prediction intervals
- Binary anomaly detection → calibrated anomaly probabilities with severity distributions

Theory:
- Aleatoric uncertainty: irreducible noise in the data
- Epistemic uncertainty: model uncertainty, reducible with more data
- Total uncertainty: combination of both, propagated to decisions

Author: Pedram Nikjooy
Thesis: AI Agent for Kubernetes Management
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class UncertainForecast:
    """Forecast result with full uncertainty quantification"""
    metric_name: str
    current_value: float
    point_forecasts: List[float]
    prediction_intervals: List[Tuple[float, float]]
    aleatoric_uncertainty: List[float]    # per-horizon data noise
    epistemic_uncertainty: List[float]    # per-horizon model uncertainty
    total_uncertainty: List[float]        # per-horizon combined
    exceedance_probabilities: Dict[int, List[float]]  # threshold → P(metric > threshold) per hour
    confidence_level: float
    trend: str
    recommendation: str
    data_points_used: int
    model_quality: str  # 'good', 'limited', 'poor'


@dataclass
class CalibratedAnomaly:
    """Anomaly detection result with calibrated probability"""
    timestamp: datetime
    anomaly_probability: float    # calibrated P(anomaly) in [0, 1]
    raw_score: float              # original Isolation Forest score
    is_anomaly: bool              # binary (probability > 0.5)
    detection_confidence: float   # confidence in the detection itself
    severity_distribution: Dict[str, float]  # P(low), P(medium), P(high), P(critical)
    affected_metrics: List[str]
    recommendation: str


@dataclass
class UncertainScalingDecision:
    """Scaling decision with uncertainty propagation"""
    recommended_replicas: int
    replica_confidence_interval: Tuple[int, int]
    point_estimate: int
    probability_of_overload: float
    probability_of_underutilization: float
    risk_strategy: str  # 'conservative', 'balanced', 'aggressive'
    decision_uncertainty: int  # width of replica CI
    reasoning: str


class UncertaintyAwareForecaster:
    """
    Enhanced forecaster with proper uncertainty quantification.

    Key improvements over the original:
    1. Prediction intervals GROW with forecast horizon (sqrt scaling)
    2. Separates aleatoric (data noise) and epistemic (model) uncertainty
    3. Bootstrap resampling for model uncertainty estimation
    4. Exceedance probabilities: P(metric > threshold) per hour
    5. Model quality assessment based on data availability
    """

    def __init__(self, n_bootstrap: int = 50):
        """
        Args:
            n_bootstrap: Number of bootstrap resamples for epistemic uncertainty.
                         Higher = more accurate but slower. 50 is a good balance.
        """
        self.n_bootstrap = n_bootstrap

    def forecast_with_uncertainty(self, data: List[float], metric_name: str = "cpu",
                                   hours_ahead: int = 6,
                                   confidence_level: float = 0.95) -> UncertainForecast:
        """
        Generate forecasts with full uncertainty quantification.

        Args:
            data: Historical metric values (e.g., CPU % over time)
            metric_name: Name of the metric being forecast
            hours_ahead: Number of hours to predict ahead
            confidence_level: Confidence level for prediction intervals (default 95%)

        Returns:
            UncertainForecast with point forecasts, growing prediction intervals,
            separated uncertainty components, and exceedance probabilities
        """
        if len(data) < 5:
            return self._insufficient_data_response(metric_name, hours_ahead, confidence_level,
                                                     data[-1] if data else 0.0, len(data))

        data_arr = np.array(data, dtype=float)
        current_value = float(data_arr[-1])

        # Z-score for the requested confidence level
        z_score = self._confidence_to_z(confidence_level)

        # --- Fit trend on full data ---
        x = np.arange(len(data_arr))
        coeffs = np.polyfit(x, data_arr, 1)
        trend_values = np.polyval(coeffs, x)
        residuals = data_arr - trend_values

        # --- Aleatoric uncertainty: variance of residuals (data noise) ---
        aleatoric_var = float(np.var(residuals)) if len(residuals) > 1 else 1.0

        # --- Epistemic uncertainty via bootstrap ---
        bootstrap_predictions = {h: [] for h in range(1, hours_ahead + 1)}

        for _ in range(self.n_bootstrap):
            indices = np.random.choice(len(data_arr), size=len(data_arr), replace=True)
            boot_data = data_arr[indices]
            boot_x = x[indices]

            try:
                boot_coeffs = np.polyfit(boot_x, boot_data, 1)
                for h in range(1, hours_ahead + 1):
                    pred = np.polyval(boot_coeffs, len(data_arr) + h)
                    bootstrap_predictions[h].append(pred)
            except (np.linalg.LinAlgError, ValueError):
                # Degenerate bootstrap sample, skip
                continue

        # --- Build per-horizon uncertainty ---
        point_forecasts = []
        prediction_intervals = []
        aleatoric_list = []
        epistemic_list = []
        total_list = []
        exceedance_probs: Dict[int, List[float]] = {t: [] for t in [50, 70, 80, 90]}

        for h in range(1, hours_ahead + 1):
            # Point forecast from original trend
            point = float(np.polyval(coeffs, len(data_arr) + h))
            point = max(0.0, min(100.0, point))
            point_forecasts.append(point)

            # Epistemic variance from bootstrap spread
            boot_preds = np.array(bootstrap_predictions.get(h, [point]))
            if len(boot_preds) < 2:
                boot_preds = np.array([point])
            epistemic_var = float(np.var(boot_preds)) if len(boot_preds) > 1 else 0.0

            # Total uncertainty GROWS with horizon
            # Aleatoric accumulates (random walk), epistemic grows with extrapolation distance
            horizon_factor = np.sqrt(h)  # uncertainty grows as sqrt(horizon)
            total_var = aleatoric_var * horizon_factor + epistemic_var * h
            total_std = np.sqrt(max(total_var, 1e-10))

            # Prediction interval
            lower = max(0.0, point - z_score * total_std)
            upper = min(100.0, point + z_score * total_std)
            prediction_intervals.append((round(lower, 2), round(upper, 2)))

            aleatoric_list.append(round(np.sqrt(aleatoric_var), 2))
            epistemic_list.append(round(np.sqrt(epistemic_var), 2))
            total_list.append(round(total_std, 2))

            # Exceedance probabilities: P(metric > threshold)
            for threshold in [50, 70, 80, 90]:
                if total_std > 0.01:
                    # Using normal CDF approximation
                    z = (threshold - point) / total_std
                    prob = self._normal_sf(z)
                else:
                    prob = 1.0 if point > threshold else 0.0
                exceedance_probs[threshold].append(round(float(prob), 4))

        # Determine trend
        slope = coeffs[0]
        if slope > 0.5:
            trend_str = "increasing"
            rec = "Resource usage trending up. Consider proactive scaling."
        elif slope < -0.5:
            trend_str = "decreasing"
            rec = "Resource usage trending down. Consider right-sizing."
        else:
            trend_str = "stable"
            rec = "Resource usage is stable. Current allocation appears adequate."

        # Model quality assessment
        n = len(data)
        if n >= 50:
            quality = 'good'
        elif n >= 20:
            quality = 'limited'
        else:
            quality = 'poor'

        return UncertainForecast(
            metric_name=metric_name,
            current_value=current_value,
            point_forecasts=point_forecasts,
            prediction_intervals=prediction_intervals,
            aleatoric_uncertainty=aleatoric_list,
            epistemic_uncertainty=epistemic_list,
            total_uncertainty=total_list,
            exceedance_probabilities=exceedance_probs,
            confidence_level=confidence_level,
            trend=trend_str,
            recommendation=rec,
            data_points_used=n,
            model_quality=quality
        )

    def _insufficient_data_response(self, metric_name: str, hours_ahead: int,
                                     confidence_level: float, current: float,
                                     n_points: int) -> UncertainForecast:
        """Return a high-uncertainty response when data is insufficient"""
        point = [current] * hours_ahead
        wide_ci = [(max(0, current - 30), min(100, current + 30))] * hours_ahead
        high_unc = [30.0] * hours_ahead

        return UncertainForecast(
            metric_name=metric_name,
            current_value=current,
            point_forecasts=point,
            prediction_intervals=wide_ci,
            aleatoric_uncertainty=high_unc,
            epistemic_uncertainty=high_unc,
            total_uncertainty=high_unc,
            exceedance_probabilities={t: [0.5] * hours_ahead for t in [50, 70, 80, 90]},
            confidence_level=confidence_level,
            trend='insufficient_data',
            recommendation=f'Only {n_points} data points available. Need at least 5 for forecasting. Uncertainty is very high.',
            data_points_used=n_points,
            model_quality='poor'
        )

    @staticmethod
    def _confidence_to_z(confidence: float) -> float:
        """Convert confidence level to z-score"""
        z_map = {0.80: 1.282, 0.85: 1.440, 0.90: 1.645,
                 0.95: 1.960, 0.99: 2.576}
        return z_map.get(confidence, 1.96)

    @staticmethod
    def _normal_sf(z: float) -> float:
        """
        Survival function (1 - CDF) of standard normal distribution.
        Uses the complementary error function for numerical stability.
        No scipy dependency required.
        """
        # Approximation using the error function (math.erfc)
        import math
        return 0.5 * math.erfc(z / math.sqrt(2))


class CalibratedAnomalyDetector:
    """
    Wraps an Isolation Forest anomaly detector to produce calibrated probabilities.

    Instead of binary is_anomaly, returns:
    - anomaly_probability: calibrated P(anomaly) in [0, 1]
    - detection_confidence: how sure we are about the classification
    - severity_distribution: probability distribution over severity levels

    Uses Platt scaling (sigmoid calibration) on the raw Isolation Forest scores.
    The calibration parameters (A, B) can be fitted on validation data or use
    reasonable defaults based on the Isolation Forest score distribution.
    """

    def __init__(self, platt_A: float = -5.0, platt_B: float = -0.5):
        """
        Args:
            platt_A: Slope parameter for Platt scaling sigmoid
            platt_B: Intercept parameter for Platt scaling sigmoid

        Note: Isolation Forest score_samples() returns values typically in [-0.7, 0.3].
              More negative = more anomalous.
              Default A=-5.0, B=-0.5 maps score≈-0.3 to P≈0.85 (high anomaly probability)
              and score≈0.1 to P≈0.12 (low anomaly probability).
        """
        self.platt_A = platt_A
        self.platt_B = platt_B
        self._score_history: List[float] = []

    def calibrate_from_scores(self, historical_scores: List[float],
                               contamination: float = 0.1):
        """
        Auto-calibrate Platt parameters from historical Isolation Forest scores.

        Sets A and B so that the contamination fraction of scores maps to
        anomaly_probability > 0.5.

        Args:
            historical_scores: List of raw IF scores from training data
            contamination: Expected fraction of anomalies
        """
        if len(historical_scores) < 10:
            logger.warning("Not enough historical scores for calibration, using defaults")
            return

        scores = np.array(historical_scores)
        self._score_history = list(historical_scores)

        # Find the threshold that separates the contamination fraction
        threshold = np.percentile(scores, contamination * 100)

        # Set Platt parameters so that sigmoid(A*threshold + B) ≈ 0.5
        # and sigmoid(A*median + B) ≈ contamination_rate
        median_score = float(np.median(scores))
        score_range = float(np.std(scores))

        if score_range > 0:
            # Scale A based on score spread
            self.platt_A = -3.0 / score_range
            # Set B so the threshold maps to ~0.5
            self.platt_B = -self.platt_A * threshold
        else:
            self.platt_A = -5.0
            self.platt_B = -0.5

        logger.info(f"Calibrated Platt parameters: A={self.platt_A:.3f}, B={self.platt_B:.3f} "
                     f"(threshold={threshold:.3f}, median={median_score:.3f})")

    def compute_anomaly_probability(self, raw_score: float) -> Dict[str, Any]:
        """
        Convert a raw Isolation Forest score to a calibrated anomaly probability.

        Args:
            raw_score: Raw score from isolation_forest.score_samples()
                       (more negative = more anomalous)

        Returns:
            Dict with anomaly_probability, confidence, severity_distribution
        """
        # Platt scaling: P(anomaly) = sigmoid(A * score + B)
        logit = self.platt_A * raw_score + self.platt_B
        # Clip to avoid overflow
        logit = max(-20.0, min(20.0, logit))
        anomaly_probability = 1.0 / (1.0 + np.exp(-logit))

        # Detection confidence: how far from the decision boundary (0.5)
        detection_confidence = abs(anomaly_probability - 0.5) * 2.0

        # Severity distribution based on anomaly probability
        severity_distribution = self._compute_severity_distribution(anomaly_probability)

        return {
            'anomaly_probability': round(float(anomaly_probability), 4),
            'detection_confidence': round(float(detection_confidence), 4),
            'severity_distribution': severity_distribution
        }

    def detect_with_calibration(self, isolation_forest, scaler,
                                 feature_vector: np.ndarray,
                                 metrics_dict: Dict[str, float],
                                 timestamp: datetime = None) -> CalibratedAnomaly:
        """
        Full anomaly detection pipeline with calibrated probabilities.

        Args:
            isolation_forest: Trained IsolationForest model
            scaler: Fitted StandardScaler
            feature_vector: Raw feature vector (1 x n_features)
            metrics_dict: Dict of metric names to values for analysis
            timestamp: Detection timestamp

        Returns:
            CalibratedAnomaly with probability, confidence, and severity
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Scale features
        feature_scaled = scaler.transform(feature_vector.reshape(1, -1))

        # Get raw score
        raw_score = float(isolation_forest.score_samples(feature_scaled)[0])

        # Calibrate
        calibration = self.compute_anomaly_probability(raw_score)
        anomaly_prob = calibration['anomaly_probability']
        confidence = calibration['detection_confidence']
        severity_dist = calibration['severity_distribution']

        # Determine affected metrics (those most unusual)
        affected_metrics = self._identify_affected_metrics(metrics_dict)

        # Determine primary severity
        is_anomaly = anomaly_prob > 0.5

        # Generate recommendation based on probability
        recommendation = self._generate_recommendation(anomaly_prob, affected_metrics)

        return CalibratedAnomaly(
            timestamp=timestamp,
            anomaly_probability=anomaly_prob,
            raw_score=raw_score,
            is_anomaly=is_anomaly,
            detection_confidence=confidence,
            severity_distribution=severity_dist,
            affected_metrics=affected_metrics,
            recommendation=recommendation
        )

    def _compute_severity_distribution(self, anomaly_prob: float) -> Dict[str, float]:
        """
        Compute probability distribution over severity levels.

        Maps anomaly probability to severity using soft boundaries:
        - P < 0.3  → mostly 'low'
        - 0.3-0.5  → mostly 'medium'
        - 0.5-0.75 → mostly 'high'
        - P > 0.75 → mostly 'critical'
        """
        # Use softmax-like approach over severity thresholds
        p = anomaly_prob

        if p < 0.2:
            dist = {'low': 0.85, 'medium': 0.12, 'high': 0.02, 'critical': 0.01}
        elif p < 0.4:
            low_w = max(0, 0.5 - p)
            med_w = 0.4
            high_w = p * 0.3
            crit_w = 0.01
            total = low_w + med_w + high_w + crit_w
            dist = {'low': low_w/total, 'medium': med_w/total,
                    'high': high_w/total, 'critical': crit_w/total}
        elif p < 0.6:
            dist = {'low': 0.05, 'medium': 0.35, 'high': 0.45, 'critical': 0.15}
        elif p < 0.8:
            dist = {'low': 0.02, 'medium': 0.13, 'high': 0.50, 'critical': 0.35}
        else:
            dist = {'low': 0.01, 'medium': 0.04, 'high': 0.25, 'critical': 0.70}

        # Normalize and round
        total = sum(dist.values())
        return {k: round(v / total, 4) for k, v in dist.items()}

    def _identify_affected_metrics(self, metrics_dict: Dict[str, float]) -> List[str]:
        """Identify which metrics are most unusual"""
        affected = []
        thresholds = {
            'cpu_usage': (5.0, 90.0),
            'memory_usage': (5.0, 90.0),
            'network_io': (0.0, 1000.0),
            'disk_io': (0.0, 1000.0),
        }

        for metric, (low, high) in thresholds.items():
            val = metrics_dict.get(metric, 0)
            if val > high or val < low:
                affected.append(metric)

        if not affected:
            affected = ['statistical_pattern']

        return affected

    def _generate_recommendation(self, anomaly_prob: float,
                                  affected_metrics: List[str]) -> str:
        """Generate recommendation based on anomaly probability"""
        if anomaly_prob > 0.8:
            prefix = "CRITICAL anomaly detected (P={:.0%}).".format(anomaly_prob)
        elif anomaly_prob > 0.6:
            prefix = "High probability anomaly (P={:.0%}).".format(anomaly_prob)
        elif anomaly_prob > 0.4:
            prefix = "Moderate anomaly probability (P={:.0%}).".format(anomaly_prob)
        else:
            return "Metrics appear normal (anomaly probability: {:.0%}).".format(anomaly_prob)

        metric_advice = {
            'cpu_usage': "Check CPU-intensive processes and consider scaling.",
            'memory_usage': "Investigate memory leaks or increase allocation.",
            'network_io': "Monitor network traffic for unusual patterns.",
            'disk_io': "Check for storage bottlenecks or I/O-intensive operations.",
            'statistical_pattern': "Unusual combination of metrics detected."
        }

        advice_parts = [prefix]
        for m in affected_metrics[:3]:  # Limit to top 3
            advice_parts.append(metric_advice.get(m, f"Check {m}."))

        return " ".join(advice_parts)


class UncertaintyAwareScaler:
    """
    Propagates forecast uncertainty into scaling decisions.

    Instead of "scale to 5 replicas", returns:
    "scale to 5 replicas (95% CI: 4-6), P(overload at 4) = 23%"

    Uses the forecast's exceedance probabilities and prediction intervals
    to compute:
    - Replica confidence intervals
    - Overload probability at different replica counts
    - Risk-adjusted recommendations (conservative/balanced/aggressive)
    """

    def __init__(self, target_cpu: float = 70.0, target_memory: float = 80.0):
        """
        Args:
            target_cpu: Target CPU utilization percentage
            target_memory: Target memory utilization percentage
        """
        self.target_cpu = target_cpu
        self.target_memory = target_memory

    def decide_with_uncertainty(self, cpu_forecast: UncertainForecast,
                                 memory_forecast: Optional[UncertainForecast],
                                 current_replicas: int,
                                 min_replicas: int = 1,
                                 max_replicas: int = 10) -> UncertainScalingDecision:
        """
        Given uncertain forecasts, compute uncertain scaling recommendations.

        Args:
            cpu_forecast: UncertainForecast for CPU
            memory_forecast: UncertainForecast for memory (optional)
            current_replicas: Current number of replicas
            min_replicas: Minimum replicas
            max_replicas: Maximum replicas

        Returns:
            UncertainScalingDecision with confidence intervals and probabilities
        """
        safe_current = max(current_replicas, 1)

        # --- CPU-based replica estimates ---
        point_forecasts = cpu_forecast.point_forecasts
        intervals = cpu_forecast.prediction_intervals
        exceedance = cpu_forecast.exceedance_probabilities

        # Peak values from point forecast and interval bounds
        peak_point = max(point_forecasts) if point_forecasts else cpu_forecast.current_value
        peak_lower = max(lo for lo, hi in intervals) if intervals else peak_point
        peak_upper = max(hi for lo, hi in intervals) if intervals else peak_point

        # Compute replica needs for each estimate
        replicas_point = self._replicas_for_cpu(peak_point, safe_current, min_replicas, max_replicas)
        replicas_lower = self._replicas_for_cpu(peak_lower, safe_current, min_replicas, max_replicas)
        replicas_upper = self._replicas_for_cpu(peak_upper, safe_current, min_replicas, max_replicas)

        # --- Factor in memory if available ---
        if memory_forecast:
            mem_peak = max(memory_forecast.point_forecasts) if memory_forecast.point_forecasts else 50.0
            mem_peak_upper = max(hi for lo, hi in memory_forecast.prediction_intervals) if memory_forecast.prediction_intervals else mem_peak
            mem_replicas = self._replicas_for_memory(mem_peak, safe_current, min_replicas, max_replicas)
            mem_replicas_upper = self._replicas_for_memory(mem_peak_upper, safe_current, min_replicas, max_replicas)
            replicas_point = max(replicas_point, mem_replicas)
            replicas_upper = max(replicas_upper, mem_replicas_upper)

        # --- Probability of overload at current replicas ---
        p_overload = max(exceedance.get(70, [0.0])) if exceedance else 0.0

        # --- Probability of underutilization ---
        # P(CPU < 30%) at recommended replicas ≈ P(peak < 30%)
        p_under = 0.0
        if exceedance.get(50):
            p_not_over_50 = [1.0 - p for p in exceedance[50]]
            p_under = max(p_not_over_50) if p_not_over_50 else 0.0
            # Rough: if at recommended replicas, CPU would be even lower
            if replicas_point > safe_current:
                p_under = min(1.0, p_under * (replicas_point / safe_current))

        # --- Risk-adjusted recommendation ---
        if p_overload > 0.7:
            recommended = replicas_upper
            risk_strategy = 'conservative'
        elif p_overload > 0.3:
            recommended = replicas_point
            risk_strategy = 'balanced'
        else:
            recommended = replicas_lower
            risk_strategy = 'aggressive'

        # Clamp
        recommended = max(min_replicas, min(recommended, max_replicas))
        replicas_lower = max(min_replicas, min(replicas_lower, max_replicas))
        replicas_upper = max(min_replicas, min(replicas_upper, max_replicas))

        # Decision uncertainty
        decision_unc = replicas_upper - replicas_lower

        # Build reasoning
        reasoning_parts = [
            f"Point estimate: {replicas_point} replicas (peak CPU forecast: {peak_point:.1f}%).",
            f"{cpu_forecast.confidence_level*100:.0f}% CI: {replicas_lower}-{replicas_upper} replicas "
            f"(peak CPU range: {peak_lower:.1f}%-{peak_upper:.1f}%).",
            f"P(overload at current={safe_current}): {p_overload:.1%}.",
            f"P(underutilization): {p_under:.1%}.",
            f"Strategy: {risk_strategy} → {recommended} replicas.",
            f"Forecast quality: {cpu_forecast.model_quality} ({cpu_forecast.data_points_used} data points).",
        ]

        return UncertainScalingDecision(
            recommended_replicas=recommended,
            replica_confidence_interval=(replicas_lower, replicas_upper),
            point_estimate=replicas_point,
            probability_of_overload=round(float(p_overload), 4),
            probability_of_underutilization=round(float(p_under), 4),
            risk_strategy=risk_strategy,
            decision_uncertainty=decision_unc,
            reasoning=' '.join(reasoning_parts)
        )

    def _replicas_for_cpu(self, peak_cpu: float, current_replicas: int,
                           min_r: int, max_r: int) -> int:
        """Calculate replicas needed to keep CPU at target"""
        if peak_cpu <= 0 or self.target_cpu <= 0:
            return current_replicas
        needed = int(np.ceil(current_replicas * (peak_cpu / self.target_cpu)))
        return max(min_r, min(needed, max_r))

    def _replicas_for_memory(self, peak_mem: float, current_replicas: int,
                              min_r: int, max_r: int) -> int:
        """Calculate replicas needed to keep memory at target"""
        if peak_mem <= 0 or self.target_memory <= 0:
            return current_replicas
        needed = int(np.ceil(current_replicas * (peak_mem / self.target_memory)))
        return max(min_r, min(needed, max_r))
