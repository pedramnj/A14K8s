#!/usr/bin/env python3
"""
AI4K8s Multi-Criteria Decision Analysis (MCDA) Optimizer
=========================================================

Replaces simple threshold-based heuristics with formal multi-criteria
optimization using the TOPSIS method (Technique for Order of Preference
by Similarity to Ideal Solution).

Instead of:
    IF cpu > 75% → scale_up
    IF cpu < 25% → scale_down

Now uses:
    1. Generate candidate scaling alternatives
    2. Evaluate each on weighted criteria (cost, performance, stability, etc.)
    3. TOPSIS ranking → select Pareto-optimal alternative
    4. Provide dominance margin and full ranking transparency

Author: Pedram Nikjooy
Thesis: AI Agent for Kubernetes Management
"""

import os
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

# --- Cost-criterion compression knob (Phase H, Knob 2) ----------------------
# Applies a non-linear transform to the per-alternative cost score:
#     cost = (target / max_replicas) ** MCDA_COST_EXPONENT
# Default 1.0 reproduces the Phase G behaviour (linear cost). Setting to 0.5
# applies sqrt compression: cost values for [2, 3, 4] replicas with max=4 go
# from [0.50, 0.75, 1.00] down to [0.71, 0.87, 1.00], shrinking the cost
# differential between adjacent actions and letting Performance carry more
# weight in TOPSIS distances. See thesis_reports/EASY_EXPLAINER.md (Phase H).
MCDA_COST_EXPONENT = float(os.getenv("MCDA_COST_EXPONENT", "1.0"))

# --- VPA restart penalty (Phase I) ------------------------------------------
# Vertical scaling causes the pod to restart; horizontal scaling does not.
# When a vertical alternative is evaluated, add this much to its
# stability_risk so MCDA does not over-prefer VPA just because it has no
# flap-tracking history. Env-gated for bracketing.
MCDA_VPA_RESTART_PENALTY = float(os.getenv("MCDA_VPA_RESTART_PENALTY", "0.3"))

# --- VPA multiplier grid (Phase I) ------------------------------------------
# Per-pod CPU and memory requests are scaled by these multipliers against
# the current request to form the vertical candidate pool. The (1.0, 1.0)
# tuple is excluded as it represents "no change". Kept small to keep the
# combined alternative-pool size manageable for TOPSIS.
VPA_MULTIPLIER_GRID = (0.7, 1.0, 1.3, 1.6)

logger = logging.getLogger(__name__)


@dataclass
class ScalingAlternative:
    """One candidate scaling action to be evaluated by MCDA.

    Horizontal alternatives change ``target_replicas`` and leave
    ``target_cpu_m``/``target_memory_mi`` at the sentinel ``-1``
    ("unchanged"). Vertical alternatives keep ``target_replicas`` equal
    to the current count and change the per-pod resource fields.
    """
    name: str
    target_replicas: int
    target_cpu_m: int = -1              # millicores; -1 = unchanged
    target_memory_mi: int = -1          # MiB;        -1 = unchanged
    scaling_type: str = 'hpa'           # 'hpa' or 'vpa'
    estimated_cost: float = 0.0         # normalized 0-1 (lower is better)
    estimated_performance: float = 0.0  # normalized 0-1 (higher is better)
    stability_risk: float = 0.0         # normalized 0-1 (lower is better)
    forecast_alignment: float = 0.0     # how well it matches forecast (higher is better)
    response_time: float = 0.0          # estimated latency impact 0-1 (lower is better)


@dataclass
class MCDAResult:
    """Result of MCDA optimization"""
    best_alternative: str
    target_replicas: int
    action: str
    mcda_score: float
    dominance_margin: float
    alternatives_evaluated: int
    ranking: List[Tuple[str, float]]
    criteria_weights: Dict[str, float]
    reasoning: str
    criteria_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Phase I additions — populated when a vertical alternative wins.
    target_cpu_m: int = -1
    target_memory_mi: int = -1
    scaling_type: str = 'hpa'


class MCDAAutoscalingOptimizer:
    """
    Multi-Criteria Decision Analysis for autoscaling using TOPSIS.

    Criteria and their optimization direction:
        - cost:               minimize (fewer replicas = lower cost)
        - performance:        maximize (CPU in sweet spot = better perf)
        - stability:          minimize (smaller change = more stable)
        - forecast_alignment: maximize (match predicted demand)
        - response_time:      minimize (lower estimated latency)

    Weights are user-configurable and default to a balanced profile
    that prioritizes performance and stability.
    """

    # Pre-defined weight profiles for different operational priorities
    WEIGHT_PROFILES = {
        'balanced': {
            'cost': 0.15,
            'performance': 0.30,
            'stability': 0.25,
            'forecast_alignment': 0.25,
            'response_time': 0.05
        },
        'performance_first': {
            'cost': 0.10,
            'performance': 0.40,
            'stability': 0.15,
            'forecast_alignment': 0.20,
            'response_time': 0.15
        },
        'cost_optimized': {
            'cost': 0.40,
            'performance': 0.20,
            'stability': 0.20,
            'forecast_alignment': 0.10,
            'response_time': 0.10
        },
        'stability_first': {
            'cost': 0.15,
            'performance': 0.20,
            'stability': 0.40,
            'forecast_alignment': 0.15,
            'response_time': 0.10
        }
    }

    # Criteria metadata: name → (attribute, is_benefit)
    # is_benefit=True means higher is better; False means lower is better
    CRITERIA = [
        ('estimated_cost',        'cost',               False),
        ('estimated_performance', 'performance',        True),
        ('stability_risk',        'stability',          False),
        ('forecast_alignment',    'forecast_alignment', True),
        ('response_time',         'response_time',      False),
    ]

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 profile: str = 'balanced'):
        """
        Initialize MCDA optimizer.

        Args:
            weights: Custom criteria weights dict. If None, uses profile.
            profile: One of 'balanced', 'performance_first', 'cost_optimized',
                     'stability_first'. Ignored if weights is provided.
        """
        if weights:
            self.weights = dict(weights)
        else:
            self.weights = dict(self.WEIGHT_PROFILES.get(profile, self.WEIGHT_PROFILES['balanced']))

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(f"MCDA optimizer initialized with weights: {self.weights}")

    def generate_alternatives(self, current_replicas: int,
                               min_replicas: int, max_replicas: int,
                               metrics: Dict[str, Any],
                               forecast: Dict[str, Any],
                               current_cpu_m: Optional[int] = None,
                               current_memory_mi: Optional[int] = None,
                               ) -> List[ScalingAlternative]:
        """
        Generate candidate horizontal scaling actions to evaluate.

        Creates alternatives around the current replica count, including:
        - The current count (maintain)
        - Small increments/decrements (±1, ±2)
        - The "ideal" count based on CPU targeting 70%
        - Min and max boundaries

        Args:
            current_replicas: Current number of replicas
            min_replicas: Minimum allowed replicas
            max_replicas: Maximum allowed replicas
            metrics: Current metrics dict with 'cpu_percent', 'memory_percent'
            forecast: Forecast dict with 'predicted_cpu', 'predicted_memory', 'cpu_trend'
            current_cpu_m: Current per-pod CPU request (millicores). Optional;
                when supplied together with current_memory_mi, criterion
                scores use the unified Phase-I model. When ``None``, the
                Phase H horizontal-only formulas are used unchanged.
            current_memory_mi: Current per-pod memory request (MiB). See above.

        Returns:
            List of horizontal-axis ScalingAlternative objects.
        """
        cpu = metrics.get('cpu_percent', 50.0)
        mem = metrics.get('memory_percent', 50.0)
        cpu_forecast = forecast.get('predicted_cpu', [cpu] * 6)
        mem_forecast = forecast.get('predicted_memory', [mem] * 6)
        cpu_trend = forecast.get('cpu_trend', 'stable')

        if not cpu_forecast:
            cpu_forecast = [cpu] * 6
        if not mem_forecast:
            mem_forecast = [mem] * 6

        peak_cpu = max(cpu_forecast)
        peak_mem = max(mem_forecast)

        # Generate candidate replica counts
        candidates = set()
        candidates.add(current_replicas)

        for delta in [-2, -1, 1, 2, 3]:
            r = current_replicas + delta
            if min_replicas <= r <= max_replicas:
                candidates.add(r)

        # Add boundary values
        candidates.add(min_replicas)
        candidates.add(max_replicas)

        # Add "ideal" count based on CPU target of 70%
        if peak_cpu > 0 and current_replicas > 0:
            ideal = int(np.ceil(current_replicas * (peak_cpu / 70.0)))
            ideal = max(min_replicas, min(ideal, max_replicas))
            candidates.add(ideal)

        # Add ideal based on average forecast
        avg_forecast_cpu = float(np.mean(cpu_forecast))
        if avg_forecast_cpu > 0 and current_replicas > 0:
            ideal_avg = int(np.ceil(current_replicas * (avg_forecast_cpu / 70.0)))
            ideal_avg = max(min_replicas, min(ideal_avg, max_replicas))
            candidates.add(ideal_avg)

        alternatives = []
        for target in sorted(candidates):
            alt = self._evaluate_alternative(
                target_replicas=target,
                target_cpu_m=current_cpu_m if current_cpu_m is not None else -1,
                target_memory_mi=current_memory_mi if current_memory_mi is not None else -1,
                current_replicas=current_replicas,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                current_cpu_m=current_cpu_m,
                current_memory_mi=current_memory_mi,
                cpu=cpu, mem=mem,
                peak_cpu=peak_cpu, peak_mem=peak_mem,
                cpu_forecast=cpu_forecast, mem_forecast=mem_forecast,
                cpu_trend=cpu_trend,
                scaling_type='hpa',
            )
            alternatives.append(alt)

        return alternatives

    def generate_vpa_alternatives(self, current_replicas: int,
                                   min_replicas: int, max_replicas: int,
                                   current_cpu_m: int,
                                   current_memory_mi: int,
                                   metrics: Dict[str, Any],
                                   forecast: Dict[str, Any],
                                   multipliers: Tuple[float, ...] = VPA_MULTIPLIER_GRID,
                                   ) -> List[ScalingAlternative]:
        """
        Generate candidate vertical scaling actions (CPU/memory request changes).

        Emits the cross product of ``multipliers`` over CPU and memory,
        excluding the (1.0, 1.0) no-change tuple. Replica count is held
        at ``current_replicas`` for every vertical alternative.

        Args:
            current_replicas: Current replica count (held constant).
            min_replicas / max_replicas: Used by the cost normaliser only.
            current_cpu_m: Current per-pod CPU request in millicores.
            current_memory_mi: Current per-pod memory request in MiB.
            metrics: Current metrics dict with 'cpu_percent', 'memory_percent'.
            forecast: Forecast dict (same shape as generate_alternatives).
            multipliers: Per-axis multiplier grid. Default is the Phase-I
                grid (0.7, 1.0, 1.3, 1.6).

        Returns:
            List of vertical-axis ScalingAlternative objects.
        """
        if current_cpu_m is None or current_memory_mi is None:
            return []

        cpu = metrics.get('cpu_percent', 50.0)
        mem = metrics.get('memory_percent', 50.0)
        cpu_forecast = forecast.get('predicted_cpu', [cpu] * 6)
        mem_forecast = forecast.get('predicted_memory', [mem] * 6)
        cpu_trend = forecast.get('cpu_trend', 'stable')

        if not cpu_forecast:
            cpu_forecast = [cpu] * 6
        if not mem_forecast:
            mem_forecast = [mem] * 6

        peak_cpu = max(cpu_forecast)
        peak_mem = max(mem_forecast)

        alternatives = []
        for cpu_mult in multipliers:
            for mem_mult in multipliers:
                if cpu_mult == 1.0 and mem_mult == 1.0:
                    continue  # baseline = "no change"; already in horizontal pool
                target_cpu_m = max(1, int(round(current_cpu_m * cpu_mult)))
                target_memory_mi = max(1, int(round(current_memory_mi * mem_mult)))
                alt = self._evaluate_alternative(
                    target_replicas=current_replicas,
                    target_cpu_m=target_cpu_m,
                    target_memory_mi=target_memory_mi,
                    current_replicas=current_replicas,
                    min_replicas=min_replicas,
                    max_replicas=max_replicas,
                    current_cpu_m=current_cpu_m,
                    current_memory_mi=current_memory_mi,
                    cpu=cpu, mem=mem,
                    peak_cpu=peak_cpu, peak_mem=peak_mem,
                    cpu_forecast=cpu_forecast, mem_forecast=mem_forecast,
                    cpu_trend=cpu_trend,
                    scaling_type='vpa',
                )
                alternatives.append(alt)

        return alternatives

    def _evaluate_alternative(self, target_replicas: int,
                               target_cpu_m: int,
                               target_memory_mi: int,
                               current_replicas: int,
                               min_replicas: int, max_replicas: int,
                               current_cpu_m: Optional[int],
                               current_memory_mi: Optional[int],
                               cpu: float, mem: float,
                               peak_cpu: float, peak_mem: float,
                               cpu_forecast: List[float],
                               mem_forecast: List[float],
                               cpu_trend: str,
                               scaling_type: str = 'hpa',
                               ) -> ScalingAlternative:
        """
        Evaluate a single candidate (horizontal or vertical) across all
        criteria. Returns a ScalingAlternative with scores in [0, 1].

        The Phase-I unified formulas reduce exactly to the Phase-H
        horizontal-only behaviour when ``current_cpu_m`` is ``None`` and
        ``target_cpu_m == -1`` (i.e. the caller did not supply resource
        information). This keeps every existing eval reproducible.
        """
        safe_target = max(target_replicas, 1)
        safe_current = max(current_replicas, 1)

        # Resolve effective per-pod resource fields. When resources are not
        # supplied, fall back to a synthetic "unchanged" ratio of 1.0.
        unified = current_cpu_m is not None and current_memory_mi is not None
        eff_target_cpu_m = target_cpu_m if target_cpu_m > 0 else (
            current_cpu_m if current_cpu_m is not None else 1
        )
        eff_target_mem_mi = target_memory_mi if target_memory_mi > 0 else (
            current_memory_mi if current_memory_mi is not None else 1
        )
        eff_current_cpu_m = current_cpu_m if current_cpu_m is not None else 1
        eff_current_mem_mi = current_memory_mi if current_memory_mi is not None else 1

        # Capacity ratios: how much total CPU / memory capacity does this
        # alternative provide relative to the current deployment?
        cpu_ratio = (safe_target * eff_target_cpu_m) / max(safe_current * eff_current_cpu_m, 1)
        mem_ratio = (safe_target * eff_target_mem_mi) / max(safe_current * eff_current_mem_mi, 1)
        cpu_ratio = max(cpu_ratio, 1e-6)
        mem_ratio = max(mem_ratio, 1e-6)

        # --- Cost --------------------------------------------------------
        # Phase-I unified: prices both pod count and pod size against the
        # worst-case point of the grid (max_replicas × max-multiplier of
        # current cpu request). Phase-H linear formula is restored when
        # resources are unknown.
        if unified:
            max_grid_cpu_m = max(VPA_MULTIPLIER_GRID) * eff_current_cpu_m
            cost_normalizer = max(max_replicas, 1) * max(max_grid_cpu_m, 1)
            cost_raw = (safe_target * eff_target_cpu_m) / cost_normalizer
        else:
            cost_raw = target_replicas / max(max_replicas, 1)
        cost_raw = max(0.0, min(1.0, cost_raw))
        cost = cost_raw ** MCDA_COST_EXPONENT

        # --- Performance: CPU pushed into the 70% sweet spot -------------
        estimated_cpu = cpu / cpu_ratio
        estimated_cpu = max(0, min(100, estimated_cpu))
        perf = 1.0 - abs(estimated_cpu - 70.0) / 70.0
        perf = max(0.0, min(1.0, perf))

        # --- Stability ---------------------------------------------------
        change_magnitude = abs(target_replicas - current_replicas)
        replica_range = max(max_replicas - min_replicas, 1)
        stability_risk = change_magnitude / replica_range

        # Oscillation penalty (same as Phase H)
        if target_replicas < current_replicas and cpu_trend == 'increasing':
            stability_risk += 0.2

        # Phase-I restart penalty: any change to the per-pod resource
        # request triggers a pod restart, which is a real disruption HPA
        # never incurs. Apply the penalty whenever the resource fields
        # actually differ from the current values.
        if unified and (
            (target_cpu_m > 0 and target_cpu_m != current_cpu_m) or
            (target_memory_mi > 0 and target_memory_mi != current_memory_mi)
        ):
            stability_risk += MCDA_VPA_RESTART_PENALTY

        stability_risk = max(0.0, min(1.0, stability_risk))

        # --- Forecast alignment ------------------------------------------
        estimated_peak_cpu = peak_cpu / cpu_ratio
        estimated_peak_cpu = max(0, min(100, estimated_peak_cpu))
        if estimated_peak_cpu <= 75:
            alignment = 1.0 - max(0, (50 - estimated_peak_cpu)) / 50.0
        else:
            alignment = 1.0 - (estimated_peak_cpu - 75) / 25.0
        alignment = max(0.0, min(1.0, alignment))

        estimated_peak_mem = peak_mem / mem_ratio
        estimated_peak_mem = max(0, min(100, estimated_peak_mem))
        if estimated_peak_mem <= 80:
            mem_alignment = 1.0 - max(0, (40 - estimated_peak_mem)) / 40.0
        else:
            mem_alignment = 1.0 - (estimated_peak_mem - 80) / 20.0
        mem_alignment = max(0.0, min(1.0, mem_alignment))

        alignment = 0.7 * alignment + 0.3 * mem_alignment

        # --- Response time -----------------------------------------------
        avg_forecast = float(np.mean(cpu_forecast)) if cpu_forecast else cpu
        estimated_avg_cpu = avg_forecast / cpu_ratio
        estimated_avg_cpu = max(0, min(100, estimated_avg_cpu))
        response_time = estimated_avg_cpu / 100.0

        # Naming: keeps the existing "scale_to_N" form for horizontal so
        # the LLM-decision lookup in validate_llm_decision still works.
        if scaling_type == 'vpa':
            name = f"vpa_cpu{eff_target_cpu_m}m_mem{eff_target_mem_mi}Mi"
        else:
            name = f"scale_to_{target_replicas}"

        return ScalingAlternative(
            name=name,
            target_replicas=target_replicas,
            target_cpu_m=target_cpu_m,
            target_memory_mi=target_memory_mi,
            scaling_type=scaling_type,
            estimated_cost=float(cost),
            estimated_performance=float(perf),
            stability_risk=float(stability_risk),
            forecast_alignment=float(alignment),
            response_time=float(response_time)
        )

    def topsis_rank(self, alternatives: List[ScalingAlternative]) -> List[Tuple[ScalingAlternative, float]]:
        """
        TOPSIS: Rank alternatives by closeness to the ideal solution.

        Steps:
            1. Build decision matrix [alternatives x criteria]
            2. Normalize using vector normalization
            3. Apply criteria weights
            4. Determine ideal (best) and anti-ideal (worst) solutions
            5. Calculate distance from each alternative to ideal/anti-ideal
            6. Compute closeness coefficient (score)

        Returns:
            List of (alternative, score) sorted by score descending.
            Score is in [0, 1] where 1.0 = ideal solution.
        """
        if not alternatives:
            return []

        if len(alternatives) == 1:
            return [(alternatives[0], 1.0)]

        attr_names = [c[0] for c in self.CRITERIA]
        weight_keys = [c[1] for c in self.CRITERIA]
        is_benefit = [c[2] for c in self.CRITERIA]

        # Step 1: Build decision matrix
        matrix = np.array([
            [getattr(a, attr) for attr in attr_names]
            for a in alternatives
        ], dtype=float)

        # Step 2: Vector normalization
        norms = np.sqrt((matrix ** 2).sum(axis=0))
        norms[norms == 0] = 1.0  # avoid division by zero
        normalized = matrix / norms

        # Step 3: Apply weights
        weights = np.array([self.weights.get(k, 0.0) for k in weight_keys])
        weighted = normalized * weights

        # Step 4: Ideal and anti-ideal solutions
        ideal = np.zeros(len(self.CRITERIA))
        anti_ideal = np.zeros(len(self.CRITERIA))
        for i, is_ben in enumerate(is_benefit):
            if is_ben:
                ideal[i] = weighted[:, i].max()
                anti_ideal[i] = weighted[:, i].min()
            else:
                ideal[i] = weighted[:, i].min()
                anti_ideal[i] = weighted[:, i].max()

        # Step 5: Euclidean distances
        dist_to_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
        dist_to_anti = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))

        # Step 6: Closeness coefficient
        denominator = dist_to_ideal + dist_to_anti
        denominator[denominator == 0] = 1e-10
        scores = dist_to_anti / denominator

        # Sort descending by score
        ranked = sorted(zip(alternatives, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked

    def optimize(self, current_replicas: int, min_replicas: int, max_replicas: int,
                 metrics: Dict[str, Any], forecast: Dict[str, Any],
                 current_cpu_m: Optional[int] = None,
                 current_memory_mi: Optional[int] = None,
                 enable_vpa: bool = True,
                 extra_alternatives: Optional[List[ScalingAlternative]] = None,
                 ) -> MCDAResult:
        """
        Main entry point: replaces threshold heuristics with MCDA optimization.

        Args:
            current_replicas: Current number of replicas
            min_replicas: Minimum allowed replicas
            max_replicas: Maximum allowed replicas
            metrics: Dict with 'cpu_percent' and 'memory_percent'
            forecast: Dict with 'predicted_cpu', 'predicted_memory', 'cpu_trend'
            current_cpu_m: Per-pod CPU request in millicores. Optional;
                required to generate any vertical alternatives.
            current_memory_mi: Per-pod memory request in MiB. Optional;
                required to generate any vertical alternatives.
            enable_vpa: When True (default) and both resource fields are
                supplied, the vertical alternative pool is included.
            extra_alternatives: Optional extra alternatives to inject into
                the TOPSIS pool. Used by validate_llm_decision() to include
                the LLM's exact recommendation when it does not fall on the
                multiplier grid.

        Returns:
            MCDAResult with optimal action, score, ranking, and reasoning.
        """
        horizontal = self.generate_alternatives(
            current_replicas, min_replicas, max_replicas, metrics, forecast,
            current_cpu_m=current_cpu_m, current_memory_mi=current_memory_mi,
        )
        vertical: List[ScalingAlternative] = []
        if enable_vpa and current_cpu_m is not None and current_memory_mi is not None:
            vertical = self.generate_vpa_alternatives(
                current_replicas=current_replicas,
                min_replicas=min_replicas, max_replicas=max_replicas,
                current_cpu_m=current_cpu_m,
                current_memory_mi=current_memory_mi,
                metrics=metrics, forecast=forecast,
            )
        alternatives = horizontal + vertical
        if extra_alternatives:
            alternatives = alternatives + list(extra_alternatives)

        ranked = self.topsis_rank(alternatives)

        if not ranked:
            return MCDAResult(
                best_alternative='maintain',
                target_replicas=current_replicas,
                action='maintain',
                mcda_score=0.0,
                dominance_margin=0.0,
                alternatives_evaluated=0,
                ranking=[],
                criteria_weights=self.weights,
                reasoning='No alternatives could be generated'
            )

        best, best_score = ranked[0]
        runner_up, runner_up_score = ranked[1] if len(ranked) > 1 else (None, 0.0)

        # Determine action -- horizontal compares replicas; vertical labels
        # by whichever per-pod dimension moved the most. A reshape (e.g.
        # CPU down + memory up) counts as scale_up because at least one
        # dimension is gaining capacity.
        if best.scaling_type == 'vpa' and current_cpu_m and current_memory_mi:
            cpu_delta = (best.target_cpu_m - current_cpu_m) / max(current_cpu_m, 1)
            mem_delta = (best.target_memory_mi - current_memory_mi) / max(current_memory_mi, 1)
            if max(cpu_delta, mem_delta) > 0.05:
                action = 'scale_up'
            elif max(cpu_delta, mem_delta) < -0.05:
                action = 'scale_down'
            else:
                action = 'maintain'
        else:
            if best.target_replicas > current_replicas:
                action = 'scale_up'
            elif best.target_replicas < current_replicas:
                action = 'scale_down'
            else:
                action = 'maintain'

        dominance_margin = best_score - runner_up_score if runner_up else 1.0

        # Build per-alternative criteria scores for transparency
        criteria_scores = {}
        for alt, score in ranked:
            criteria_scores[alt.name] = {
                'score': round(score, 4),
                'scaling_type': alt.scaling_type,
                'cost': round(alt.estimated_cost, 3),
                'performance': round(alt.estimated_performance, 3),
                'stability_risk': round(alt.stability_risk, 3),
                'forecast_alignment': round(alt.forecast_alignment, 3),
                'response_time': round(alt.response_time, 3),
            }

        # Build reasoning string
        n_horiz = len(horizontal)
        n_vert = len(vertical)
        reasoning_parts = [
            f"TOPSIS multi-criteria optimization evaluated {len(alternatives)} alternatives "
            f"({n_horiz} horizontal, {n_vert} vertical).",
            f"Best: {best.name} (score={best_score:.4f}, type={best.scaling_type})",
        ]
        if runner_up:
            reasoning_parts.append(
                f"Runner-up: {runner_up.name} (score={runner_up_score:.4f}, type={runner_up.scaling_type})"
            )
        reasoning_parts.append(f"Dominance margin: {dominance_margin:.4f}")
        reasoning_parts.append(f"Weights: cost={self.weights['cost']:.2f}, perf={self.weights['performance']:.2f}, "
                               f"stability={self.weights['stability']:.2f}, forecast={self.weights['forecast_alignment']:.2f}, "
                               f"latency={self.weights['response_time']:.2f}")

        cpu = metrics.get('cpu_percent', 0)
        mem = metrics.get('memory_percent', 0)
        reasoning_parts.append(f"Input metrics: CPU={cpu:.1f}%, Memory={mem:.1f}%, "
                               f"current_replicas={current_replicas}")

        return MCDAResult(
            best_alternative=best.name,
            target_replicas=best.target_replicas,
            target_cpu_m=best.target_cpu_m,
            target_memory_mi=best.target_memory_mi,
            scaling_type=best.scaling_type,
            action=action,
            mcda_score=round(best_score, 4),
            dominance_margin=round(dominance_margin, 4),
            alternatives_evaluated=len(alternatives),
            ranking=[(a.name, round(s, 4)) for a, s in ranked],
            criteria_weights=self.weights,
            reasoning=' | '.join(reasoning_parts),
            criteria_scores=criteria_scores
        )

    def validate_llm_decision(self, llm_action: str, llm_target: int,
                               current_replicas: int, min_replicas: int, max_replicas: int,
                               metrics: Dict[str, Any], forecast: Dict[str, Any],
                               agreement_threshold: float = 0.15,
                               llm_scaling_type: str = 'hpa',
                               llm_target_cpu_m: Optional[int] = None,
                               llm_target_memory_mi: Optional[int] = None,
                               current_cpu_m: Optional[int] = None,
                               current_memory_mi: Optional[int] = None,
                               ) -> Dict[str, Any]:
        """
        Validate an LLM scaling decision against MCDA optimization.

        Compares the LLM's recommended scaling action with the MCDA-optimal
        choice across the unified horizontal-and-vertical alternative pool.
        Phase-I addition: when ``llm_scaling_type='vpa'`` and the LLM's
        exact (cpu, memory) target is not on the multiplier grid, it is
        injected into the TOPSIS pool so the LLM gets a real score on the
        same scale as MCDA's own picks.

        Args:
            llm_action: LLM's recommended action ('scale_up', 'scale_down',
                'maintain', 'at_max').
            llm_target: For HPA, the LLM's recommended target replica count.
                For VPA, this is the replicas value to hold (typically the
                current count).
            current_replicas / min_replicas / max_replicas: Replica bounds.
            metrics: Current metrics dict.
            forecast: Forecast dict.
            agreement_threshold: Max score difference to keep the LLM decision.
            llm_scaling_type: 'hpa' (default; preserves Phase H behaviour)
                or 'vpa' (Phase I).
            llm_target_cpu_m: For VPA, the LLM's recommended per-pod CPU
                request in millicores.
            llm_target_memory_mi: For VPA, the LLM's recommended per-pod
                memory request in MiB.
            current_cpu_m / current_memory_mi: Current per-pod resource
                requests. Required when ``llm_scaling_type='vpa'`` and also
                when the caller wants MCDA to consider vertical alternatives
                even if the LLM picked HPA.

        Returns:
            Dict with validation result, MCDA recommendation, and whether to override.
        """
        # If LLM picked VPA and its exact target is not on our multiplier
        # grid, inject it explicitly so it appears in the ranking.
        extra_alternatives: List[ScalingAlternative] = []
        if (llm_scaling_type == 'vpa'
                and llm_target_cpu_m is not None
                and llm_target_memory_mi is not None
                and current_cpu_m is not None
                and current_memory_mi is not None):
            cpu_pf = metrics.get('cpu_percent', 50.0)
            mem_pf = metrics.get('memory_percent', 50.0)
            cpu_forecast = forecast.get('predicted_cpu', [cpu_pf] * 6) or [cpu_pf] * 6
            mem_forecast = forecast.get('predicted_memory', [mem_pf] * 6) or [mem_pf] * 6
            peak_cpu = max(cpu_forecast)
            peak_mem = max(mem_forecast)
            llm_pick = self._evaluate_alternative(
                target_replicas=current_replicas,
                target_cpu_m=int(llm_target_cpu_m),
                target_memory_mi=int(llm_target_memory_mi),
                current_replicas=current_replicas,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                current_cpu_m=current_cpu_m,
                current_memory_mi=current_memory_mi,
                cpu=cpu_pf, mem=mem_pf,
                peak_cpu=peak_cpu, peak_mem=peak_mem,
                cpu_forecast=cpu_forecast, mem_forecast=mem_forecast,
                cpu_trend=forecast.get('cpu_trend', 'stable'),
                scaling_type='vpa',
            )
            # Tag the name so it is unambiguously the LLM-injected entry
            # even if it coincides numerically with a grid point.
            llm_pick.name = (
                f"llm_vpa_cpu{int(llm_target_cpu_m)}m_"
                f"mem{int(llm_target_memory_mi)}Mi"
            )
            extra_alternatives.append(llm_pick)

        mcda_result = self.optimize(
            current_replicas, min_replicas, max_replicas, metrics, forecast,
            current_cpu_m=current_cpu_m, current_memory_mi=current_memory_mi,
            extra_alternatives=extra_alternatives,
        )

        # Build the canonical name of the LLM's pick to look it up in the
        # ranking. For VPA we used the "llm_vpa_..." synthetic name above.
        if llm_scaling_type == 'vpa' and extra_alternatives:
            llm_alt_name = extra_alternatives[0].name
        else:
            llm_alt_name = f"scale_to_{llm_target}"

        llm_score = 0.0
        for name, score in mcda_result.ranking:
            if name == llm_alt_name:
                llm_score = score
                break

        # Score gap on the same 0..1 TOPSIS scale
        score_difference = mcda_result.mcda_score - llm_score

        # Agreement label -- direction first, then magnitude. For VPA the
        # "magnitude" is whether the per-pod resource targets are close.
        direction_agrees = (
            (llm_action == mcda_result.action) or
            (llm_action in ('scale_up', 'at_max') and mcda_result.action == 'scale_up') or
            (
                llm_scaling_type == 'hpa'
                and llm_target == mcda_result.target_replicas
            )
        )

        if llm_scaling_type == 'vpa':
            close_in_magnitude = (
                mcda_result.scaling_type == 'vpa'
                and llm_target_cpu_m is not None
                and llm_target_memory_mi is not None
                and mcda_result.target_cpu_m > 0
                and mcda_result.target_memory_mi > 0
                and abs(mcda_result.target_cpu_m - llm_target_cpu_m) / max(llm_target_cpu_m, 1) <= 0.15
                and abs(mcda_result.target_memory_mi - llm_target_memory_mi) / max(llm_target_memory_mi, 1) <= 0.15
            )
        else:
            close_in_magnitude = abs(llm_target - mcda_result.target_replicas) <= 1

        if direction_agrees and close_in_magnitude:
            agreement = 'full'
            should_override = False
            validation_note = (
                f"LLM and MCDA agree: {llm_action} ({llm_scaling_type}). "
                f"MCDA score for LLM choice: {llm_score:.4f}, "
                f"MCDA optimal: {mcda_result.mcda_score:.4f}"
            )
        elif direction_agrees:
            agreement = 'partial'
            should_override = score_difference > agreement_threshold
            validation_note = (
                f"LLM direction agrees ({llm_action}) but magnitude differs. "
                f"LLM={llm_scaling_type}:{llm_alt_name}, "
                f"MCDA={mcda_result.scaling_type}:{mcda_result.best_alternative}. "
                f"Score gap: {score_difference:.4f}"
            )
        else:
            agreement = 'disagree'
            should_override = score_difference > agreement_threshold
            validation_note = (
                f"LLM ({llm_scaling_type}:{llm_action}→{llm_alt_name}) disagrees with "
                f"MCDA ({mcda_result.scaling_type}:{mcda_result.action}→{mcda_result.best_alternative}). "
                f"Score gap: {score_difference:.4f}. "
                f"{'MCDA overrides LLM.' if should_override else 'Keeping LLM decision (within threshold).'}"
            )

        return {
            'agreement': agreement,
            'should_override': should_override,
            'llm_action': llm_action,
            'llm_target': llm_target,
            'llm_scaling_type': llm_scaling_type,
            'llm_target_cpu_m': llm_target_cpu_m,
            'llm_target_memory_mi': llm_target_memory_mi,
            'llm_score': round(llm_score, 4),
            'mcda_action': mcda_result.action,
            'mcda_target': mcda_result.target_replicas,
            'mcda_target_cpu_m': mcda_result.target_cpu_m,
            'mcda_target_memory_mi': mcda_result.target_memory_mi,
            'mcda_scaling_type': mcda_result.scaling_type,
            'mcda_score': mcda_result.mcda_score,
            'score_difference': round(score_difference, 4),
            'dominance_margin': mcda_result.dominance_margin,
            'validation_note': validation_note,
            'mcda_ranking': mcda_result.ranking,
            'mcda_criteria_scores': mcda_result.criteria_scores,
            'criteria_weights': mcda_result.criteria_weights
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize optimizer state for API responses"""
        return {
            'weights': self.weights,
            'criteria': [
                {'attribute': c[0], 'weight_key': c[1], 'is_benefit': c[2]}
                for c in self.CRITERIA
            ],
            'available_profiles': list(self.WEIGHT_PROFILES.keys())
        }
