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

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScalingAlternative:
    """One candidate scaling action to be evaluated by MCDA"""
    name: str
    target_replicas: int
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
            'cost': 0.20,
            'performance': 0.30,
            'stability': 0.25,
            'forecast_alignment': 0.15,
            'response_time': 0.10
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
                               forecast: Dict[str, Any]) -> List[ScalingAlternative]:
        """
        Generate candidate scaling actions to evaluate.

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

        Returns:
            List of ScalingAlternative objects to rank
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
                target, current_replicas, min_replicas, max_replicas,
                cpu, mem, peak_cpu, peak_mem,
                cpu_forecast, mem_forecast, cpu_trend
            )
            alternatives.append(alt)

        return alternatives

    def _evaluate_alternative(self, target: int, current_replicas: int,
                               min_replicas: int, max_replicas: int,
                               cpu: float, mem: float,
                               peak_cpu: float, peak_mem: float,
                               cpu_forecast: List[float],
                               mem_forecast: List[float],
                               cpu_trend: str) -> ScalingAlternative:
        """
        Evaluate a single candidate replica count across all criteria.

        Returns a ScalingAlternative with all criteria scores normalized to [0, 1].
        """
        safe_target = max(target, 1)
        safe_current = max(current_replicas, 1)

        # --- Cost: normalized by max_replicas (linear, more replicas = more cost) ---
        cost = target / max(max_replicas, 1)

        # --- Performance: how close estimated CPU will be to the 60-80% sweet spot ---
        # Estimate CPU at this replica count (proportional redistribution)
        estimated_cpu = cpu * (safe_current / safe_target)
        estimated_cpu = max(0, min(100, estimated_cpu))
        # Performance is 1.0 when CPU is exactly 70%, degrades toward 0% and 100%
        perf = 1.0 - abs(estimated_cpu - 70.0) / 70.0
        perf = max(0.0, min(1.0, perf))

        # --- Stability: penalize large changes from current ---
        change_magnitude = abs(target - current_replicas)
        # Normalize by the replica range to make it relative
        replica_range = max(max_replicas - min_replicas, 1)
        stability_risk = min(1.0, change_magnitude / replica_range)

        # Extra penalty for frequent oscillation (scaling down then up)
        if target < current_replicas and cpu_trend == 'increasing':
            stability_risk = min(1.0, stability_risk + 0.2)

        # --- Forecast alignment: how well this target handles predicted peak demand ---
        estimated_peak_cpu = peak_cpu * (safe_current / safe_target)
        estimated_peak_cpu = max(0, min(100, estimated_peak_cpu))
        # Alignment is best when estimated peak is in 50-75% range (comfortable headroom)
        if estimated_peak_cpu <= 75:
            alignment = 1.0 - max(0, (50 - estimated_peak_cpu)) / 50.0
        else:
            alignment = 1.0 - (estimated_peak_cpu - 75) / 25.0
        alignment = max(0.0, min(1.0, alignment))

        # Also factor in memory forecast
        estimated_peak_mem = peak_mem * (safe_current / safe_target)
        estimated_peak_mem = max(0, min(100, estimated_peak_mem))
        if estimated_peak_mem <= 80:
            mem_alignment = 1.0 - max(0, (40 - estimated_peak_mem)) / 40.0
        else:
            mem_alignment = 1.0 - (estimated_peak_mem - 80) / 20.0
        mem_alignment = max(0.0, min(1.0, mem_alignment))

        # Combined alignment (CPU weighted more heavily)
        alignment = 0.7 * alignment + 0.3 * mem_alignment

        # --- Response time: higher CPU → higher latency (rough model) ---
        avg_forecast = float(np.mean(cpu_forecast)) if cpu_forecast else cpu
        estimated_avg_cpu = avg_forecast * (safe_current / safe_target)
        estimated_avg_cpu = max(0, min(100, estimated_avg_cpu))
        response_time = estimated_avg_cpu / 100.0

        return ScalingAlternative(
            name=f"scale_to_{target}",
            target_replicas=target,
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
                 metrics: Dict[str, Any], forecast: Dict[str, Any]) -> MCDAResult:
        """
        Main entry point: replaces threshold heuristics with MCDA optimization.

        Args:
            current_replicas: Current number of replicas
            min_replicas: Minimum allowed replicas
            max_replicas: Maximum allowed replicas
            metrics: Dict with 'cpu_percent' and 'memory_percent'
            forecast: Dict with 'predicted_cpu', 'predicted_memory', 'cpu_trend'

        Returns:
            MCDAResult with optimal action, score, ranking, and reasoning
        """
        alternatives = self.generate_alternatives(
            current_replicas, min_replicas, max_replicas, metrics, forecast
        )

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

        # Determine action
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
                'cost': round(alt.estimated_cost, 3),
                'performance': round(alt.estimated_performance, 3),
                'stability_risk': round(alt.stability_risk, 3),
                'forecast_alignment': round(alt.forecast_alignment, 3),
                'response_time': round(alt.response_time, 3),
            }

        # Build reasoning string
        reasoning_parts = [
            f"TOPSIS multi-criteria optimization evaluated {len(alternatives)} alternatives.",
            f"Best: {best.name} (score={best_score:.4f})",
        ]
        if runner_up:
            reasoning_parts.append(f"Runner-up: {runner_up.name} (score={runner_up_score:.4f})")
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
                               agreement_threshold: float = 0.15) -> Dict[str, Any]:
        """
        Validate an LLM scaling decision against MCDA optimization.

        Compares the LLM's recommended target with the MCDA-optimal target.
        If they disagree significantly, flags the discrepancy with reasoning.

        Args:
            llm_action: LLM's recommended action ('scale_up', 'scale_down', 'maintain')
            llm_target: LLM's recommended target replicas
            current_replicas: Current number of replicas
            min_replicas: Minimum replicas
            max_replicas: Maximum replicas
            metrics: Current metrics
            forecast: Forecast data
            agreement_threshold: Max score difference to consider "in agreement"

        Returns:
            Dict with validation result, MCDA recommendation, and whether to override
        """
        mcda_result = self.optimize(
            current_replicas, min_replicas, max_replicas, metrics, forecast
        )

        # Find the score of the LLM's chosen alternative
        llm_alt_name = f"scale_to_{llm_target}"
        llm_score = 0.0
        for name, score in mcda_result.ranking:
            if name == llm_alt_name:
                llm_score = score
                break

        # Check agreement
        score_difference = mcda_result.mcda_score - llm_score
        direction_agrees = (
            (llm_action == mcda_result.action) or
            (llm_action in ('scale_up', 'at_max') and mcda_result.action == 'scale_up') or
            (llm_target == mcda_result.target_replicas)
        )

        if direction_agrees and abs(llm_target - mcda_result.target_replicas) <= 1:
            agreement = 'full'
            should_override = False
            validation_note = (f"LLM and MCDA agree: {llm_action} to {llm_target} replicas. "
                               f"MCDA score for LLM choice: {llm_score:.4f}, "
                               f"MCDA optimal: {mcda_result.mcda_score:.4f}")
        elif direction_agrees:
            agreement = 'partial'
            should_override = score_difference > agreement_threshold
            validation_note = (f"LLM direction agrees ({llm_action}) but magnitude differs: "
                               f"LLM={llm_target}, MCDA={mcda_result.target_replicas}. "
                               f"Score gap: {score_difference:.4f}")
        else:
            agreement = 'disagree'
            should_override = score_difference > agreement_threshold
            validation_note = (f"LLM ({llm_action}→{llm_target}) disagrees with "
                               f"MCDA ({mcda_result.action}→{mcda_result.target_replicas}). "
                               f"Score gap: {score_difference:.4f}. "
                               f"{'MCDA overrides LLM.' if should_override else 'Keeping LLM decision (within threshold).'}")

        return {
            'agreement': agreement,
            'should_override': should_override,
            'llm_action': llm_action,
            'llm_target': llm_target,
            'llm_score': round(llm_score, 4),
            'mcda_action': mcda_result.action,
            'mcda_target': mcda_result.target_replicas,
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
