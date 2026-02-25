#!/usr/bin/env python3
"""
Common decision contract for autoscaling recommendations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ScalingDecision:
    action: str
    scaling_type: str = "hpa"
    target_replicas: Optional[int] = None
    target_cpu: Optional[str] = None
    target_memory: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "scaling_type": self.scaling_type,
            "target_replicas": self.target_replicas,
            "target_cpu": self.target_cpu,
            "target_memory": self.target_memory,
            "confidence": self.confidence,
            "reason": self.reason,
            "source": self.source,
            "metadata": self.metadata,
        }
