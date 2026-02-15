from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from driftguard_ray.federate.server.cluster import Group


@dataclass
class Fp:
    """Fingerprint for a client.

    Attributes:
        label_gate_norm: Normalized per-label gate activations, shape (labels, layers, experts).
        w: Per-label weights used for distance computation, shape (labels,).
    """

    label_gate_norm: np.ndarray
    w: np.ndarray

    @staticmethod
    def build(
        out_softs: np.ndarray,
        gate_activations: np.ndarray,
        w_size: int, # 可靠性权重, 几个样本才可信
        eps: float = 1e-12,
    ) -> Fp:
        """Build Fp from out_softs and gate activations.

        Args:
            out_softs: Soft label distributions, shape (samples, labels).
            gate_activations: Gate activations, shape (samples, layers, experts).
            w_size: Smoothing factor for weight calculation.
            eps: Small constant to avoid division by zero.

        Returns:
            A constructed Fp instance.
        """
        out_softs.shape  # (sample size, label size)
        gate_activations.shape  # (sample, layer, experts)
        assert out_softs.shape[0] == gate_activations.shape[0]

        label_weighted_counts = out_softs.sum(0)  # label weighted counts
        w = label_weighted_counts / (label_weighted_counts + w_size)

        # 对expert 的类软计数加权计数
        label_gate_sum = np.einsum("sc,sle->cle", out_softs, gate_activations)

        label_gate_norm = label_gate_sum / (label_weighted_counts + eps)[:, None, None]
        return Fp(label_gate_norm, w)

    @staticmethod
    def dist(a: Fp, b: Fp) -> float:
        """Compute distance between two Fp instances.

        Args:
            a: First fingerprint.
            b: Second fingerprint.

        Returns:
            Squared weighted distance between fingerprints.
        """
        w = (a.w * b.w) ** 0.5
        diff = a.label_gate_norm - b.label_gate_norm
        weights = np.broadcast_to(w[:, None, None], diff.shape)
        dist = np.sqrt(np.average(diff**2, weights=weights))

        return float(dist)

    @staticmethod
    def pairwise_D(fps_list: List[Fp]) -> np.ndarray:
        """Compute pairwise distance matrix for a list of Fp instances.

        Args:
            fps_list: Sequence of fingerprints.

        Returns:
            Symmetric distance matrix with zeros on the diagonal.
        """
        n = len(fps_list)
        D = np.zeros((n, n), float)
        for i in range(n):
            for j in range(i + 1, n):
                d = Fp.dist(fps_list[i], fps_list[j])
                D[i, j] = D[j, i] = d
        return D
    
@dataclass
class Observation:
    accuracy: float
    reliance: float
    fingerprint: Fp


    @classmethod
    def get_fingerprints(
        cls, observations: List[Observation]
    ) -> List[Fp | None]:
        return [obs.fingerprint for obs in observations]
    
    @staticmethod
    def group_ave_acc(
        observations: List[Observation], groups: List[Group]
    ) -> List[tuple[Group, float]]:
        """Aggregate accs and relies by groups."""
        aggr_obs = [] # group_id -> Observation
        for g in groups:
            accs = [observations[i].accuracy for i in g.clients]
            avg_acc = sum(accs) / len(accs)
            aggr_obs.append(
                (
                    g,
                    avg_acc
                )
            )
        return aggr_obs # group_id -> Observation
    
    @staticmethod
    def ave_reliance(observations: List[Observation]) -> float:
        """Compute average reliance from observations."""
        relies = [obs.reliance for obs in observations]
        return sum(relies) / len(relies)
    
