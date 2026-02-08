"""
Analysis tools for understanding induction head behavior.

Provides detailed analysis of attention patterns and
head-specific behavior.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class AttentionPattern:
    """Captured attention pattern from a head."""
    layer: int
    head: int
    attention: torch.Tensor  # [batch, seq, seq]
    tokens: Optional[torch.Tensor] = None

    @property
    def seq_length(self) -> int:
        return self.attention.shape[-1]

    def get_diagonal_score(self, offset: int = 1) -> float:
        """Get average attention on a diagonal (offset from main diagonal)."""
        seq_len = self.seq_length
        total = 0.0
        count = 0

        for i in range(offset, seq_len):
            total += self.attention[0, i, i - offset].item()
            count += 1

        return total / count if count > 0 else 0.0

    def get_induction_diagonal_score(self, period: int) -> float:
        """Get attention on the induction diagonal (offset = period)."""
        return self.get_diagonal_score(offset=period)


@dataclass
class HeadBehavior:
    """Analyzed behavior of a single head."""
    layer: int
    head: int
    is_induction: bool
    is_previous_token: bool
    is_duplicate_token: bool
    primary_behavior: str
    confidence: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "head": self.head,
            "is_induction": self.is_induction,
            "primary_behavior": self.primary_behavior,
            "confidence": round(self.confidence, 4),
            "metrics": self.metrics,
        }


class InductionHeadAnalyzer:
    """
    Detailed analysis of attention head behavior.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._device = next(model.parameters()).device

    def _get_blocks(self) -> nn.ModuleList:
        """Get transformer blocks."""
        if hasattr(self.model, 'blocks'):
            return self.model.blocks
        elif hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'h'):
                return self.model.transformer.h
            return self.model.transformer.blocks
        elif hasattr(self.model, 'layers'):
            return self.model.layers
        raise ValueError("Could not find transformer blocks")

    def analyze_head(
        self,
        layer: int,
        head: int,
        tokens: torch.Tensor,
        attention: Optional[torch.Tensor] = None,
    ) -> HeadBehavior:
        """
        Analyze the behavior of a specific head.

        Args:
            layer: Layer index.
            head: Head index.
            tokens: Input tokens.
            attention: Pre-computed attention (optional).

        Returns:
            HeadBehavior with classification and metrics.
        """
        if attention is None:
            attention = self._get_attention(layer, head, tokens)

        if attention is None:
            return HeadBehavior(
                layer=layer,
                head=head,
                is_induction=False,
                is_previous_token=False,
                is_duplicate_token=False,
                primary_behavior="unknown",
                confidence=0.0,
            )

        metrics = {}

        # Compute various attention metrics
        metrics["prev_token_attn"] = self._compute_prev_token_attention(attention)
        metrics["induction_attn"] = self._compute_induction_attention(attention, tokens)
        metrics["duplicate_attn"] = self._compute_duplicate_attention(attention, tokens)
        metrics["uniform_score"] = self._compute_uniformity(attention)

        # Classify behavior
        is_induction = metrics["induction_attn"] > 0.3
        is_previous = metrics["prev_token_attn"] > 0.5
        is_duplicate = metrics["duplicate_attn"] > 0.3

        # Determine primary behavior
        behaviors = [
            ("induction", metrics["induction_attn"]),
            ("previous_token", metrics["prev_token_attn"]),
            ("duplicate_token", metrics["duplicate_attn"]),
            ("uniform", metrics["uniform_score"]),
        ]
        primary, confidence = max(behaviors, key=lambda x: x[1])

        return HeadBehavior(
            layer=layer,
            head=head,
            is_induction=is_induction,
            is_previous_token=is_previous,
            is_duplicate_token=is_duplicate,
            primary_behavior=primary,
            confidence=confidence,
            metrics=metrics,
        )

    def _get_attention(
        self,
        layer: int,
        head: int,
        tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Get attention pattern for a specific head."""
        # This is a placeholder - actual implementation depends on model architecture
        return None

    def _compute_prev_token_attention(
        self,
        attention: torch.Tensor,
    ) -> float:
        """Compute average attention to previous token."""
        seq_len = attention.shape[-1]
        total = 0.0

        for i in range(1, seq_len):
            total += attention[0, i, i - 1].item()

        return total / (seq_len - 1)

    def _compute_induction_attention(
        self,
        attention: torch.Tensor,
        tokens: torch.Tensor,
    ) -> float:
        """Compute induction-specific attention pattern."""
        seq_len = attention.shape[-1]
        total = 0.0
        count = 0

        for i in range(2, seq_len):
            for j in range(0, i - 1):
                # Check if tokens match
                if tokens[0, i - 1] == tokens[0, j]:
                    # Measure attention to j + 1
                    if j + 1 < seq_len:
                        total += attention[0, i, j + 1].item()
                        count += 1

        return total / count if count > 0 else 0.0

    def _compute_duplicate_attention(
        self,
        attention: torch.Tensor,
        tokens: torch.Tensor,
    ) -> float:
        """Compute attention to duplicate tokens."""
        seq_len = attention.shape[-1]
        total = 0.0
        count = 0

        for i in range(1, seq_len):
            for j in range(0, i):
                if tokens[0, i] == tokens[0, j]:
                    total += attention[0, i, j].item()
                    count += 1

        return total / count if count > 0 else 0.0

    def _compute_uniformity(
        self,
        attention: torch.Tensor,
    ) -> float:
        """Compute how uniform the attention is (entropy-based)."""
        seq_len = attention.shape[-1]

        # Expected uniform attention
        uniform_attn = 1.0 / seq_len

        # Compute average deviation from uniform
        deviations = []
        for i in range(1, seq_len):
            attn_row = attention[0, i, :i + 1]
            expected = torch.ones_like(attn_row) / (i + 1)
            deviation = (attn_row - expected).abs().mean().item()
            deviations.append(deviation)

        return 1.0 - sum(deviations) / len(deviations) if deviations else 0.0

    def compare_heads(
        self,
        tokens: torch.Tensor,
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> List[HeadBehavior]:
        """
        Compare behavior across all heads.

        Returns list of HeadBehavior sorted by induction score.
        """
        blocks = self._get_blocks()
        n_layers = len(blocks)

        if layer_range is None:
            layer_range = (0, n_layers)

        start, end = layer_range
        end = min(end, n_layers)

        behaviors = []

        # This would need proper attention extraction for each model type
        # Placeholder implementation
        for layer in range(start, end):
            # Assume we can get attention somehow
            behavior = HeadBehavior(
                layer=layer,
                head=0,
                is_induction=False,
                is_previous_token=False,
                is_duplicate_token=False,
                primary_behavior="unknown",
                confidence=0.0,
            )
            behaviors.append(behavior)

        return sorted(behaviors, key=lambda b: b.confidence, reverse=True)

    def find_induction_heads(
        self,
        tokens: torch.Tensor,
        threshold: float = 0.3,
    ) -> List[Tuple[int, int]]:
        """
        Find all heads that exhibit induction behavior.

        Returns list of (layer, head) tuples.
        """
        behaviors = self.compare_heads(tokens)
        return [
            (b.layer, b.head)
            for b in behaviors
            if b.is_induction and b.confidence >= threshold
        ]


def analyze_head_attention(
    attention: torch.Tensor,
    tokens: torch.Tensor,
) -> Dict[str, Any]:
    """
    Standalone function to analyze an attention pattern.

    Args:
        attention: Attention weights [batch, seq, seq] or [batch, heads, seq, seq]
        tokens: Input tokens [batch, seq]

    Returns:
        Dictionary with analysis metrics.
    """
    if attention.dim() == 4:
        # Average across heads for overall analysis
        attention = attention.mean(dim=1)

    seq_len = attention.shape[-1]
    batch_size = attention.shape[0]

    metrics = {}

    # Previous token attention
    prev_attn = 0.0
    for i in range(1, seq_len):
        prev_attn += attention[:, i, i - 1].mean().item()
    metrics["previous_token_attention"] = prev_attn / (seq_len - 1)

    # Diagonal attention patterns
    for offset in [1, 2, 4, 8]:
        if offset < seq_len:
            diag_attn = 0.0
            count = 0
            for i in range(offset, seq_len):
                diag_attn += attention[:, i, i - offset].mean().item()
                count += 1
            metrics[f"diagonal_{offset}_attention"] = diag_attn / count

    # Entropy (attention spread)
    entropy = 0.0
    for i in range(1, seq_len):
        attn_dist = attention[:, i, :i + 1]
        attn_dist = attn_dist + 1e-10
        ent = -(attn_dist * attn_dist.log()).sum(dim=-1).mean().item()
        entropy += ent
    metrics["average_entropy"] = entropy / (seq_len - 1)

    # Peak sharpness (how focused attention is)
    max_attn = 0.0
    for i in range(1, seq_len):
        max_attn += attention[:, i, :i + 1].max(dim=-1)[0].mean().item()
    metrics["average_peak_attention"] = max_attn / (seq_len - 1)

    return metrics
