"""
Detection of induction heads in transformer models.

An induction head is an attention head that implements pattern completion:
When it sees [A][B]...[A], it predicts [B] by attending back to the
previous occurrence of [A] and copying what came after it.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable


@dataclass
class HeadScore:
    """Score for a single attention head."""
    layer: int
    head: int
    induction_score: float
    prefix_matching_score: float
    copying_score: float
    overall_score: float

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "head": self.head,
            "induction_score": round(self.induction_score, 4),
            "prefix_matching_score": round(self.prefix_matching_score, 4),
            "copying_score": round(self.copying_score, 4),
            "overall_score": round(self.overall_score, 4),
        }


@dataclass
class DetectionResult:
    """Results from induction head detection."""
    head_scores: List[HeadScore]
    induction_heads: List[Tuple[int, int]]  # (layer, head) pairs
    threshold: float
    model_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_heads_detected(self) -> int:
        return len(self.induction_heads)

    def get_top_heads(self, n: int = 5) -> List[HeadScore]:
        """Get top N heads by overall score."""
        sorted_heads = sorted(
            self.head_scores,
            key=lambda h: h.overall_score,
            reverse=True
        )
        return sorted_heads[:n]

    def to_dict(self) -> dict:
        return {
            "num_heads_detected": self.num_heads_detected,
            "threshold": self.threshold,
            "induction_heads": self.induction_heads,
            "model_info": self.model_info,
            "top_heads": [h.to_dict() for h in self.get_top_heads()],
        }


class InductionHeadDetector:
    """
    Detects induction heads in transformer models.

    Uses three metrics:
    1. Induction score: Does the head attend to token after previous occurrence?
    2. Prefix matching: Does it attend more when prefixes match?
    3. Copying: Does attention copying predict the next token?
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.4,
        device: Optional[str] = None,
    ):
        self.model = model
        self.threshold = threshold
        self.device = device or next(model.parameters()).device
        self._attention_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _get_attention_hook(
        self,
        layer_idx: int,
        cache: Dict,
    ) -> Callable:
        """Create a hook to capture attention patterns."""
        def hook(module, inputs, outputs):
            # Handle different model architectures
            if isinstance(outputs, tuple):
                if len(outputs) >= 2 and outputs[1] is not None:
                    # (hidden_states, attention_weights)
                    cache[layer_idx] = outputs[1].detach()
                else:
                    # Try to extract from attention module directly
                    if hasattr(module, 'attention_weights'):
                        cache[layer_idx] = module.attention_weights.detach()
            return outputs
        return hook

    def _get_model_blocks(self) -> nn.ModuleList:
        """Get the transformer blocks from the model."""
        if hasattr(self.model, 'blocks'):
            return self.model.blocks
        elif hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'h'):
                return self.model.transformer.h
            elif hasattr(self.model.transformer, 'blocks'):
                return self.model.transformer.blocks
        elif hasattr(self.model, 'layers'):
            return self.model.layers
        raise ValueError("Could not find transformer blocks")

    def _get_attention_module(self, block: nn.Module) -> nn.Module:
        """Get the attention module from a block."""
        if hasattr(block, 'attn'):
            return block.attn
        elif hasattr(block, 'attention'):
            return block.attention
        elif hasattr(block, 'self_attn'):
            return block.self_attn
        return block

    def compute_induction_score(
        self,
        attention: torch.Tensor,
        tokens: torch.Tensor,
    ) -> float:
        """
        Compute induction score for an attention pattern.

        For each position i where tokens[i] matches a previous token at j,
        measure how much attention position i+1 places on position j+1.
        """
        batch, n_heads, seq_len, _ = attention.shape

        if seq_len < 4:
            return 0.0

        total_score = 0.0
        count = 0

        # For each position, find matching previous tokens
        for i in range(2, seq_len):
            for j in range(0, i - 1):
                # Check if tokens match: tokens[i-1] == tokens[j]
                if torch.all(tokens[:, i - 1] == tokens[:, j]):
                    # Measure attention from position i to position j+1
                    # This is the "induction" pattern: attending to what came after
                    attn_to_next = attention[:, :, i, j + 1].mean()
                    total_score += attn_to_next.item()
                    count += 1

        if count == 0:
            return 0.0

        return total_score / count

    def compute_prefix_matching_score(
        self,
        attention: torch.Tensor,
        tokens: torch.Tensor,
        prefix_len: int = 2,
    ) -> float:
        """
        Compute how much the head attends based on prefix matching.

        A strong induction head should attend more to positions
        where the prefix matches.
        """
        batch, n_heads, seq_len, _ = attention.shape

        if seq_len < prefix_len + 2:
            return 0.0

        matching_attention = 0.0
        non_matching_attention = 0.0
        match_count = 0
        non_match_count = 0

        for i in range(prefix_len, seq_len):
            current_prefix = tokens[:, i - prefix_len:i]

            for j in range(prefix_len, i):
                prev_prefix = tokens[:, j - prefix_len:j]

                if torch.all(current_prefix == prev_prefix):
                    matching_attention += attention[:, :, i, j].mean().item()
                    match_count += 1
                else:
                    non_matching_attention += attention[:, :, i, j].mean().item()
                    non_match_count += 1

        if match_count == 0 or non_match_count == 0:
            return 0.0

        avg_match = matching_attention / match_count
        avg_non_match = non_matching_attention / non_match_count

        # Score is how much more attention goes to matches
        return max(0, avg_match - avg_non_match)

    def compute_copying_score(
        self,
        attention: torch.Tensor,
        tokens: torch.Tensor,
    ) -> float:
        """
        Compute copying score: does attending lead to correct prediction?

        Measures correlation between attention to position j
        and prediction of tokens[j+1].
        """
        batch, n_heads, seq_len, _ = attention.shape

        if seq_len < 3:
            return 0.0

        correct_copies = 0.0
        total = 0

        for i in range(1, seq_len - 1):
            # Find position with max attention
            max_attn_pos = attention[:, :, i, :i].argmax(dim=-1)  # [batch, n_heads]

            # Check if token at max_attn_pos + 1 matches tokens[i + 1]
            for b in range(batch):
                for h in range(n_heads):
                    j = max_attn_pos[b, h].item()
                    if j + 1 < seq_len:
                        if tokens[b, int(j) + 1] == tokens[b, i + 1]:
                            correct_copies += 1
                        total += 1

        if total == 0:
            return 0.0

        return correct_copies / total

    def detect(
        self,
        input_ids: Optional[torch.Tensor] = None,
        sequence_length: int = 64,
        vocab_size: int = 100,
        n_repeats: int = 2,
    ) -> DetectionResult:
        """
        Detect induction heads in the model.

        Args:
            input_ids: Optional input tokens. If None, generates repeated sequences.
            sequence_length: Length of generated sequences.
            vocab_size: Vocab size for random tokens.
            n_repeats: Number of sequence repeats for detection.

        Returns:
            DetectionResult with scores for each head.
        """
        self.model.eval()
        blocks = self._get_model_blocks()
        n_layers = len(blocks)

        # Generate or use input
        if input_ids is None:
            # Generate repeated random sequence for induction detection
            half_len = sequence_length // (2 * n_repeats)
            base_seq = torch.randint(1, vocab_size, (1, half_len), device=self.device)
            input_ids = base_seq.repeat(1, 2 * n_repeats)

        input_ids = input_ids.to(self.device)

        # Collect attention patterns
        attention_cache = {}
        hooks = []

        for layer_idx, block in enumerate(blocks):
            attn_module = self._get_attention_module(block)
            hook = attn_module.register_forward_hook(
                self._get_attention_hook(layer_idx, attention_cache)
            )
            hooks.append(hook)

        try:
            with torch.no_grad():
                # Forward pass with attention output
                if hasattr(self.model, 'forward'):
                    try:
                        _ = self.model(input_ids, output_attentions=True)
                    except TypeError:
                        _ = self.model(input_ids)
                else:
                    _ = self.model(input_ids)
        finally:
            for hook in hooks:
                hook.remove()

        # If no attention was cached, try simpler detection
        if not attention_cache:
            return self._fallback_detection(input_ids, blocks)

        # Score each head
        head_scores = []
        induction_heads = []

        for layer_idx in range(n_layers):
            if layer_idx not in attention_cache:
                continue

            attention = attention_cache[layer_idx]

            # Handle different attention shapes
            if attention.dim() == 3:
                # [batch, seq, seq] - need to add head dim
                attention = attention.unsqueeze(1)

            n_heads = attention.shape[1]

            for head_idx in range(n_heads):
                head_attention = attention[:, head_idx:head_idx+1, :, :]

                induction = self.compute_induction_score(head_attention, input_ids)
                prefix = self.compute_prefix_matching_score(head_attention, input_ids)
                copying = self.compute_copying_score(head_attention, input_ids)

                overall = (induction + prefix + copying) / 3

                score = HeadScore(
                    layer=layer_idx,
                    head=head_idx,
                    induction_score=induction,
                    prefix_matching_score=prefix,
                    copying_score=copying,
                    overall_score=overall,
                )
                head_scores.append(score)

                if overall >= self.threshold:
                    induction_heads.append((layer_idx, head_idx))

        return DetectionResult(
            head_scores=head_scores,
            induction_heads=induction_heads,
            threshold=self.threshold,
            model_info={
                "n_layers": n_layers,
                "sequence_length": input_ids.shape[1],
            },
        )

    def _fallback_detection(
        self,
        input_ids: torch.Tensor,
        blocks: nn.ModuleList,
    ) -> DetectionResult:
        """Fallback detection when attention isn't easily extractable."""
        # Return empty result for models without accessible attention
        return DetectionResult(
            head_scores=[],
            induction_heads=[],
            threshold=self.threshold,
            model_info={
                "n_layers": len(blocks),
                "note": "Attention not accessible",
            },
        )


def detect_induction_heads(
    model: nn.Module,
    threshold: float = 0.4,
    sequence_length: int = 64,
    device: Optional[str] = None,
) -> DetectionResult:
    """
    Convenience function to detect induction heads.

    Args:
        model: Transformer model to analyze.
        threshold: Score threshold for classifying as induction head.
        sequence_length: Length of test sequences.
        device: Device to use.

    Returns:
        DetectionResult with detected induction heads.
    """
    detector = InductionHeadDetector(model, threshold=threshold, device=device)
    return detector.detect(sequence_length=sequence_length)
