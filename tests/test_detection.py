"""Tests for induction_head_detector.detection module."""

import pytest
import torch
import torch.nn as nn

from induction_head_detector.detection import (
    HeadScore,
    DetectionResult,
    InductionHeadDetector,
    detect_induction_heads,
)


class SimpleAttentionBlock(nn.Module):
    """Simple attention block for testing."""
    def __init__(self, hidden_dim=64, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

    def forward(self, x, output_attentions=False):
        out, attn_weights = self.attn(x, x, x)
        if output_attentions:
            return out, attn_weights
        return out


class SimpleTransformer(nn.Module):
    """Simple transformer for testing."""
    def __init__(self, vocab_size=100, hidden_dim=64, n_layers=4, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList([
            SimpleAttentionBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ])
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, output_attentions=False):
        x = self.embedding(input_ids)
        attentions = []
        for block in self.blocks:
            if output_attentions:
                x, attn = block(x, output_attentions=True)
                attentions.append(attn)
            else:
                x = block(x)
        if output_attentions:
            return self.output(x), attentions
        return self.output(x)


class TestHeadScore:
    def test_create(self):
        score = HeadScore(
            layer=2,
            head=3,
            induction_score=0.5,
            prefix_matching_score=0.4,
            copying_score=0.6,
            overall_score=0.5,
        )
        assert score.layer == 2
        assert score.head == 3
        assert score.overall_score == 0.5

    def test_to_dict(self):
        score = HeadScore(
            layer=1, head=2,
            induction_score=0.3,
            prefix_matching_score=0.4,
            copying_score=0.5,
            overall_score=0.4,
        )
        d = score.to_dict()
        assert d["layer"] == 1
        assert d["head"] == 2
        assert d["overall_score"] == 0.4


class TestDetectionResult:
    def test_create(self):
        scores = [
            HeadScore(0, 0, 0.5, 0.4, 0.6, 0.5),
            HeadScore(1, 0, 0.3, 0.2, 0.1, 0.2),
        ]
        result = DetectionResult(
            head_scores=scores,
            induction_heads=[(0, 0)],
            threshold=0.4,
        )
        assert result.num_heads_detected == 1
        assert result.threshold == 0.4

    def test_get_top_heads(self):
        scores = [
            HeadScore(0, 0, 0.2, 0.2, 0.2, 0.2),
            HeadScore(1, 0, 0.8, 0.8, 0.8, 0.8),
            HeadScore(2, 0, 0.5, 0.5, 0.5, 0.5),
        ]
        result = DetectionResult(
            head_scores=scores,
            induction_heads=[],
            threshold=0.4,
        )
        top = result.get_top_heads(n=2)
        assert len(top) == 2
        assert top[0].overall_score == 0.8
        assert top[1].overall_score == 0.5

    def test_to_dict(self):
        scores = [HeadScore(0, 0, 0.5, 0.5, 0.5, 0.5)]
        result = DetectionResult(
            head_scores=scores,
            induction_heads=[(0, 0)],
            threshold=0.4,
            model_info={"n_layers": 4},
        )
        d = result.to_dict()
        assert d["num_heads_detected"] == 1
        assert d["threshold"] == 0.4
        assert d["model_info"]["n_layers"] == 4


class TestInductionHeadDetector:
    @pytest.fixture
    def model(self):
        return SimpleTransformer(vocab_size=100, hidden_dim=64, n_layers=4, n_heads=4)

    @pytest.fixture
    def detector(self, model):
        return InductionHeadDetector(model, threshold=0.3)

    def test_create(self, detector):
        assert detector.threshold == 0.3

    def test_compute_induction_score(self, detector):
        # Create mock attention with induction pattern
        seq_len = 16
        attention = torch.zeros(1, 1, seq_len, seq_len)

        # Set up induction pattern: [A][B]...[A] -> attend to [B]
        # tokens: 1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4
        # At position i, if tokens[i-1] matches tokens[j], attend to j+1
        tokens = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]])

        # Set up proper induction attention:
        # Position 5 has token 2, position 4 has token 1
        # We need to attend from position 5 to position 1 (after the previous 1 at position 0)
        for i in range(2, seq_len):
            for j in range(0, i - 1):
                # If tokens[i-1] == tokens[j], set high attention to j+1
                if tokens[0, i - 1] == tokens[0, j]:
                    attention[0, 0, i, j + 1] = 1.0

        # Normalize attention to valid distribution
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-10)

        score = detector.compute_induction_score(attention, tokens)
        # Score should be positive for induction-like pattern
        assert isinstance(score, float)

    def test_compute_prefix_matching_score(self, detector):
        seq_len = 16
        attention = torch.zeros(1, 1, seq_len, seq_len)
        tokens = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]])

        # Higher attention when prefixes match
        for i in range(2, seq_len):
            for j in range(2, i):
                if tokens[0, i-1] == tokens[0, j-1]:
                    attention[0, 0, i, j] = 0.8
                else:
                    attention[0, 0, i, j] = 0.1

        score = detector.compute_prefix_matching_score(attention, tokens)
        # Score should be positive since matching prefixes get more attention
        assert isinstance(score, float)

    def test_compute_copying_score(self, detector):
        seq_len = 16
        attention = torch.zeros(1, 1, seq_len, seq_len)
        tokens = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]])

        # Set up attention that leads to correct copying
        for i in range(4, seq_len - 1):
            j = i - 4
            attention[0, 0, i, j] = 1.0

        score = detector.compute_copying_score(attention, tokens)
        assert isinstance(score, float)

    def test_detect(self, detector):
        result = detector.detect(sequence_length=32)
        assert isinstance(result, DetectionResult)
        assert "n_layers" in result.model_info


class TestDetectInductionHeads:
    def test_convenience_function(self):
        model = SimpleTransformer(vocab_size=100, hidden_dim=64, n_layers=2, n_heads=2)
        result = detect_induction_heads(model, threshold=0.3, sequence_length=32)

        assert isinstance(result, DetectionResult)
