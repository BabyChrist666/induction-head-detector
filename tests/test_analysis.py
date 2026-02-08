"""Tests for induction_head_detector.analysis module."""

import pytest
import torch
import torch.nn as nn

from induction_head_detector.analysis import (
    AttentionPattern,
    HeadBehavior,
    InductionHeadAnalyzer,
    analyze_head_attention,
)


class SimpleTransformer(nn.Module):
    """Simple transformer for testing."""
    def __init__(self, vocab_size=100, hidden_dim=64, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


class TestAttentionPattern:
    def test_create(self):
        attention = torch.randn(1, 16, 16)
        pattern = AttentionPattern(
            layer=2,
            head=3,
            attention=attention,
        )
        assert pattern.layer == 2
        assert pattern.head == 3
        assert pattern.seq_length == 16

    def test_get_diagonal_score(self):
        attention = torch.zeros(1, 16, 16)
        # Set up diagonal at offset 1
        for i in range(1, 16):
            attention[0, i, i - 1] = 0.5

        pattern = AttentionPattern(layer=0, head=0, attention=attention)
        score = pattern.get_diagonal_score(offset=1)
        assert abs(score - 0.5) < 0.01

    def test_get_induction_diagonal_score(self):
        attention = torch.zeros(1, 16, 16)
        period = 4
        for i in range(period, 16):
            attention[0, i, i - period] = 0.8

        pattern = AttentionPattern(layer=0, head=0, attention=attention)
        score = pattern.get_induction_diagonal_score(period=period)
        assert abs(score - 0.8) < 0.01


class TestHeadBehavior:
    def test_create(self):
        behavior = HeadBehavior(
            layer=1,
            head=2,
            is_induction=True,
            is_previous_token=False,
            is_duplicate_token=False,
            primary_behavior="induction",
            confidence=0.8,
        )
        assert behavior.is_induction
        assert behavior.primary_behavior == "induction"

    def test_to_dict(self):
        behavior = HeadBehavior(
            layer=0,
            head=1,
            is_induction=True,
            is_previous_token=False,
            is_duplicate_token=False,
            primary_behavior="induction",
            confidence=0.75,
            metrics={"test": 0.5},
        )
        d = behavior.to_dict()
        assert d["layer"] == 0
        assert d["is_induction"] is True
        assert d["confidence"] == 0.75


class TestInductionHeadAnalyzer:
    @pytest.fixture
    def model(self):
        return SimpleTransformer(vocab_size=100, hidden_dim=64, n_layers=4)

    @pytest.fixture
    def analyzer(self, model):
        return InductionHeadAnalyzer(model)

    def test_create(self, analyzer):
        assert analyzer.model is not None

    def test_analyze_head_no_attention(self, analyzer):
        tokens = torch.randint(0, 100, (1, 16))
        behavior = analyzer.analyze_head(
            layer=0,
            head=0,
            tokens=tokens,
            attention=None,
        )
        assert isinstance(behavior, HeadBehavior)

    def test_analyze_head_with_attention(self, analyzer):
        tokens = torch.randint(0, 100, (1, 16))
        attention = torch.softmax(torch.randn(1, 16, 16), dim=-1)

        behavior = analyzer.analyze_head(
            layer=0,
            head=0,
            tokens=tokens,
            attention=attention,
        )
        assert isinstance(behavior, HeadBehavior)
        assert "prev_token_attn" in behavior.metrics

    def test_find_induction_heads(self, analyzer):
        tokens = torch.randint(0, 100, (1, 16))
        heads = analyzer.find_induction_heads(tokens, threshold=0.3)
        assert isinstance(heads, list)


class TestAnalyzeHeadAttention:
    def test_with_3d_attention(self):
        attention = torch.softmax(torch.randn(1, 16, 16), dim=-1)
        tokens = torch.randint(0, 100, (1, 16))

        metrics = analyze_head_attention(attention, tokens)

        assert "previous_token_attention" in metrics
        assert "average_entropy" in metrics
        assert "average_peak_attention" in metrics

    def test_with_4d_attention(self):
        attention = torch.softmax(torch.randn(1, 4, 16, 16), dim=-1)
        tokens = torch.randint(0, 100, (1, 16))

        metrics = analyze_head_attention(attention, tokens)

        assert "previous_token_attention" in metrics
        assert "diagonal_1_attention" in metrics

    def test_diagonal_metrics(self):
        attention = torch.zeros(1, 16, 16)
        # Strong previous token attention
        for i in range(1, 16):
            attention[0, i, i - 1] = 0.8

        tokens = torch.randint(0, 100, (1, 16))
        metrics = analyze_head_attention(attention, tokens)

        assert metrics["previous_token_attention"] > 0.5
        assert metrics["diagonal_1_attention"] > 0.5
