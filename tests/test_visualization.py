"""Tests for induction_head_detector.visualization module."""

import pytest
import torch

from induction_head_detector.visualization import (
    plot_attention_pattern,
    plot_head_scores,
    plot_induction_stripe,
    plot_layer_comparison,
    plot_attention_comparison,
    create_analysis_report,
)
from induction_head_detector.detection import DetectionResult, HeadScore


class TestPlotAttentionPattern:
    def test_basic_plot(self):
        attention = torch.softmax(torch.randn(16, 16), dim=-1)
        output = plot_attention_pattern(attention)

        assert isinstance(output, str)
        assert "Attention Pattern:" in output
        assert "Legend:" in output

    def test_with_batch_dimension(self):
        attention = torch.softmax(torch.randn(1, 16, 16), dim=-1)
        output = plot_attention_pattern(attention)

        assert "Attention Pattern:" in output

    def test_max_size_limit(self):
        attention = torch.softmax(torch.randn(100, 100), dim=-1)
        output = plot_attention_pattern(attention, max_size=10)

        # Should be limited in size
        lines = output.split("\n")
        assert len(lines) < 20


class TestPlotHeadScores:
    def test_basic_plot(self):
        scores = [
            {"layer": 0, "head": 0, "overall_score": 0.8},
            {"layer": 1, "head": 1, "overall_score": 0.6},
            {"layer": 2, "head": 0, "overall_score": 0.4},
        ]
        output = plot_head_scores(scores)

        assert isinstance(output, str)
        assert "Induction Head Scores:" in output

    def test_empty_scores(self):
        output = plot_head_scores([])
        assert "No scores to display" in output

    def test_sorting(self):
        scores = [
            {"layer": 0, "head": 0, "overall_score": 0.3},
            {"layer": 1, "head": 0, "overall_score": 0.9},
            {"layer": 2, "head": 0, "overall_score": 0.5},
        ]
        output = plot_head_scores(scores, n_heads=3)

        # Highest score should appear first
        lines = output.split("\n")
        # Find the data lines
        data_lines = [l for l in lines if "L01H00" in l or "L00H00" in l or "L02H00" in l]
        assert "L01H00" in data_lines[0]  # Layer 1 has highest score


class TestPlotInductionStripe:
    def test_basic_plot(self):
        attention = torch.softmax(torch.randn(32, 32), dim=-1)
        output = plot_induction_stripe(attention, period=4)

        assert isinstance(output, str)
        assert "Induction Stripe" in output
        assert "Mean diagonal attention:" in output

    def test_short_sequence(self):
        attention = torch.softmax(torch.randn(4, 4), dim=-1)
        output = plot_induction_stripe(attention, period=8)

        assert "Sequence too short" in output


class TestPlotLayerComparison:
    def test_basic_plot(self):
        layer_scores = {0: 0.3, 1: 0.5, 2: 0.8, 3: 0.4}
        output = plot_layer_comparison(layer_scores)

        assert isinstance(output, str)
        assert "Induction Score by Layer:" in output
        assert "L00" in output
        assert "L02" in output

    def test_empty_scores(self):
        output = plot_layer_comparison({})
        assert "No layer scores" in output


class TestPlotAttentionComparison:
    def test_basic_comparison(self):
        attention1 = torch.softmax(torch.randn(1, 16, 16), dim=-1)
        attention2 = torch.softmax(torch.randn(1, 16, 16), dim=-1)

        output = plot_attention_comparison(
            attention1, attention2,
            labels=("Head A", "Head B"),
            max_size=10,
        )

        assert isinstance(output, str)
        assert "Head A" in output
        assert "Head B" in output


class TestCreateAnalysisReport:
    def test_basic_report(self):
        scores = [
            HeadScore(0, 0, 0.5, 0.4, 0.6, 0.5),
            HeadScore(1, 0, 0.3, 0.2, 0.4, 0.3),
        ]
        result = DetectionResult(
            head_scores=scores,
            induction_heads=[(0, 0)],
            threshold=0.4,
            model_info={"n_layers": 4},
        )

        report = create_analysis_report(result)

        assert isinstance(report, str)
        assert "INDUCTION HEAD ANALYSIS REPORT" in report
        assert "Threshold:" in report
        assert "Heads detected:" in report

    def test_detailed_report(self):
        scores = [
            HeadScore(0, 0, 0.5, 0.4, 0.6, 0.5),
        ]
        result = DetectionResult(
            head_scores=scores,
            induction_heads=[(0, 0)],
            threshold=0.4,
        )

        report = create_analysis_report(result, detailed=True)

        assert "Top Heads by Score" in report
        assert "Score Distribution" in report

    def test_report_with_model_info(self):
        result = DetectionResult(
            head_scores=[],
            induction_heads=[],
            threshold=0.4,
            model_info={"n_layers": 6, "note": "test"},
        )

        report = create_analysis_report(result)

        assert "n_layers" in report
        assert "note" in report
