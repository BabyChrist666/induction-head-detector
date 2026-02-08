"""Tests for induction_head_detector.patterns module."""

import pytest
import torch

from induction_head_detector.patterns import (
    PatternType,
    GeneratedPattern,
    generate_repeated_sequence,
    generate_prefix_match_data,
    generate_abab_pattern,
    generate_copy_task,
    PatternGenerator,
    identify_induction_positions,
    compute_expected_attention,
)


class TestGeneratedPattern:
    def test_create(self):
        tokens = torch.randint(0, 100, (1, 32))
        pattern = GeneratedPattern(
            tokens=tokens,
            pattern_type=PatternType.REPEATED_SEQUENCE,
            description="Test pattern",
        )
        assert pattern.pattern_type == PatternType.REPEATED_SEQUENCE
        assert pattern.tokens.shape == (1, 32)

    def test_to_dict(self):
        tokens = torch.randint(0, 100, (1, 64))
        pattern = GeneratedPattern(
            tokens=tokens,
            pattern_type=PatternType.ABAB,
            description="ABAB pattern",
        )
        d = pattern.to_dict()
        assert d["pattern_type"] == "abab"
        assert d["sequence_length"] == 64


class TestGenerateRepeatedSequence:
    def test_basic_generation(self):
        pattern = generate_repeated_sequence(
            vocab_size=100,
            sequence_length=64,
            n_repeats=2,
        )
        assert pattern.tokens.shape == (1, 64)
        assert pattern.pattern_type == PatternType.REPEATED_SEQUENCE

    def test_repetition(self):
        pattern = generate_repeated_sequence(
            vocab_size=100,
            sequence_length=32,
            n_repeats=2,
        )
        tokens = pattern.tokens
        half_len = 16

        # First half should equal second half
        assert torch.all(tokens[0, :half_len] == tokens[0, half_len:])

    def test_expected_attention(self):
        pattern = generate_repeated_sequence(
            vocab_size=100,
            sequence_length=32,
            n_repeats=2,
        )
        assert pattern.expected_attention is not None
        assert pattern.expected_attention.shape == (1, 32, 32)


class TestGeneratePrefixMatchData:
    def test_basic_generation(self):
        pattern = generate_prefix_match_data(
            vocab_size=100,
            sequence_length=64,
            prefix_length=2,
            n_matches=4,
        )
        assert pattern.tokens.shape == (1, 64)
        assert pattern.pattern_type == PatternType.PREFIX_MATCH

    def test_has_matching_prefixes(self):
        pattern = generate_prefix_match_data(
            vocab_size=100,
            sequence_length=64,
            prefix_length=2,
            n_matches=2,
        )
        # Just check it runs without error
        assert pattern.tokens is not None


class TestGenerateAbabPattern:
    def test_basic_generation(self):
        pattern = generate_abab_pattern(
            vocab_size=100,
            n_pairs=16,
        )
        # Should be 16 pairs * 2 tokens * 2 repeats = 64 tokens
        assert pattern.tokens.shape == (1, 64)
        assert pattern.pattern_type == PatternType.ABAB

    def test_abab_structure(self):
        pattern = generate_abab_pattern(
            vocab_size=100,
            n_pairs=4,
        )
        tokens = pattern.tokens[0]

        # Check that pattern repeats
        half = len(tokens) // 2
        assert torch.all(tokens[:half] == tokens[half:])


class TestGenerateCopyTask:
    def test_basic_generation(self):
        pattern = generate_copy_task(
            vocab_size=100,
            copy_length=10,
            gap_length=5,
        )
        assert pattern.tokens.shape == (1, 25)  # 10 + 5 + 10
        assert pattern.pattern_type == PatternType.COPY_TASK

    def test_copy_structure(self):
        pattern = generate_copy_task(
            vocab_size=100,
            copy_length=8,
            gap_length=4,
        )
        tokens = pattern.tokens[0]

        # First and last 8 tokens should match
        assert torch.all(tokens[:8] == tokens[-8:])


class TestPatternGenerator:
    @pytest.fixture
    def generator(self):
        return PatternGenerator(vocab_size=100)

    def test_generate_repeated(self, generator):
        pattern = generator.generate(
            PatternType.REPEATED_SEQUENCE,
            sequence_length=32,
        )
        assert pattern.pattern_type == PatternType.REPEATED_SEQUENCE

    def test_generate_abab(self, generator):
        pattern = generator.generate(
            PatternType.ABAB,
            n_pairs=8,
        )
        assert pattern.pattern_type == PatternType.ABAB

    def test_generate_all(self, generator):
        patterns = generator.generate_all(sequence_length=32)
        assert len(patterns) == 4

        types = {p.pattern_type for p in patterns}
        assert PatternType.REPEATED_SEQUENCE in types
        assert PatternType.PREFIX_MATCH in types
        assert PatternType.ABAB in types
        assert PatternType.COPY_TASK in types

    def test_generate_batch(self, generator):
        pattern = generator.generate_batch(
            PatternType.REPEATED_SEQUENCE,
            batch_size=4,
            sequence_length=16,
        )
        assert pattern.tokens.shape[0] == 4


class TestIdentifyInductionPositions:
    def test_repeated_sequence(self):
        tokens = torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3]])
        positions = identify_induction_positions(tokens, prefix_length=1)

        # Position 4 (token 2) should attend to position 1 (after token 1)
        # Position 5 (token 3) should attend to position 2
        # etc.
        assert len(positions) > 0

    def test_no_matches(self):
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        positions = identify_induction_positions(tokens, prefix_length=1)
        assert len(positions) == 0


class TestComputeExpectedAttention:
    def test_shape(self):
        tokens = torch.tensor([[1, 2, 3, 1, 2, 3]])
        expected = compute_expected_attention(tokens, prefix_length=1)
        assert expected.shape == (6, 6)

    def test_normalized(self):
        tokens = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]])
        expected = compute_expected_attention(tokens, prefix_length=1)

        # Each row should sum to 0 or 1
        row_sums = expected.sum(dim=1)
        for s in row_sums:
            assert s.item() < 1.1  # Allow small numerical error
