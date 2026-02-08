"""
Pattern generation for induction head testing.

Generates sequences with specific patterns that induction heads
should recognize and complete.
"""

import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


class PatternType(Enum):
    """Types of patterns for testing induction heads."""
    REPEATED_SEQUENCE = "repeated_sequence"
    PREFIX_MATCH = "prefix_match"
    RANDOM_REPEATED = "random_repeated"
    ABAB = "abab"
    COPY_TASK = "copy_task"


@dataclass
class GeneratedPattern:
    """A generated pattern for testing."""
    tokens: torch.Tensor
    pattern_type: PatternType
    expected_attention: Optional[torch.Tensor] = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type.value,
            "sequence_length": self.tokens.shape[-1],
            "description": self.description,
        }


def generate_repeated_sequence(
    vocab_size: int = 100,
    sequence_length: int = 64,
    n_repeats: int = 2,
    device: str = "cpu",
) -> GeneratedPattern:
    """
    Generate a repeated random sequence.

    Example: [A B C D E][A B C D E] where [...] is the repeated part.
    An induction head should attend to the previous occurrence.
    """
    unit_length = sequence_length // n_repeats
    base_sequence = torch.randint(1, vocab_size, (1, unit_length), device=device)
    tokens = base_sequence.repeat(1, n_repeats)

    # Expected attention: position i should attend to i - unit_length
    expected = torch.zeros(1, sequence_length, sequence_length, device=device)
    for i in range(unit_length, sequence_length):
        prev_pos = i - unit_length
        expected[0, i, prev_pos] = 1.0

    return GeneratedPattern(
        tokens=tokens,
        pattern_type=PatternType.REPEATED_SEQUENCE,
        expected_attention=expected,
        description=f"Random sequence repeated {n_repeats} times",
    )


def generate_prefix_match_data(
    vocab_size: int = 100,
    sequence_length: int = 64,
    prefix_length: int = 2,
    n_matches: int = 4,
    device: str = "cpu",
) -> GeneratedPattern:
    """
    Generate sequence with matching prefixes at specific positions.

    Creates patterns like: ... [A B] X ... [A B] ?
    where the induction head should predict the token after [A B].
    """
    tokens = torch.randint(1, vocab_size, (1, sequence_length), device=device)

    # Create matching prefixes at specific positions
    match_positions = []
    step = sequence_length // (n_matches + 1)

    for i in range(n_matches):
        pos1 = step * (i + 1) - prefix_length
        pos2 = pos1 + step

        if pos2 + prefix_length < sequence_length:
            # Copy prefix from pos1 to pos2
            tokens[0, pos2:pos2 + prefix_length] = tokens[0, pos1:pos1 + prefix_length]
            match_positions.append((pos1, pos2))

    return GeneratedPattern(
        tokens=tokens,
        pattern_type=PatternType.PREFIX_MATCH,
        description=f"Sequence with {len(match_positions)} matching prefixes",
    )


def generate_abab_pattern(
    vocab_size: int = 100,
    n_pairs: int = 16,
    device: str = "cpu",
) -> GeneratedPattern:
    """
    Generate ABAB pattern sequence.

    Creates: A1 B1 A2 B2 A3 B3 ... A1 B1 A2 B2 ...
    Classic test for induction heads.
    """
    # Generate random A and B tokens
    a_tokens = torch.randint(1, vocab_size // 2, (1, n_pairs), device=device)
    b_tokens = torch.randint(vocab_size // 2, vocab_size, (1, n_pairs), device=device)

    # Interleave
    first_half = torch.stack([a_tokens, b_tokens], dim=-1).reshape(1, -1)
    tokens = first_half.repeat(1, 2)

    return GeneratedPattern(
        tokens=tokens,
        pattern_type=PatternType.ABAB,
        description=f"ABAB pattern with {n_pairs} pairs repeated twice",
    )


def generate_copy_task(
    vocab_size: int = 100,
    copy_length: int = 10,
    gap_length: int = 5,
    device: str = "cpu",
) -> GeneratedPattern:
    """
    Generate a copy task sequence.

    Creates: [sequence][gap][sequence]
    Tests if model can copy a sequence after a gap.
    """
    sequence = torch.randint(1, vocab_size, (1, copy_length), device=device)
    gap = torch.zeros(1, gap_length, dtype=torch.long, device=device)  # Use 0 as gap token
    tokens = torch.cat([sequence, gap, sequence], dim=1)

    return GeneratedPattern(
        tokens=tokens,
        pattern_type=PatternType.COPY_TASK,
        description=f"Copy task: {copy_length} tokens with {gap_length} gap",
    )


class PatternGenerator:
    """
    Generator for various induction head test patterns.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        device: str = "cpu",
    ):
        self.vocab_size = vocab_size
        self.device = device

    def generate(
        self,
        pattern_type: PatternType,
        **kwargs,
    ) -> GeneratedPattern:
        """Generate a specific pattern type."""
        generators = {
            PatternType.REPEATED_SEQUENCE: self._gen_repeated,
            PatternType.PREFIX_MATCH: self._gen_prefix_match,
            PatternType.ABAB: self._gen_abab,
            PatternType.COPY_TASK: self._gen_copy_task,
        }

        if pattern_type not in generators:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

        return generators[pattern_type](**kwargs)

    def generate_all(
        self,
        sequence_length: int = 64,
    ) -> List[GeneratedPattern]:
        """Generate one of each pattern type."""
        patterns = []

        patterns.append(self._gen_repeated(sequence_length=sequence_length))
        patterns.append(self._gen_prefix_match(sequence_length=sequence_length))
        patterns.append(self._gen_abab(n_pairs=sequence_length // 4))
        patterns.append(self._gen_copy_task(copy_length=sequence_length // 4))

        return patterns

    def _gen_repeated(self, **kwargs) -> GeneratedPattern:
        return generate_repeated_sequence(
            vocab_size=self.vocab_size,
            device=self.device,
            **kwargs,
        )

    def _gen_prefix_match(self, **kwargs) -> GeneratedPattern:
        return generate_prefix_match_data(
            vocab_size=self.vocab_size,
            device=self.device,
            **kwargs,
        )

    def _gen_abab(self, **kwargs) -> GeneratedPattern:
        return generate_abab_pattern(
            vocab_size=self.vocab_size,
            device=self.device,
            **kwargs,
        )

    def _gen_copy_task(self, **kwargs) -> GeneratedPattern:
        return generate_copy_task(
            vocab_size=self.vocab_size,
            device=self.device,
            **kwargs,
        )

    def generate_batch(
        self,
        pattern_type: PatternType,
        batch_size: int,
        **kwargs,
    ) -> GeneratedPattern:
        """Generate a batch of patterns of the same type."""
        patterns = [self.generate(pattern_type, **kwargs) for _ in range(batch_size)]

        # Stack tokens
        tokens = torch.cat([p.tokens for p in patterns], dim=0)

        return GeneratedPattern(
            tokens=tokens,
            pattern_type=pattern_type,
            description=f"Batch of {batch_size} {pattern_type.value} patterns",
        )


def identify_induction_positions(
    tokens: torch.Tensor,
    prefix_length: int = 1,
) -> List[Tuple[int, int]]:
    """
    Find positions where induction should occur.

    Returns list of (current_pos, should_attend_to_pos) pairs.
    """
    seq_len = tokens.shape[-1]
    positions = []

    for i in range(prefix_length + 1, seq_len):
        current_prefix = tokens[0, i - prefix_length:i]

        for j in range(prefix_length, i - prefix_length):
            prev_prefix = tokens[0, j - prefix_length:j]

            if torch.all(current_prefix == prev_prefix):
                # Position i should attend to j (the position after matching prefix)
                positions.append((i, j))

    return positions


def compute_expected_attention(
    tokens: torch.Tensor,
    prefix_length: int = 1,
) -> torch.Tensor:
    """
    Compute expected attention pattern for a perfect induction head.

    Returns attention matrix where entry [i, j] is high if
    position i should attend to position j for induction.
    """
    seq_len = tokens.shape[-1]
    expected = torch.zeros(seq_len, seq_len)

    positions = identify_induction_positions(tokens, prefix_length)

    for current_pos, attend_pos in positions:
        expected[current_pos, attend_pos] = 1.0

    # Normalize rows
    row_sums = expected.sum(dim=1, keepdim=True)
    expected = expected / (row_sums + 1e-8)

    return expected
