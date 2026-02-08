#!/usr/bin/env python3
"""
Induction Head Detection Example

This example demonstrates how to find and analyze induction heads
in transformer models - attention heads that implement the pattern:
[A][B]...[A] -> predict [B]
"""

import numpy as np
from induction_head_detector import (
    InductionHeadDetector,
    AttentionPatternAnalyzer,
    DetectorConfig,
)


def main():
    print("=" * 60)
    print("Induction Head Detector")
    print("=" * 60)

    # Create mock attention patterns (in practice, extract from real model)
    # Shape: (n_layers, n_heads, seq_len, seq_len)
    n_layers = 12
    n_heads = 12
    seq_len = 128

    # Generate mock attention patterns with one clear induction head
    attention_patterns = generate_mock_patterns(n_layers, n_heads, seq_len)

    # Initialize detector
    config = DetectorConfig(
        induction_threshold=0.5,
        min_pattern_strength=0.1,
        analyze_composition=True,
    )
    detector = InductionHeadDetector(config)

    # Example 1: Detect induction heads
    print("\n1. Detecting induction heads...")

    induction_heads = detector.detect(attention_patterns)

    print(f"   Found {len(induction_heads)} induction heads:")
    for head in induction_heads[:5]:
        print(f"   - Layer {head.layer}, Head {head.head}: score={head.score:.3f}")

    # Example 2: Analyze specific head
    print("\n2. Analyzing top induction head...")

    if induction_heads:
        top_head = induction_heads[0]
        analysis = detector.analyze_head(
            attention_patterns[top_head.layer, top_head.head],
            layer=top_head.layer,
            head=top_head.head,
        )

        print(f"   Pattern type: {analysis.pattern_type}")
        print(f"   Copying strength: {analysis.copying_strength:.3f}")
        print(f"   Matching strength: {analysis.matching_strength:.3f}")
        print(f"   Previous token attention: {analysis.prev_token_attention:.3f}")

    # Example 3: Find head composition
    print("\n3. Analyzing head composition...")

    compositions = detector.find_compositions(attention_patterns)

    print(f"   Found {len(compositions)} composition patterns:")
    for comp in compositions[:3]:
        print(f"   - L{comp.layer1}H{comp.head1} -> L{comp.layer2}H{comp.head2}")
        print(f"     Type: {comp.composition_type}, Strength: {comp.strength:.3f}")

    # Example 4: Visualize attention pattern
    print("\n4. Generating visualizations...")

    if induction_heads:
        top_head = induction_heads[0]
        detector.visualize_attention_pattern(
            attention_patterns[top_head.layer, top_head.head],
            save_path="induction_head_pattern.png",
            title=f"Layer {top_head.layer}, Head {top_head.head}",
        )
        print("   Saved induction_head_pattern.png")

    # Example 5: Test on repeated sequence
    print("\n5. Testing on repeated sequence pattern...")

    # Create [A][B][C][A][B] pattern
    repeated_pattern = create_repeated_sequence_pattern(seq_len)
    test_results = detector.test_copying_behavior(
        attention_patterns,
        repeated_pattern,
    )

    print(f"   Copying accuracy: {test_results.accuracy:.2%}")
    print(f"   Best copying head: L{test_results.best_head[0]}H{test_results.best_head[1]}")

    print("\n" + "=" * 60)
    print("Detection complete!")
    print("=" * 60)


def generate_mock_patterns(n_layers, n_heads, seq_len):
    """Generate mock attention patterns with induction-like behavior."""
    patterns = np.random.rand(n_layers, n_heads, seq_len, seq_len)

    # Make patterns valid (sum to 1 along last axis)
    patterns = patterns / patterns.sum(axis=-1, keepdims=True)

    # Add strong induction pattern to layer 5, head 7
    induction_layer, induction_head = 5, 7
    for i in range(seq_len):
        if i > 0:
            # Attend to position that came before matching token
            patterns[induction_layer, induction_head, i, i-1] = 0.8

    # Re-normalize
    patterns = patterns / patterns.sum(axis=-1, keepdims=True)

    return patterns


def create_repeated_sequence_pattern(seq_len):
    """Create a repeated sequence for testing."""
    # [A B C D A B C D ...]
    pattern_len = 4
    pattern = list(range(pattern_len)) * (seq_len // pattern_len + 1)
    return np.array(pattern[:seq_len])


if __name__ == "__main__":
    main()
