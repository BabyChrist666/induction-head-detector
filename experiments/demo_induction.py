"""
Demo: Detecting Induction Heads in a Simple Transformer

This experiment demonstrates how to find induction heads in a simple
transformer model and analyze their attention patterns.

Induction heads implement: [A][B]...[A] -> [B]
They're crucial for in-context learning and pattern completion.
"""

import torch
import torch.nn as nn

import sys
sys.path.insert(0, "..")

from induction_head_detector import (
    InductionHeadDetector,
    detect_induction_heads,
    PatternGenerator,
    InductionHeadAnalyzer,
    plot_attention_pattern,
    plot_head_scores,
    plot_induction_stripe,
    create_analysis_report,
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


class DemoTransformer(nn.Module):
    """Simple transformer for demonstration."""
    def __init__(self, vocab_size=100, hidden_dim=64, n_layers=6, n_heads=4):
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


def main():
    print("=" * 60)
    print("INDUCTION HEAD DETECTION DEMO")
    print("=" * 60)
    print()

    # Create model
    print("[1] Creating demo transformer...")
    model = DemoTransformer(
        vocab_size=100,
        hidden_dim=64,
        n_layers=6,
        n_heads=4
    )
    model.eval()
    print(f"    Layers: 6, Heads per layer: 4")
    print(f"    Total heads to analyze: 24")
    print()

    # Run detection
    print("[2] Running induction head detection...")
    detector = InductionHeadDetector(model, threshold=0.3)
    result = detector.detect(sequence_length=64)
    print()

    # Show results
    print("[3] Detection Results:")
    print("-" * 40)
    print(f"    Threshold: {result.threshold}")
    print(f"    Heads detected: {result.num_heads_detected}")
    print()

    if result.induction_heads:
        print("    Induction heads found:")
        for layer, head in result.induction_heads:
            print(f"      - Layer {layer}, Head {head}")
    else:
        print("    No strong induction heads found")
        print("    (This is expected for random weights)")
    print()

    # Top heads by score
    print("[4] Top Heads by Overall Score:")
    print("-" * 40)
    top_heads = result.get_top_heads(n=8)
    for score in top_heads:
        bar_len = int(score.overall_score * 20)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"    L{score.layer:02d}H{score.head:02d}: {score.overall_score:.3f} [{bar}]")
    print()

    # Score breakdown for top head
    if top_heads:
        print("[5] Score Breakdown (Top Head):")
        print("-" * 40)
        top = top_heads[0]
        print(f"    Head: Layer {top.layer}, Head {top.head}")
        print(f"    Induction Score:      {top.induction_score:.4f}")
        print(f"    Prefix Matching:      {top.prefix_matching_score:.4f}")
        print(f"    Copying Score:        {top.copying_score:.4f}")
        print(f"    Overall Score:        {top.overall_score:.4f}")
    print()

    # Pattern generation
    print("[6] Test Pattern Generation:")
    print("-" * 40)
    gen = PatternGenerator(vocab_size=50)

    patterns = gen.generate_all(seq_len=32)
    for pattern in patterns:
        print(f"    {pattern.pattern_type:12}: {pattern.tokens.shape}")
        print(f"                  first 12 tokens: {pattern.tokens[0,:12].tolist()}")
    print()

    # ASCII visualization
    print("[7] Head Score Visualization:")
    print("-" * 40)
    score_dicts = [s.to_dict() for s in result.head_scores]
    print(plot_head_scores(score_dicts, n_heads=10))

    # Full report
    print()
    print("[8] Full Analysis Report:")
    print("-" * 40)
    report = create_analysis_report(result, detailed=True)
    print(report)

    print()
    print("=" * 60)
    print("Demo complete!")
    print()
    print("In a trained model, you would typically see:")
    print("  - Induction heads emerging in layers 2-6")
    print("  - Scores > 0.5 for strong induction heads")
    print("  - Clear induction stripes in attention patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
