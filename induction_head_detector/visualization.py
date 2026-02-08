"""
Visualization tools for induction head analysis.

ASCII-based visualizations for terminal display.
"""

import torch
from typing import Optional, List, Dict, Any


def plot_attention_pattern(
    attention: torch.Tensor,
    tokens: Optional[torch.Tensor] = None,
    max_size: int = 20,
) -> str:
    """
    Create ASCII visualization of attention pattern.

    Args:
        attention: Attention weights [seq, seq] or [batch, seq, seq]
        tokens: Optional token IDs for labeling
        max_size: Maximum size to display

    Returns:
        ASCII string representation.
    """
    if attention.dim() == 3:
        attention = attention[0]

    seq_len = min(attention.shape[0], max_size)
    attention = attention[:seq_len, :seq_len]

    # Normalize to [0, 1]
    attention = attention.cpu()
    max_val = attention.max()
    if max_val > 0:
        attention = attention / max_val

    chars = " .:-=+*#@"

    lines = ["Attention Pattern:"]

    # Header
    header = "     "
    for j in range(seq_len):
        header += f"{j % 10}"
    lines.append(header)
    lines.append("    +" + "-" * seq_len)

    # Rows
    for i in range(seq_len):
        row = f" {i:2d} |"
        for j in range(seq_len):
            if j > i:
                row += " "  # Future positions (masked)
            else:
                val = attention[i, j].item()
                idx = min(int(val * (len(chars) - 1)), len(chars) - 1)
                row += chars[idx]
        lines.append(row)

    lines.append("")
    lines.append("Legend: ' '=0, '@'=max attention")

    return "\n".join(lines)


def plot_head_scores(
    scores: List[Dict[str, Any]],
    sort_by: str = "overall_score",
    n_heads: int = 10,
    width: int = 50,
) -> str:
    """
    Visualize scores for multiple heads.

    Args:
        scores: List of score dictionaries with layer, head, scores.
        sort_by: Which score to sort by.
        n_heads: Number of heads to show.
        width: Width of bars.

    Returns:
        ASCII visualization.
    """
    lines = ["Induction Head Scores:"]
    lines.append("=" * width)

    # Sort and limit
    sorted_scores = sorted(scores, key=lambda s: s.get(sort_by, 0), reverse=True)
    sorted_scores = sorted_scores[:n_heads]

    if not sorted_scores:
        return "No scores to display"

    max_score = max(s.get(sort_by, 0) for s in sorted_scores)
    bar_width = width - 25

    for s in sorted_scores:
        layer = s.get("layer", 0)
        head = s.get("head", 0)
        score = s.get(sort_by, 0)

        bar_len = int((score / max_score) * bar_width) if max_score > 0 else 0
        bar = "#" * bar_len

        lines.append(f"  L{layer:02d}H{head:02d} |{bar} {score:.3f}")

    lines.append("=" * width)

    return "\n".join(lines)


def plot_induction_stripe(
    attention: torch.Tensor,
    period: int,
    height: int = 10,
    width: int = 50,
) -> str:
    """
    Visualize the "induction stripe" at a specific offset.

    An induction head should show strong attention at offset = period
    (attending to token after previous occurrence).

    Args:
        attention: Attention weights [seq, seq] or [batch, seq, seq]
        period: The period/offset to visualize
        height: Height of visualization
        width: Width of visualization

    Returns:
        ASCII visualization of diagonal attention.
    """
    if attention.dim() == 3:
        attention = attention[0]

    attention = attention.cpu()
    seq_len = attention.shape[0]

    lines = [f"Induction Stripe (offset={period}):"]
    lines.append("-" * width)

    # Collect diagonal values
    diagonal_values = []
    for i in range(period, seq_len):
        diagonal_values.append(attention[i, i - period].item())

    if not diagonal_values:
        return "Sequence too short for this offset"

    # Normalize
    max_val = max(diagonal_values) if diagonal_values else 1
    min_val = min(diagonal_values) if diagonal_values else 0

    # Sample if too many
    if len(diagonal_values) > width - 10:
        step = len(diagonal_values) // (width - 10)
        diagonal_values = diagonal_values[::step]

    # Create plot
    grid = [[" " for _ in range(len(diagonal_values))] for _ in range(height)]

    for x, val in enumerate(diagonal_values):
        if max_val > min_val:
            y = int((1 - (val - min_val) / (max_val - min_val)) * (height - 1))
        else:
            y = height // 2
        y = max(0, min(height - 1, y))
        grid[y][x] = "*"

    # Add y-axis labels
    for i, row in enumerate(grid):
        val = max_val - (i / (height - 1)) * (max_val - min_val) if height > 1 else max_val
        label = f"{val:.2f}" if i in [0, height // 2, height - 1] else "    "
        lines.append(f"{label} |{''.join(row)}")

    lines.append("     +" + "-" * len(diagonal_values))
    lines.append(f"     Position {period} -> {seq_len}")
    lines.append("")
    lines.append(f"Mean diagonal attention: {sum(diagonal_values)/len(diagonal_values):.4f}")

    return "\n".join(lines)


def plot_layer_comparison(
    layer_scores: Dict[int, float],
    width: int = 50,
) -> str:
    """
    Compare induction scores across layers.
    """
    lines = ["Induction Score by Layer:"]
    lines.append("-" * width)

    if not layer_scores:
        return "No layer scores to display"

    max_score = max(layer_scores.values())
    bar_width = width - 15

    for layer in sorted(layer_scores.keys()):
        score = layer_scores[layer]
        bar_len = int((score / max_score) * bar_width) if max_score > 0 else 0
        bar = "=" * bar_len
        lines.append(f"  L{layer:02d} |{bar} {score:.3f}")

    lines.append("-" * width)

    return "\n".join(lines)


def plot_attention_comparison(
    attention1: torch.Tensor,
    attention2: torch.Tensor,
    labels: tuple = ("Head 1", "Head 2"),
    max_size: int = 15,
) -> str:
    """
    Compare two attention patterns side by side.
    """
    if attention1.dim() == 3:
        attention1 = attention1[0]
    if attention2.dim() == 3:
        attention2 = attention2[0]

    seq_len = min(attention1.shape[0], attention2.shape[0], max_size)
    attention1 = attention1[:seq_len, :seq_len].cpu()
    attention2 = attention2[:seq_len, :seq_len].cpu()

    chars = " .:-=+*#@"

    def to_char(val):
        idx = min(int(val * (len(chars) - 1)), len(chars) - 1)
        return chars[idx]

    lines = [f"{labels[0]:^{seq_len + 5}}    {labels[1]:^{seq_len + 5}}"]
    lines.append(" " * 5 + "-" * seq_len + "    " + " " * 5 + "-" * seq_len)

    for i in range(seq_len):
        row1 = f" {i:2d} |"
        row2 = f" {i:2d} |"

        for j in range(seq_len):
            if j > i:
                row1 += " "
                row2 += " "
            else:
                val1 = attention1[i, j].item() / (attention1[i, :i+1].max().item() + 1e-8)
                val2 = attention2[i, j].item() / (attention2[i, :i+1].max().item() + 1e-8)
                row1 += to_char(val1)
                row2 += to_char(val2)

        lines.append(row1 + "    " + row2)

    return "\n".join(lines)


def create_analysis_report(
    detection_result,
    detailed: bool = True,
) -> str:
    """
    Create a comprehensive analysis report.
    """
    lines = ["=" * 60]
    lines.append("INDUCTION HEAD ANALYSIS REPORT")
    lines.append("=" * 60)

    # Model info
    lines.append("\n--- Model Information ---")
    for key, val in detection_result.model_info.items():
        lines.append(f"  {key}: {val}")

    # Detection summary
    lines.append("\n--- Detection Summary ---")
    lines.append(f"  Threshold: {detection_result.threshold}")
    lines.append(f"  Heads detected: {detection_result.num_heads_detected}")

    if detection_result.induction_heads:
        lines.append(f"  Induction heads: {detection_result.induction_heads}")

    # Top heads
    if detailed and detection_result.head_scores:
        lines.append("\n--- Top Heads by Score ---")
        top_heads = detection_result.get_top_heads(n=5)

        for h in top_heads:
            lines.append(f"  Layer {h.layer}, Head {h.head}:")
            lines.append(f"    Induction: {h.induction_score:.4f}")
            lines.append(f"    Prefix matching: {h.prefix_matching_score:.4f}")
            lines.append(f"    Copying: {h.copying_score:.4f}")
            lines.append(f"    Overall: {h.overall_score:.4f}")

        # Score distribution
        lines.append("\n--- Score Distribution ---")
        all_scores = [h.overall_score for h in detection_result.head_scores]
        if all_scores:
            lines.append(f"  Mean: {sum(all_scores)/len(all_scores):.4f}")
            lines.append(f"  Max: {max(all_scores):.4f}")
            lines.append(f"  Min: {min(all_scores):.4f}")

            # Count by threshold
            above_threshold = sum(1 for s in all_scores if s >= detection_result.threshold)
            lines.append(f"  Above threshold: {above_threshold}/{len(all_scores)}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)
