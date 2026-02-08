# Induction Head Detector

[![Tests](https://github.com/BabyChrist666/induction-head-detector/actions/workflows/tests.yml/badge.svg)](https://github.com/BabyChrist666/induction-head-detector/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BabyChrist666/induction-head-detector/branch/master/graph/badge.svg)](https://codecov.io/gh/BabyChrist666/induction-head-detector)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Find and analyze induction heads in any transformer model.

Induction heads implement a key copying mechanism: when they see `[A][B]...[A]`, they predict `[B]` by attending back to what followed the previous occurrence of `[A]`.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from induction_head_detector import (
    InductionHeadDetector,
    detect_induction_heads,
    PatternGenerator,
    InductionHeadAnalyzer,
)

# Detect induction heads in your model
result = detect_induction_heads(model, threshold=0.4)

print(f"Found {result.num_heads_detected} induction heads")
for layer, head in result.induction_heads:
    print(f"  Layer {layer}, Head {head}")

# Detailed analysis
detector = InductionHeadDetector(model, threshold=0.3)
result = detector.detect(sequence_length=64)

for score in result.get_top_heads(n=5):
    print(f"L{score.layer}H{score.head}: {score.overall_score:.3f}")
```

## Detection Metrics

The detector uses three complementary metrics:

### 1. Induction Score
Measures attention to the position after previous token matches:
- When `tokens[i-1] == tokens[j]`, measure attention from `i` to `j+1`
- High scores indicate pattern: "I've seen A before, attend to what followed"

### 2. Prefix Matching Score
Compares attention when prefixes match vs. don't match:
- If current position's prefix matches a previous prefix, is there more attention?
- Captures the "fuzzy matching" aspect of induction

### 3. Copying Score
Measures whether attention leads to correct predictions:
- Where does max attention point?
- Does the next token after that position match what we're predicting?

## Pattern Generation

Generate test sequences designed to elicit induction behavior:

```python
from induction_head_detector import PatternGenerator

gen = PatternGenerator(vocab_size=50, device="cuda")

# Repeated sequence: [A,B,C,D,A,B,C,D,A,B,C,D]
pattern = gen.generate("repeated", seq_len=64, n_repeats=4)

# ABAB pattern: [A,B,A,B,A,B,...]
pattern = gen.generate("abab", seq_len=64)

# Copy task: [content][SEP][content]
pattern = gen.generate("copy", seq_len=64)

# Generate all pattern types
all_patterns = gen.generate_all(seq_len=64)
```

## Attention Analysis

Analyze attention patterns in detail:

```python
from induction_head_detector import InductionHeadAnalyzer, analyze_head_attention

# Full model analysis
analyzer = InductionHeadAnalyzer(model)
induction_heads = analyzer.find_induction_heads(threshold=0.4)

# Analyze specific attention matrix
behavior = analyze_head_attention(
    attention_pattern,  # [seq_len, seq_len]
    tokens,             # [seq_len]
    period=16,          # For repeated sequences
)

print(f"Diagonal score: {behavior.diagonal_score:.3f}")
print(f"Induction stripe: {behavior.induction_stripe_score:.3f}")
```

## Visualization

ASCII visualizations work in any terminal:

```python
from induction_head_detector import (
    plot_attention_pattern,
    plot_head_scores,
    plot_induction_stripe,
    create_analysis_report,
)

# Visualize attention pattern
print(plot_attention_pattern(attention))

# Score comparison
print(plot_head_scores([s.to_dict() for s in result.head_scores]))

# Induction stripe analysis
print(plot_induction_stripe(attention, period=8))

# Full report
print(create_analysis_report(result, detailed=True))
```

## Example Output

```
INDUCTION HEAD ANALYSIS REPORT
==============================

Model Info:
  n_layers: 12

Detection Summary:
  Threshold: 0.40
  Heads detected: 4

Top Heads by Score:
  L05H07: 0.823  [########--]
  L06H03: 0.756  [########--]
  L05H02: 0.698  [#######---]
  L07H01: 0.612  [######----]
```

## Understanding Induction Heads

Induction heads are a crucial circuit in transformers that enable in-context learning. They work in two stages:

1. **Previous Token Head** (often in early layers): Creates Q/K composition where the key is the previous token
2. **Induction Head** (middle layers): Uses this to implement [A][B]...[A] -> [B]

Key papers:
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

## Testing

```bash
pytest tests/ -v
```

56 tests covering:
- Detection algorithms
- Pattern generation
- Attention analysis
- Visualization

## License

MIT

