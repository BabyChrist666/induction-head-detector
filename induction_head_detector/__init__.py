"""
Induction Head Detector - Find and analyze induction heads in transformers.

Tools for detecting induction heads (attention heads that implement
pattern completion: [A][B]...[A] -> [B]) and analyzing their behavior.
"""

from induction_head_detector.detection import (
    InductionHeadDetector,
    DetectionResult,
    HeadScore,
    detect_induction_heads,
)
from induction_head_detector.patterns import (
    generate_repeated_sequence,
    generate_prefix_match_data,
    PatternType,
    PatternGenerator,
)
from induction_head_detector.analysis import (
    InductionHeadAnalyzer,
    AttentionPattern,
    HeadBehavior,
    analyze_head_attention,
)
from induction_head_detector.visualization import (
    plot_attention_pattern,
    plot_head_scores,
    plot_induction_stripe,
    create_analysis_report,
)

__version__ = "0.1.0"
__all__ = [
    # Detection
    "InductionHeadDetector",
    "DetectionResult",
    "HeadScore",
    "detect_induction_heads",
    # Patterns
    "generate_repeated_sequence",
    "generate_prefix_match_data",
    "PatternType",
    "PatternGenerator",
    # Analysis
    "InductionHeadAnalyzer",
    "AttentionPattern",
    "HeadBehavior",
    "analyze_head_attention",
    # Visualization
    "plot_attention_pattern",
    "plot_head_scores",
    "plot_induction_stripe",
    "create_analysis_report",
]
