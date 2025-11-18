"""
Robustness computation utilities for STL specifications.

Robustness provides a quantitative measure of how well a specification
is satisfied or violated.
"""

from typing import List, Tuple, Dict
import math


def compute_robustness(
    values: List[float],
    operator: str = 'min'
) -> float:
    """
    Compute aggregate robustness over multiple values.

    Args:
        values: List of robustness values
        operator: Aggregation operator ('min', 'max', 'avg')

    Returns:
        float: Aggregated robustness value
    """
    if not values:
        return 0.0

    if operator == 'min':
        return min(values)
    elif operator == 'max':
        return max(values)
    elif operator == 'avg':
        return sum(values) / len(values)
    else:
        raise ValueError(f"Unknown aggregation operator: {operator}")


def robustness_distance(
    value: float,
    threshold: float,
    comparison: str
) -> float:
    """
    Compute robustness as distance from threshold.

    Args:
        value: Signal value
        threshold: Threshold for comparison
        comparison: Type of comparison ('<', '<=', '>', '>=')

    Returns:
        float: Robustness degree (positive = satisfied, negative = violated)
    """
    if comparison in ['<', '<=']:
        # Satisfied when value < threshold
        # Robustness = threshold - value
        return threshold - value
    elif comparison in ['>', '>=']:
        # Satisfied when value > threshold
        # Robustness = value - threshold
        return value - threshold
    else:
        raise ValueError(f"Unsupported comparison operator: {comparison}")


def temporal_robustness_always(
    signal: List[Tuple[float, float]],
    threshold: float,
    comparison: str,
    time_interval: Tuple[float, float] = None
) -> float:
    """
    Compute robustness for 'always' (globally) operator.

    The robustness of G[a,b](φ) is the minimum robustness of φ over [a,b].

    Args:
        signal: Time-series signal as [(time, value), ...]
        threshold: Threshold value
        comparison: Comparison operator
        time_interval: Optional (start, end) time bounds

    Returns:
        float: Minimum robustness over the interval
    """
    if time_interval:
        start, end = time_interval
        filtered_signal = [(t, v) for t, v in signal if start <= t <= end]
    else:
        filtered_signal = signal

    if not filtered_signal:
        return 0.0

    robustness_values = [
        robustness_distance(value, threshold, comparison)
        for _, value in filtered_signal
    ]

    return min(robustness_values)


def temporal_robustness_eventually(
    signal: List[Tuple[float, float]],
    threshold: float,
    comparison: str,
    time_interval: Tuple[float, float] = None
) -> float:
    """
    Compute robustness for 'eventually' (finally) operator.

    The robustness of F[a,b](φ) is the maximum robustness of φ over [a,b].

    Args:
        signal: Time-series signal as [(time, value), ...]
        threshold: Threshold value
        comparison: Comparison operator
        time_interval: Optional (start, end) time bounds

    Returns:
        float: Maximum robustness over the interval
    """
    if time_interval:
        start, end = time_interval
        filtered_signal = [(t, v) for t, v in signal if start <= t <= end]
    else:
        filtered_signal = signal

    if not filtered_signal:
        return 0.0

    robustness_values = [
        robustness_distance(value, threshold, comparison)
        for _, value in filtered_signal
    ]

    return max(robustness_values)


def combine_robustness_and(
    robustness1: float,
    robustness2: float
) -> float:
    """
    Combine two robustness values with AND operator.

    The robustness of (φ1 ∧ φ2) is min(ρ(φ1), ρ(φ2)).

    Args:
        robustness1: First robustness value
        robustness2: Second robustness value

    Returns:
        float: Combined robustness
    """
    return min(robustness1, robustness2)


def combine_robustness_or(
    robustness1: float,
    robustness2: float
) -> float:
    """
    Combine two robustness values with OR operator.

    The robustness of (φ1 ∨ φ2) is max(ρ(φ1), ρ(φ2)).

    Args:
        robustness1: First robustness value
        robustness2: Second robustness value

    Returns:
        float: Combined robustness
    """
    return max(robustness1, robustness2)


def normalize_robustness(
    robustness: float,
    min_value: float = -1.0,
    max_value: float = 1.0
) -> float:
    """
    Normalize robustness value to a specified range.

    Useful for comparing robustness values across different specifications
    with different scales.

    Args:
        robustness: Raw robustness value
        min_value: Minimum value for normalization
        max_value: Maximum value for normalization

    Returns:
        float: Normalized robustness in [min_value, max_value]
    """
    # Clip to avoid extreme values
    if robustness > 0:
        return min(robustness, max_value)
    else:
        return max(robustness, min_value)


def robustness_summary(
    robustness_dict: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute summary statistics for a set of robustness values.

    Args:
        robustness_dict: Dictionary mapping specification names to robustness values

    Returns:
        dict: Summary statistics including min, max, avg, num_satisfied, num_violated
    """
    if not robustness_dict:
        return {
            'min': 0.0,
            'max': 0.0,
            'avg': 0.0,
            'num_satisfied': 0,
            'num_violated': 0,
            'total': 0
        }

    values = list(robustness_dict.values())

    return {
        'min': min(values),
        'max': max(values),
        'avg': sum(values) / len(values),
        'num_satisfied': sum(1 for v in values if v >= 0),
        'num_violated': sum(1 for v in values if v < 0),
        'total': len(values)
    }
