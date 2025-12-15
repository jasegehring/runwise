"""
Sparkline generation for token-efficient trend visualization.

Converts numeric sequences to compact visual representations using
Unicode block characters. A single sparkline conveys trend information
in ~10 tokens vs 50+ for raw numbers.

Examples:
    >>> sparkline([1, 2, 3, 4, 5])
    '▁▂▄▆█'
    >>> sparkline([5, 4, 3, 2, 1])
    '█▆▄▂▁'
    >>> sparkline([1, 1, 1, 5, 1])  # Spike detection
    '▁▁▁█▁'
"""

from typing import Optional

# Unicode block characters for sparklines (8 levels)
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(
    values: list[float],
    width: Optional[int] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> str:
    """
    Convert a list of values to a sparkline string.

    Args:
        values: List of numeric values
        width: Optional fixed width (downsamples if needed)
        min_val: Optional minimum value for scaling (auto-detected if None)
        max_val: Optional maximum value for scaling (auto-detected if None)

    Returns:
        Sparkline string using Unicode block characters
    """
    if not values:
        return ""

    # Filter out None and NaN values
    clean_values = [v for v in values if v is not None and v == v]  # NaN != NaN
    if not clean_values:
        return "?"  # No valid data

    # Downsample if width specified
    if width and len(clean_values) > width:
        clean_values = _downsample(clean_values, width)

    # Calculate range
    if min_val is None:
        min_val = min(clean_values)
    if max_val is None:
        max_val = max(clean_values)

    # Handle constant values
    value_range = max_val - min_val
    if value_range == 0:
        # All values are the same - use middle character
        return SPARK_CHARS[len(SPARK_CHARS) // 2] * len(clean_values)

    # Map values to characters
    result = []
    for v in clean_values:
        # Normalize to 0-1 range
        normalized = (v - min_val) / value_range
        # Map to character index (0 to 7)
        idx = int(normalized * (len(SPARK_CHARS) - 1))
        idx = max(0, min(len(SPARK_CHARS) - 1, idx))  # Clamp
        result.append(SPARK_CHARS[idx])

    return "".join(result)


def sparkline_with_stats(
    values: list[float],
    width: int = 10,
) -> str:
    """
    Generate sparkline with inline min/max stats.

    Example: '▁▂▄▆█ (0.5→0.1)'
    """
    if not values:
        return "- (no data)"

    clean_values = [v for v in values if v is not None and v == v]
    if not clean_values:
        return "? (no valid data)"

    spark = sparkline(clean_values, width=width)

    # First and last valid values
    first = clean_values[0]
    last = clean_values[-1]

    # Format compactly
    def fmt(v):
        if abs(v) < 0.001 or abs(v) > 1000:
            return f"{v:.1e}"
        elif abs(v) < 1:
            return f"{v:.3f}"
        else:
            return f"{v:.2f}"

    return f"{spark} ({fmt(first)}→{fmt(last)})"


def trend_indicator(values: list[float]) -> str:
    """
    Get a simple trend indicator.

    Returns: '↑' (improving), '↓' (worsening), '→' (stable), '~' (volatile)
    """
    if len(values) < 2:
        return "→"

    clean_values = [v for v in values if v is not None and v == v]
    if len(clean_values) < 2:
        return "→"

    first_half = clean_values[: len(clean_values) // 2]
    second_half = clean_values[len(clean_values) // 2:]

    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)

    # Calculate volatility
    if len(clean_values) > 4:
        diffs = [abs(clean_values[i] - clean_values[i - 1]) for i in range(1, len(clean_values))]
        avg_diff = sum(diffs) / len(diffs)
        value_range = max(clean_values) - min(clean_values)
        if value_range > 0 and avg_diff / value_range > 0.3:
            return "~"  # Volatile

    # Check trend
    change_ratio = (avg_second - avg_first) / (abs(avg_first) + 1e-10)

    if change_ratio < -0.05:
        return "↓"  # Decreasing (good for loss)
    elif change_ratio > 0.05:
        return "↑"  # Increasing
    else:
        return "→"  # Stable


def calculate_slope(
    values: list[float],
    steps: Optional[list[int]] = None,
) -> tuple[float, float, str]:
    """
    Calculate robust slope using Theil-Sen estimator.

    The Theil-Sen estimator calculates the median of slopes between all pairs
    of points, making it robust to outliers (up to 29% of data can be outliers).

    Args:
        values: List of metric values
        steps: Optional list of step numbers (if None, uses indices)

    Returns:
        Tuple of (slope_per_step, slope_per_1000_steps, direction)
        - slope_per_step: Raw slope (change per step)
        - slope_per_1000_steps: Normalized slope (change per 1000 steps)
        - direction: "improving", "declining", or "stable"
    """
    # Clean values (remove None and NaN)
    if steps is None:
        pairs = [(i, v) for i, v in enumerate(values) if v is not None and v == v]
    else:
        pairs = [(s, v) for s, v in zip(steps, values) if v is not None and v == v]

    if len(pairs) < 2:
        return 0.0, 0.0, "stable"

    # Calculate all pairwise slopes
    slopes = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            x1, y1 = pairs[i]
            x2, y2 = pairs[j]
            if x2 != x1:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                slopes.append(slope)

    if not slopes:
        return 0.0, 0.0, "stable"

    # Theil-Sen: take median of all slopes
    slopes.sort()
    n = len(slopes)
    if n % 2 == 0:
        median_slope = (slopes[n // 2 - 1] + slopes[n // 2]) / 2
    else:
        median_slope = slopes[n // 2]

    # Calculate slope per 1000 steps
    slope_per_1000 = median_slope * 1000

    # Determine direction based on significance
    # Use coefficient of variation to determine if change is meaningful
    mean_val = sum(v for _, v in pairs) / len(pairs)
    if abs(mean_val) > 1e-10:
        relative_change = abs(slope_per_1000) / abs(mean_val)
        if relative_change < 0.01:  # Less than 1% change per 1000 steps
            direction = "stable"
        elif median_slope < 0:
            direction = "declining"
        else:
            direction = "improving"
    else:
        direction = "stable"

    return median_slope, slope_per_1000, direction


def calculate_windowed_slopes(
    values: list[float],
    steps: Optional[list[int]] = None,
    window: int = 100,
) -> list[tuple[int, float]]:
    """
    Calculate slope over sliding windows.

    Useful for seeing how the rate of change evolves during training.

    Args:
        values: List of metric values
        steps: Optional list of step numbers
        window: Window size for slope calculation

    Returns:
        List of (step, slope) tuples for each window
    """
    if steps is None:
        steps = list(range(len(values)))

    # Clean data
    pairs = [(s, v) for s, v in zip(steps, values) if v is not None and v == v]

    if len(pairs) < window:
        return []

    results = []
    for i in range(len(pairs) - window + 1):
        window_pairs = pairs[i:i + window]
        window_steps = [p[0] for p in window_pairs]
        window_values = [p[1] for p in window_pairs]

        slope, _, _ = calculate_slope(window_values, window_steps)
        center_step = window_pairs[window // 2][0]
        results.append((center_step, slope))

    return results


def _downsample(values: list[float], target_width: int) -> list[float]:
    """Downsample values to target width using averaging."""
    if len(values) <= target_width:
        return values

    result = []
    bucket_size = len(values) / target_width

    for i in range(target_width):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size)
        bucket = values[start:end]
        if bucket:
            result.append(sum(bucket) / len(bucket))

    return result


def format_metric_with_spark(
    name: str,
    values: list[float],
    width: int = 10,
    higher_is_better: bool = False,
) -> str:
    """
    Format a metric with sparkline and trend.

    Example: 'loss: ▇▆▅▃▂▁ ↓ (1.5→0.2)'
    """
    if not values:
        return f"{name}: - (no data)"

    spark = sparkline(values, width=width)
    trend = trend_indicator(values)

    clean_values = [v for v in values if v is not None and v == v]
    if not clean_values:
        return f"{name}: ? (no valid data)"

    first = clean_values[0]
    last = clean_values[-1]

    def fmt(v):
        if abs(v) < 0.001 or abs(v) > 1000:
            return f"{v:.1e}"
        elif abs(v) < 1:
            return f"{v:.3f}"
        else:
            return f"{v:.2f}"

    # Flip trend meaning based on higher_is_better
    if higher_is_better:
        # For accuracy: ↑ is good, ↓ is bad
        pass
    else:
        # For loss: ↓ is good, ↑ is bad - keep as is since that's the natural interpretation
        pass

    return f"{name}: {spark} {trend} ({fmt(first)}→{fmt(last)})"
