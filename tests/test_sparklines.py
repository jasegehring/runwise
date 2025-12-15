"""Tests for sparkline generation."""


from runwise.sparklines import (
    format_metric_with_spark,
    sparkline,
    sparkline_with_stats,
    trend_indicator,
)


class TestSparkline:
    """Tests for sparkline function."""

    def test_basic_ascending(self):
        """Ascending values produce ascending sparkline."""
        result = sparkline([1, 2, 3, 4, 5])
        assert result[0] < result[-1]  # First char lower than last
        assert len(result) == 5

    def test_basic_descending(self):
        """Descending values produce descending sparkline."""
        result = sparkline([5, 4, 3, 2, 1])
        assert result[0] > result[-1]  # First char higher than last

    def test_constant_values(self):
        """Constant values produce uniform sparkline."""
        result = sparkline([5, 5, 5, 5])
        assert len(set(result)) == 1  # All chars the same

    def test_empty_list(self):
        """Empty list returns empty string."""
        assert sparkline([]) == ""

    def test_single_value(self):
        """Single value returns single char."""
        result = sparkline([5])
        assert len(result) == 1

    def test_nan_handling(self):
        """NaN values are filtered out."""
        result = sparkline([1, float('nan'), 2, 3])
        assert "?" not in result  # Valid data exists
        assert len(result) == 3  # NaN excluded

    def test_all_nan_returns_question(self):
        """All NaN returns question mark."""
        result = sparkline([float('nan'), float('nan')])
        assert result == "?"

    def test_none_handling(self):
        """None values are filtered out."""
        result = sparkline([1, None, 2, 3])
        assert len(result) == 3

    def test_width_downsampling(self):
        """Width parameter downsamples to specified length."""
        values = list(range(100))
        result = sparkline(values, width=10)
        assert len(result) == 10

    def test_custom_min_max(self):
        """Custom min/max values affect scaling."""
        # With default scaling, [5, 6, 7] fills full range
        default = sparkline([5, 6, 7])
        # With wider range, they're compressed
        custom = sparkline([5, 6, 7], min_val=0, max_val=100)
        assert default != custom

    def test_spike_visible(self):
        """Spike in data is visible in sparkline."""
        values = [1, 1, 1, 10, 1, 1]
        result = sparkline(values)
        # The 4th character (index 3) should be highest
        chars = list(result)
        assert chars[3] == max(chars)


class TestTrendIndicator:
    """Tests for trend_indicator function."""

    def test_decreasing_trend(self):
        """Decreasing values show down arrow."""
        result = trend_indicator([10, 8, 6, 4, 2])
        assert result == "↓"

    def test_increasing_trend(self):
        """Increasing values show up arrow."""
        result = trend_indicator([1, 3, 5, 7, 9])
        assert result == "↑"

    def test_stable_trend(self):
        """Stable values show right arrow."""
        # Use truly stable values with minimal variation
        result = trend_indicator([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        assert result == "→"

    def test_volatile_trend(self):
        """Highly volatile values show tilde."""
        result = trend_indicator([1, 10, 1, 10, 1, 10, 1, 10])
        assert result == "~"

    def test_single_value(self):
        """Single value returns stable indicator."""
        result = trend_indicator([5])
        assert result == "→"

    def test_empty_list(self):
        """Empty list returns stable indicator."""
        result = trend_indicator([])
        assert result == "→"


class TestSparklineWithStats:
    """Tests for sparkline_with_stats function."""

    def test_basic_format(self):
        """Output includes sparkline and values."""
        result = sparkline_with_stats([1.5, 1.2, 0.8, 0.3])
        assert "→" in result  # Contains arrow
        assert "1.5" in result or "1.50" in result  # First value
        assert "0.3" in result or "0.30" in result  # Last value

    def test_empty_data(self):
        """Empty list returns placeholder."""
        result = sparkline_with_stats([])
        assert "no data" in result

    def test_all_invalid(self):
        """All invalid data returns placeholder."""
        result = sparkline_with_stats([float('nan'), None])
        assert "no valid data" in result


class TestFormatMetricWithSpark:
    """Tests for format_metric_with_spark function."""

    def test_basic_format(self):
        """Output includes metric name and sparkline."""
        result = format_metric_with_spark("loss", [1.0, 0.8, 0.5, 0.2])
        assert "loss:" in result
        assert "→" in result or "↓" in result  # Has trend indicator

    def test_empty_values(self):
        """Empty values handled gracefully."""
        result = format_metric_with_spark("loss", [])
        assert "loss:" in result
        assert "no data" in result

    def test_width_parameter(self):
        """Width affects sparkline length."""
        result_5 = format_metric_with_spark("loss", list(range(100)), width=5)
        result_20 = format_metric_with_spark("loss", list(range(100)), width=20)
        # Different widths produce different results
        assert len(result_5) != len(result_20)
