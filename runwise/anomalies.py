"""
Anomaly detection for training runs.

Lightweight, deterministic heuristics that produce zero tokens when
everything's fine, but flag issues clearly when problems occur.

Detection types:
- Loss spikes (robust z-score using MAD)
- Overfitting (val/train ratio divergence)
- Gradient issues (vanishing/exploding)
- Plateaus (no improvement)
- NaN/Inf detection
- Throughput drops (system issues)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Anomaly:
    """A detected anomaly in training."""

    type: str  # spike, overfit, plateau, gradient, nan, system
    severity: str  # warning, critical
    message: str
    step: Optional[int] = None
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        icon = "!!" if self.severity == "critical" else "!"
        step_str = f" (step {self.step:,})" if self.step else ""
        return f"{icon} {self.type.upper()}: {self.message}{step_str}"

    def compact(self) -> str:
        """Single-line compact format for token efficiency."""
        icon = "!!" if self.severity == "critical" else "!"
        return f"{icon} {self.message}"


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection thresholds."""

    # Spike detection (MAD-based)
    spike_threshold: float = 3.5  # MAD score threshold
    spike_window: int = 100  # Rolling window size

    # Overfitting detection
    overfit_ratio_threshold: float = 1.5  # Val/train ratio increase
    overfit_baseline_steps: tuple = (100, 500)  # Steps for baseline ratio

    # Plateau detection
    plateau_min_steps: int = 500  # Min steps to check for plateau
    plateau_improvement_threshold: float = 0.01  # 1% improvement required

    # Gradient detection
    gradient_vanish_threshold: float = 1e-7  # Below this = vanishing
    gradient_explode_multiplier: float = 10.0  # Above 10x mean = exploding
    gradient_sustained_steps: int = 10  # Sustained low/high for this long

    # Throughput detection
    throughput_drop_threshold: float = 0.4  # 40% drop = problem


def detect_anomalies(
    history: list[dict],
    config: Optional[AnomalyConfig] = None,
    loss_key: str = "loss",
    val_loss_key: str = "val_loss",
    grad_norm_key: str = "grad_norm",
    throughput_key: str = "steps_per_sec",
) -> list[Anomaly]:
    """
    Detect anomalies in training history.

    Args:
        history: List of metric dictionaries (one per step)
        config: AnomalyConfig for threshold tuning
        loss_key: Key for training loss metric
        val_loss_key: Key for validation loss metric
        grad_norm_key: Key for gradient norm metric
        throughput_key: Key for throughput metric

    Returns:
        List of detected Anomaly objects (empty if run is healthy)
    """
    if not history:
        return []

    if config is None:
        config = AnomalyConfig()

    anomalies = []

    # Extract metric series
    losses = _extract_series(history, loss_key)
    val_losses = _extract_series(history, val_loss_key)
    grad_norms = _extract_series(history, grad_norm_key)
    throughputs = _extract_series(history, throughput_key)

    # Check for NaN/Inf first (critical)
    nan_anomalies = _detect_nan_inf(history, loss_key, val_loss_key)
    anomalies.extend(nan_anomalies)

    # Check for spikes
    if losses:
        spike_anomalies = _detect_spikes(losses, config, "loss")
        anomalies.extend(spike_anomalies)

    # Check for overfitting
    if losses and val_losses:
        overfit_anomalies = _detect_overfitting(losses, val_losses, config)
        anomalies.extend(overfit_anomalies)

    # Check for plateaus
    if losses:
        plateau_anomalies = _detect_plateau(losses, config)
        anomalies.extend(plateau_anomalies)

    # Check gradients
    if grad_norms:
        grad_anomalies = _detect_gradient_issues(grad_norms, config)
        anomalies.extend(grad_anomalies)

    # Check throughput
    if throughputs:
        throughput_anomalies = _detect_throughput_issues(throughputs, config)
        anomalies.extend(throughput_anomalies)

    # Sort by severity (critical first) then by step
    anomalies.sort(key=lambda a: (0 if a.severity == "critical" else 1, a.step or 0))

    return anomalies


def _extract_series(history: list[dict], key: str) -> list[tuple[int, float]]:
    """Extract (step, value) pairs for a metric key."""
    series = []
    for i, record in enumerate(history):
        if key in record:
            value = record[key]
            if isinstance(value, (int, float)) and value == value:  # NaN check
                step = record.get("_step", record.get("step", i))
                series.append((step, float(value)))
    return series


def _detect_nan_inf(
    history: list[dict], loss_key: str, val_loss_key: str
) -> list[Anomaly]:
    """Detect NaN or Inf values in critical metrics."""
    anomalies = []

    for i, record in enumerate(history):
        step = record.get("_step", record.get("step", i))

        for key in [loss_key, val_loss_key]:
            if key in record:
                val = record[key]
                if val is None:
                    continue
                if isinstance(val, float):
                    if val != val:  # NaN
                        anomalies.append(
                            Anomaly(
                                type="nan",
                                severity="critical",
                                message=f"NaN detected in {key}",
                                step=step,
                            )
                        )
                        return anomalies  # Stop at first NaN
                    if abs(val) == float("inf"):
                        anomalies.append(
                            Anomaly(
                                type="nan",
                                severity="critical",
                                message=f"Inf detected in {key}",
                                step=step,
                            )
                        )
                        return anomalies

    return anomalies


def _detect_spikes(
    series: list[tuple[int, float]], config: AnomalyConfig, metric_name: str
) -> list[Anomaly]:
    """
    Detect spikes using Median Absolute Deviation (MAD).

    MAD is robust to outliers unlike standard deviation.
    Score = |x - median| / MAD
    If score > threshold, it's an anomaly.
    """
    if len(series) < config.spike_window:
        return []

    anomalies = []
    values = [v for _, v in series]

    # Use sliding window for local anomaly detection
    window_size = min(config.spike_window, len(values))

    for i in range(window_size, len(values)):
        window = values[i - window_size: i]
        current_value = values[i]
        current_step = series[i][0]

        # Calculate MAD
        median = sorted(window)[len(window) // 2]
        deviations = [abs(v - median) for v in window]
        mad = sorted(deviations)[len(deviations) // 2]

        if mad == 0:
            continue  # All values identical

        # Calculate robust z-score
        score = abs(current_value - median) / mad

        if score > config.spike_threshold:
            # Check if it's actually a spike (sudden increase)
            if current_value > median * 2:
                anomalies.append(
                    Anomaly(
                        type="spike",
                        severity="warning",
                        message=f"{metric_name} spike: {current_value:.4g} vs median {median:.4g}",
                        step=current_step,
                        details={"value": current_value, "median": median, "mad_score": score},
                    )
                )

    return anomalies[:3]  # Limit to first 3 spikes


def _detect_overfitting(
    train_losses: list[tuple[int, float]],
    val_losses: list[tuple[int, float]],
    config: AnomalyConfig,
) -> list[Anomaly]:
    """
    Detect overfitting by tracking val/train loss ratio.

    If the ratio increases significantly from baseline, overfitting is occurring.
    """
    if len(train_losses) < config.overfit_baseline_steps[1]:
        return []
    if len(val_losses) < 10:
        return []

    # Build step -> value maps
    train_map = {step: val for step, val in train_losses}
    val_map = {step: val for step, val in val_losses}

    # Find common steps for ratio calculation
    common_steps = sorted(set(train_map.keys()) & set(val_map.keys()))
    if len(common_steps) < 10:
        return []

    # Calculate baseline ratio (early in training)
    baseline_steps = [
        s for s in common_steps
        if config.overfit_baseline_steps[0] <= s <= config.overfit_baseline_steps[1]
    ]
    if len(baseline_steps) < 3:
        baseline_steps = common_steps[: min(20, len(common_steps) // 4)]

    baseline_ratios = []
    for step in baseline_steps:
        train_val = train_map[step]
        val_val = val_map[step]
        if train_val > 0:
            baseline_ratios.append(val_val / train_val)

    if not baseline_ratios:
        return []

    baseline_ratio = sum(baseline_ratios) / len(baseline_ratios)

    # Check recent ratio
    recent_steps = common_steps[-20:]
    recent_ratios = []
    for step in recent_steps:
        train_val = train_map[step]
        val_val = val_map[step]
        if train_val > 0:
            recent_ratios.append(val_val / train_val)

    if not recent_ratios:
        return []

    current_ratio = sum(recent_ratios) / len(recent_ratios)

    # Check for significant increase
    if baseline_ratio > 0 and current_ratio > baseline_ratio * config.overfit_ratio_threshold:
        increase_pct = (current_ratio / baseline_ratio - 1) * 100
        return [
            Anomaly(
                type="overfit",
                severity="warning",
                message=f"Overfitting: val/train ratio +{increase_pct:.0f}% vs baseline",
                step=common_steps[-1],
                details={
                    "baseline_ratio": baseline_ratio,
                    "current_ratio": current_ratio,
                    "increase_pct": increase_pct,
                },
            )
        ]

    return []


def _detect_plateau(
    losses: list[tuple[int, float]], config: AnomalyConfig
) -> list[Anomaly]:
    """
    Detect training plateau (no improvement for extended period).
    """
    if len(losses) < config.plateau_min_steps:
        return []

    values = [v for _, v in losses]

    # Compare recent quarter to previous quarter
    quarter_size = len(values) // 4
    if quarter_size < 50:
        return []

    recent_quarter = values[-quarter_size:]
    previous_quarter = values[-2 * quarter_size: -quarter_size]

    recent_mean = sum(recent_quarter) / len(recent_quarter)
    previous_mean = sum(previous_quarter) / len(previous_quarter)

    # Check if improvement is below threshold
    if previous_mean > 0:
        improvement = (previous_mean - recent_mean) / previous_mean

        if improvement < config.plateau_improvement_threshold:
            # Also check minimum value improvement
            recent_min = min(recent_quarter)
            overall_min = min(values)

            if recent_min >= overall_min * 0.99:  # No new minimum in recent quarter
                return [
                    Anomaly(
                        type="plateau",
                        severity="warning",
                        message=f"Plateau: <{config.plateau_improvement_threshold * 100:.0f}% improvement in last {quarter_size} steps",
                        step=losses[-1][0],
                        details={
                            "recent_mean": recent_mean,
                            "previous_mean": previous_mean,
                            "improvement_pct": improvement * 100,
                        },
                    )
                ]

    return []


def _detect_gradient_issues(
    grad_norms: list[tuple[int, float]], config: AnomalyConfig
) -> list[Anomaly]:
    """
    Detect vanishing or exploding gradients.
    """
    if len(grad_norms) < config.gradient_sustained_steps:
        return []

    anomalies = []
    values = [v for _, v in grad_norms]

    # Calculate running mean
    mean_grad = sum(values) / len(values)

    # Check for sustained vanishing gradients
    recent = values[-config.gradient_sustained_steps:]
    if all(v < config.gradient_vanish_threshold for v in recent):
        anomalies.append(
            Anomaly(
                type="gradient",
                severity="critical",
                message=f"Vanishing gradients: <{config.gradient_vanish_threshold} for {config.gradient_sustained_steps} steps",
                step=grad_norms[-1][0],
                details={"recent_mean": sum(recent) / len(recent)},
            )
        )

    # Check for exploding gradients
    if mean_grad > 0:
        threshold = mean_grad * config.gradient_explode_multiplier
        if any(v > threshold for v in recent):
            max_val = max(recent)
            anomalies.append(
                Anomaly(
                    type="gradient",
                    severity="warning",
                    message=f"Gradient explosion: {max_val:.2e} ({max_val / mean_grad:.1f}x mean)",
                    step=grad_norms[-1][0],
                    details={"max_grad": max_val, "mean_grad": mean_grad},
                )
            )

    return anomalies


def _detect_throughput_issues(
    throughputs: list[tuple[int, float]], config: AnomalyConfig
) -> list[Anomaly]:
    """
    Detect throughput drops (potential hardware/system issues).
    """
    if len(throughputs) < 20:
        return []

    values = [v for _, v in throughputs]

    # Calculate baseline from first quarter
    baseline = sum(values[: len(values) // 4]) / (len(values) // 4)

    if baseline <= 0:
        return []

    # Check for sustained drops in later parts
    recent = values[-10:]
    recent_mean = sum(recent) / len(recent)

    drop_ratio = 1 - (recent_mean / baseline)

    if drop_ratio > config.throughput_drop_threshold:
        return [
            Anomaly(
                type="system",
                severity="warning",
                message=f"Throughput drop: {drop_ratio * 100:.0f}% below baseline",
                step=throughputs[-1][0],
                details={
                    "baseline": baseline,
                    "current": recent_mean,
                    "drop_pct": drop_ratio * 100,
                },
            )
        ]

    return []


def format_anomalies(anomalies: list[Anomaly], compact: bool = True) -> str:
    """
    Format anomalies for output.

    Returns empty string if no anomalies (zero tokens for healthy runs).
    """
    if not anomalies:
        return ""

    if compact:
        lines = ["ANOMALIES:"]
        for a in anomalies:
            lines.append(f"  {a.compact()}")
        return "\n".join(lines)
    else:
        lines = ["ANOMALIES DETECTED:"]
        for a in anomalies:
            lines.append(f"  {a}")
        return "\n".join(lines)
