"""
Signal builder for creating custom time-series signals.

Provides utilities for building and manipulating time-series signals
for STL monitoring.
"""

from typing import List, Tuple, Callable
import math


class SignalBuilder:
    """
    Utility class for building and manipulating time-series signals.
    """

    @staticmethod
    def constant_signal(
        value: float,
        duration: int,
        start_time: int = 0
    ) -> List[Tuple[float, float]]:
        """
        Create a constant signal.

        Args:
            value: Constant value
            duration: Signal duration
            start_time: Start time (default: 0)

        Returns:
            List of (time, value) tuples
        """
        return [(float(t), value) for t in range(start_time, start_time + duration)]

    @staticmethod
    def step_signal(
        initial_value: float,
        final_value: float,
        step_time: int,
        duration: int
    ) -> List[Tuple[float, float]]:
        """
        Create a step signal (changes value at step_time).

        Args:
            initial_value: Value before step
            final_value: Value after step
            step_time: Time of step change
            duration: Total duration

        Returns:
            List of (time, value) tuples
        """
        signal = []
        for t in range(duration):
            value = initial_value if t < step_time else final_value
            signal.append((float(t), value))
        return signal

    @staticmethod
    def linear_signal(
        start_value: float,
        end_value: float,
        duration: int
    ) -> List[Tuple[float, float]]:
        """
        Create a linearly increasing/decreasing signal.

        Args:
            start_value: Initial value
            end_value: Final value
            duration: Signal duration

        Returns:
            List of (time, value) tuples
        """
        if duration <= 1:
            return [(0.0, start_value)]

        slope = (end_value - start_value) / (duration - 1)
        return [
            (float(t), start_value + slope * t)
            for t in range(duration)
        ]

    @staticmethod
    def combine_signals(
        signal1: List[Tuple[float, float]],
        signal2: List[Tuple[float, float]],
        operation: Callable[[float, float], float]
    ) -> List[Tuple[float, float]]:
        """
        Combine two signals using a binary operation.

        Args:
            signal1: First signal
            signal2: Second signal
            operation: Binary operation (e.g., lambda x, y: x + y)

        Returns:
            Combined signal

        Raises:
            ValueError: If signals have different lengths
        """
        if len(signal1) != len(signal2):
            raise ValueError("Signals must have the same length")

        return [
            (t1, operation(v1, v2))
            for (t1, v1), (t2, v2) in zip(signal1, signal2)
        ]

    @staticmethod
    def apply_function(
        signal: List[Tuple[float, float]],
        func: Callable[[float], float]
    ) -> List[Tuple[float, float]]:
        """
        Apply a unary function to signal values.

        Args:
            signal: Input signal
            func: Unary function to apply

        Returns:
            Transformed signal
        """
        return [(t, func(v)) for t, v in signal]

    @staticmethod
    def resample_signal(
        signal: List[Tuple[float, float]],
        new_duration: int
    ) -> List[Tuple[float, float]]:
        """
        Resample signal to a different duration (simple nearest-neighbor).

        Args:
            signal: Input signal
            new_duration: Desired duration

        Returns:
            Resampled signal
        """
        if not signal:
            return []

        old_duration = len(signal)
        if old_duration == new_duration:
            return signal

        scale = old_duration / new_duration

        resampled = []
        for t in range(new_duration):
            old_index = int(t * scale)
            old_index = min(old_index, old_duration - 1)
            resampled.append((float(t), signal[old_index][1]))

        return resampled

    @staticmethod
    def normalize_signal(
        signal: List[Tuple[float, float]],
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> List[Tuple[float, float]]:
        """
        Normalize signal values to [min_val, max_val] range.

        Args:
            signal: Input signal
            min_val: Minimum value after normalization
            max_val: Maximum value after normalization

        Returns:
            Normalized signal
        """
        if not signal:
            return []

        values = [v for _, v in signal]
        signal_min = min(values)
        signal_max = max(values)

        if signal_max == signal_min:
            # Constant signal
            mid_val = (min_val + max_val) / 2
            return [(t, mid_val) for t, _ in signal]

        # Normalize to [min_val, max_val]
        normalized = []
        for t, v in signal:
            norm_v = (v - signal_min) / (signal_max - signal_min)
            norm_v = min_val + norm_v * (max_val - min_val)
            normalized.append((t, norm_v))

        return normalized

    @staticmethod
    def moving_average(
        signal: List[Tuple[float, float]],
        window_size: int
    ) -> List[Tuple[float, float]]:
        """
        Apply moving average filter to signal.

        Args:
            signal: Input signal
            window_size: Size of averaging window

        Returns:
            Smoothed signal
        """
        if window_size <= 1 or not signal:
            return signal

        smoothed = []
        for i, (t, _) in enumerate(signal):
            # Compute average over window
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(signal), i + window_size // 2 + 1)

            window_values = [signal[j][1] for j in range(start_idx, end_idx)]
            avg_value = sum(window_values) / len(window_values)

            smoothed.append((t, avg_value))

        return smoothed
