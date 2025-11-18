"""
Signal extraction from TransInferSim simulation statistics.

Converts the statistics dictionary from accelerator.get_statistics()
into time-series signals suitable for STL monitoring.
"""

from typing import Dict, List, Tuple, Optional


class SignalExtractor:
    """
    Extracts time-series signals from simulation statistics.

    This class converts the aggregate statistics produced by TransInferSim
    into time-series signals that can be monitored using STL specifications.

    In Phase 1, most signals are treated as constant values over the simulation
    duration (since per-cycle tracking is not yet implemented).

    Future enhancement: Hook into the event scheduler to capture per-cycle values.
    """

    def __init__(self, hw_arch=None):
        """
        Initialize signal extractor.

        Args:
            hw_arch: Optional hardware accelerator instance (for accessing real-time data)
        """
        self.hw_arch = hw_arch

    def extract_signals(
        self,
        stats_dict: Dict
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Extract all available signals from statistics dictionary.

        Args:
            stats_dict: Statistics dictionary from accelerator.get_statistics()

        Returns:
            Dictionary mapping signal names to time-series data [(time, value), ...]
        """
        signals = {}

        # Extract global metrics
        signals.update(self._extract_global_signals(stats_dict))

        # Extract per-component signals
        signals.update(self._extract_compute_signals(stats_dict))
        signals.update(self._extract_memory_signals(stats_dict))

        return signals

    def _extract_global_signals(
        self,
        stats_dict: Dict
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Extract global accelerator-level signals."""
        signals = {}
        duration = stats_dict.get('global_cycles', 1)

        # Global performance metrics (as constant signals)
        signals['latency'] = self._scalar_to_signal(
            stats_dict.get('latency', 0), duration
        )
        signals['energy'] = self._scalar_to_signal(
            stats_dict.get('energy', 0), duration
        )
        signals['edp_latency'] = self._scalar_to_signal(
            stats_dict.get('edp_latency', 0), duration
        )
        signals['edp_cycles'] = self._scalar_to_signal(
            stats_dict.get('edp_cycles', 0), duration
        )
        signals['area'] = self._scalar_to_signal(
            stats_dict.get('area', 0), duration
        )
        signals['avg_throughput'] = self._scalar_to_signal(
            stats_dict.get('avg_throughput', 0), duration
        )
        signals['min_utilization'] = self._scalar_to_signal(
            stats_dict.get('min_utilization', 0), duration
        )
        signals['max_utilization'] = self._scalar_to_signal(
            stats_dict.get('max_utilization', 0), duration
        )

        return signals

    def _extract_compute_signals(
        self,
        stats_dict: Dict
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Extract per-compute-block signals."""
        signals = {}
        duration = stats_dict.get('global_cycles', 1)

        compute_stats = stats_dict.get('compute_stats', [])

        for comp_stat in compute_stats:
            name = comp_stat.get('name', 'unknown')

            # Utilization
            signals[f'{name}_utilization'] = self._scalar_to_signal(
                comp_stat.get('utilization', 0), duration
            )

            # Throughput
            signals[f'{name}_throughput'] = self._scalar_to_signal(
                comp_stat.get('throughput', 0), duration
            )

            # Idle cycles ratio
            idle_cycles = comp_stat.get('idle_cycles', 0)
            computational_cycles = comp_stat.get('computational_cycles', 0)
            total_cycles = idle_cycles + computational_cycles
            idle_ratio = idle_cycles / total_cycles if total_cycles > 0 else 0
            signals[f'{name}_idle_ratio'] = self._scalar_to_signal(
                idle_ratio, duration
            )

        return signals

    def _extract_memory_signals(
        self,
        stats_dict: Dict
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Extract per-memory-block signals."""
        signals = {}
        duration = stats_dict.get('global_cycles', 1)

        # On-chip memories
        memory_stats = stats_dict.get('memory_stats', [])
        for mem_stat in memory_stats:
            name = mem_stat.get('name', 'unknown')

            # Hit rate
            signals[f'{name}_hit_rate'] = self._scalar_to_signal(
                mem_stat.get('hit_rate', 0), duration
            )

            # Miss rate
            signals[f'{name}_miss_rate'] = self._scalar_to_signal(
                mem_stat.get('miss_rate', 0), duration
            )

            # Bandwidth utilization (if available)
            if 'bandwidth_per_port' in mem_stat:
                signals[f'{name}_bandwidth'] = self._scalar_to_signal(
                    mem_stat.get('bandwidth_per_port', 0), duration
                )

        # DRAM statistics
        dram_stats = stats_dict.get('dram_stats', {})
        if dram_stats:
            dram_name = dram_stats.get('name', 'dram')

            signals[f'{dram_name}_hit_rate'] = self._scalar_to_signal(
                dram_stats.get('hit_rate', 0), duration
            )
            signals[f'{dram_name}_accesses'] = self._scalar_to_signal(
                dram_stats.get('accesses', 0), duration
            )

        return signals

    def _scalar_to_signal(
        self,
        value: float,
        duration: int
    ) -> List[Tuple[float, float]]:
        """
        Convert a scalar value to a constant time-series signal.

        Args:
            value: Scalar value
            duration: Signal duration (number of time steps)

        Returns:
            List of (time, value) tuples representing constant signal
        """
        # Create constant signal over duration
        # Sample at regular intervals (every cycle for now)
        return [(float(t), value) for t in range(int(duration))]

    def extract_signal_subset(
        self,
        stats_dict: Dict,
        signal_names: List[str]
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Extract only specified signals.

        Args:
            stats_dict: Statistics dictionary
            signal_names: List of signal names to extract

        Returns:
            Dictionary containing only requested signals

        Raises:
            ValueError: If a requested signal is not available
        """
        all_signals = self.extract_signals(stats_dict)

        subset = {}
        for name in signal_names:
            if name not in all_signals:
                raise ValueError(
                    f"Signal '{name}' not available. "
                    f"Available signals: {list(all_signals.keys())}"
                )
            subset[name] = all_signals[name]

        return subset

    def get_available_signals(self, stats_dict: Dict) -> List[str]:
        """
        Get list of all available signal names from statistics.

        Args:
            stats_dict: Statistics dictionary

        Returns:
            List of signal names
        """
        signals = self.extract_signals(stats_dict)
        return list(signals.keys())

    def signal_statistics(
        self,
        signal: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Compute basic statistics for a signal.

        Args:
            signal: Time-series signal [(time, value), ...]

        Returns:
            Dictionary with min, max, mean, std statistics
        """
        if not signal:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}

        values = [v for _, v in signal]

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5

        return {
            'min': min(values),
            'max': max(values),
            'mean': mean,
            'std': std
        }
