"""
Pre-defined STL constraints for performance metrics.

Provides convenient factory methods for creating common performance-related
STL specifications.
"""

from ..core.specification import STLSpecification
from typing import Optional, Tuple


class PerformanceConstraints:
    """
    Factory class for performance-related STL constraints.

    All methods return STLSpecification objects that can be used
    with STL monitors.
    """

    @staticmethod
    def max_latency(
        threshold_seconds: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Latency must always be below threshold.

        Formula: G(latency < threshold)

        Args:
            threshold_seconds: Maximum allowed latency in seconds
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = f"always(latency < {threshold_seconds})"
        spec_name = name if name else f"max_latency_{threshold_seconds}s"

        return STLSpecification(
            formula=formula,
            signal_names=['latency'],
            name=spec_name
        )

    @staticmethod
    def min_utilization(
        threshold_percent: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Average utilization must always exceed threshold.

        Formula: G(avg_utilization > threshold)

        Args:
            threshold_percent: Minimum utilization (0.0 to 1.0)
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = f"always(avg_utilization > {threshold_percent})"
        spec_name = name if name else f"min_utilization_{threshold_percent}"

        return STLSpecification(
            formula=formula,
            signal_names=['avg_utilization'],
            name=spec_name
        )

    @staticmethod
    def min_throughput(
        threshold_flops: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Average throughput must always exceed threshold.

        Formula: G(avg_throughput > threshold)

        Args:
            threshold_flops: Minimum throughput in FLOPS
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = f"always(avg_throughput > {threshold_flops})"
        spec_name = name if name else f"min_throughput_{threshold_flops}_flops"

        return STLSpecification(
            formula=formula,
            signal_names=['avg_throughput'],
            name=spec_name
        )

    @staticmethod
    def bounded_latency(
        max_latency: float,
        max_cycles: int,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Latency must be below threshold within specified cycles.

        Formula: F[0:max_cycles](latency < max_latency)

        Args:
            max_latency: Maximum latency threshold
            max_cycles: Time bound in cycles
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = f"eventually[0:{max_cycles}](latency < {max_latency})"
        spec_name = name if name else f"bounded_latency_{max_latency}s_in_{max_cycles}cyc"

        return STLSpecification(
            formula=formula,
            signal_names=['latency'],
            time_bounds=(0, max_cycles),
            name=spec_name
        )

    @staticmethod
    def component_utilization(
        component_name: str,
        min_utilization: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Specific component utilization must exceed threshold.

        Formula: G(component_utilization > threshold)

        Args:
            component_name: Name of the compute component
            min_utilization: Minimum utilization (0.0 to 1.0)
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        signal_name = f"{component_name}_utilization"
        formula = f"always({signal_name} > {min_utilization})"
        spec_name = name if name else f"{component_name}_min_util_{min_utilization}"

        return STLSpecification(
            formula=formula,
            signal_names=[signal_name],
            name=spec_name
        )

    @staticmethod
    def component_idle_limit(
        component_name: str,
        max_idle_ratio: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Component idle time ratio must stay below threshold.

        Formula: G(component_idle_ratio < threshold)

        Args:
            component_name: Name of the compute component
            max_idle_ratio: Maximum idle ratio (0.0 to 1.0)
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        signal_name = f"{component_name}_idle_ratio"
        formula = f"always({signal_name} < {max_idle_ratio})"
        spec_name = name if name else f"{component_name}_max_idle_{max_idle_ratio}"

        return STLSpecification(
            formula=formula,
            signal_names=[signal_name],
            name=spec_name
        )

    @staticmethod
    def balanced_utilization(
        min_utilization: float,
        max_utilization: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Minimum and maximum utilization must stay within bounds.

        Formula: G((min_utilization > min_thresh) and (max_utilization < max_thresh))

        Args:
            min_utilization: Lower bound for minimum utilization
            max_utilization: Upper bound for maximum utilization
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = (f"always((min_utilization > {min_utilization}) and "
                   f"(max_utilization < {max_utilization}))")
        spec_name = name if name else f"balanced_util_{min_utilization}_to_{max_utilization}"

        return STLSpecification(
            formula=formula,
            signal_names=['min_utilization', 'max_utilization'],
            name=spec_name
        )
