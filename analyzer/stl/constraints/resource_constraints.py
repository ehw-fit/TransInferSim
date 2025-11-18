"""
Pre-defined STL constraints for resource utilization (memory, bandwidth, etc.).
"""

from ..core.specification import STLSpecification
from typing import Optional


class ResourceConstraints:
    """
    Factory class for resource utilization STL constraints.
    """

    @staticmethod
    def max_area(
        threshold_mm2: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Total chip area must be below threshold.

        Formula: G(area < threshold)

        Args:
            threshold_mm2: Maximum area in mm²
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = f"always(area < {threshold_mm2})"
        spec_name = name if name else f"max_area_{threshold_mm2}mm2"

        return STLSpecification(
            formula=formula,
            signal_names=['area'],
            name=spec_name
        )

    @staticmethod
    def min_cache_hit_rate(
        memory_name: str,
        threshold: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Cache hit rate must exceed threshold.

        Formula: G(memory_hit_rate > threshold)

        Args:
            memory_name: Name of the memory component
            threshold: Minimum hit rate (0.0 to 1.0)
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        signal_name = f"{memory_name}_hit_rate"
        formula = f"always({signal_name} > {threshold})"
        spec_name = name if name else f"{memory_name}_min_hit_rate_{threshold}"

        return STLSpecification(
            formula=formula,
            signal_names=[signal_name],
            name=spec_name
        )

    @staticmethod
    def max_cache_miss_rate(
        memory_name: str,
        threshold: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Cache miss rate must stay below threshold.

        Formula: G(memory_miss_rate < threshold)

        Args:
            memory_name: Name of the memory component
            threshold: Maximum miss rate (0.0 to 1.0)
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        signal_name = f"{memory_name}_miss_rate"
        formula = f"always({signal_name} < {threshold})"
        spec_name = name if name else f"{memory_name}_max_miss_rate_{threshold}"

        return STLSpecification(
            formula=formula,
            signal_names=[signal_name],
            name=spec_name
        )

    @staticmethod
    def min_memory_bandwidth(
        memory_name: str,
        threshold_gbps: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Memory bandwidth must exceed threshold.

        Formula: G(memory_bandwidth > threshold)

        Args:
            memory_name: Name of the memory component
            threshold_gbps: Minimum bandwidth in Gbps
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        signal_name = f"{memory_name}_bandwidth"
        formula = f"always({signal_name} > {threshold_gbps})"
        spec_name = name if name else f"{memory_name}_min_bw_{threshold_gbps}gbps"

        return STLSpecification(
            formula=formula,
            signal_names=[signal_name],
            name=spec_name
        )

    @staticmethod
    def dram_access_limit(
        max_accesses: int,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        DRAM accesses must stay below threshold (to minimize power).

        Formula: G(dram_accesses < threshold)

        Args:
            max_accesses: Maximum number of DRAM accesses
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = f"always(offchip_mem_1_accesses < {max_accesses})"
        spec_name = name if name else f"max_dram_accesses_{max_accesses}"

        return STLSpecification(
            formula=formula,
            signal_names=['offchip_mem_1_accesses'],
            name=spec_name
        )
