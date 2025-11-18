"""
Composite STL constraints combining multiple metrics.

Provides factory methods for creating complex multi-objective constraints.
"""

from ..core.specification import STLSpecification
from typing import Optional, List


class CompositeConstraints:
    """
    Factory class for composite (multi-objective) STL constraints.

    These constraints combine multiple metrics into a single specification.
    """

    @staticmethod
    def pareto_optimal(
        max_latency: float,
        max_energy: float,
        max_area: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Design must satisfy Pareto frontier constraints.

        Formula: G((latency < max_lat) and (energy < max_eng) and (area < max_area))

        Args:
            max_latency: Maximum latency in seconds
            max_energy: Maximum energy in picojoules
            max_area: Maximum area in mm²
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = (f"always((latency < {max_latency}) and "
                   f"(energy < {max_energy}) and "
                   f"(area < {max_area}))")
        spec_name = name if name else "pareto_optimal_constraint"

        return STLSpecification(
            formula=formula,
            signal_names=['latency', 'energy', 'area'],
            name=spec_name
        )

    @staticmethod
    def performance_power_trade_off(
        max_latency: float,
        max_edp: float,
        min_utilization: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Balance performance (latency, utilization) and power (EDP).

        Formula: G((latency < max_lat) and (edp_latency < max_edp) and (avg_utilization > min_util))

        Args:
            max_latency: Maximum latency
            max_edp: Maximum energy-delay product
            min_utilization: Minimum utilization
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = (f"always((latency < {max_latency}) and "
                   f"(edp_latency < {max_edp}) and "
                   f"(avg_utilization > {min_utilization}))")
        spec_name = name if name else "perf_power_tradeoff"

        return STLSpecification(
            formula=formula,
            signal_names=['latency', 'edp_latency', 'avg_utilization'],
            name=spec_name
        )

    @staticmethod
    def memory_hierarchy_efficiency(
        memory_name: str,
        min_hit_rate: float,
        max_dram_accesses: int,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Memory hierarchy must be efficient (high hit rate, low DRAM accesses).

        Formula: G((memory_hit_rate > threshold) and (dram_accesses < max))

        Args:
            memory_name: Name of the cache memory
            min_hit_rate: Minimum cache hit rate (0.0 to 1.0)
            max_dram_accesses: Maximum DRAM accesses
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        hit_rate_signal = f"{memory_name}_hit_rate"
        formula = (f"always(({hit_rate_signal} > {min_hit_rate}) and "
                   f"(offchip_mem_1_accesses < {max_dram_accesses}))")
        spec_name = name if name else f"{memory_name}_hierarchy_efficiency"

        return STLSpecification(
            formula=formula,
            signal_names=[hit_rate_signal, 'offchip_mem_1_accesses'],
            name=spec_name
        )

    @staticmethod
    def real_time_constraint(
        max_latency: float,
        deadline_cycles: int,
        min_throughput: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Real-time system constraint with deadline and throughput requirements.

        Formula: F[0:deadline](latency < max_latency) and G(avg_throughput > min_throughput)

        Args:
            max_latency: Maximum latency threshold
            deadline_cycles: Deadline in cycles
            min_throughput: Minimum throughput requirement
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = (f"(eventually[0:{deadline_cycles}](latency < {max_latency})) and "
                   f"(always(avg_throughput > {min_throughput}))")
        spec_name = name if name else f"realtime_constraint_{deadline_cycles}cyc"

        return STLSpecification(
            formula=formula,
            signal_names=['latency', 'avg_throughput'],
            time_bounds=(0, deadline_cycles),
            name=spec_name
        )

    @staticmethod
    def custom_and(
        specs: List[STLSpecification],
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Combine multiple specifications with AND operator.

        Formula: spec1 and spec2 and ... and specN

        Args:
            specs: List of STL specifications to combine
            name: Optional custom name

        Returns:
            Combined STLSpecification instance
        """
        assert len(specs) >= 2, "At least 2 specifications required for AND"

        formulas = [f"({spec.formula})" for spec in specs]
        combined_formula = " and ".join(formulas)

        all_signals = []
        for spec in specs:
            all_signals.extend(spec.signal_names)
        all_signals = list(set(all_signals))  # Remove duplicates

        spec_name = name if name else "custom_and_constraint"

        return STLSpecification(
            formula=combined_formula,
            signal_names=all_signals,
            name=spec_name
        )

    @staticmethod
    def custom_or(
        specs: List[STLSpecification],
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Combine multiple specifications with OR operator.

        Formula: spec1 or spec2 or ... or specN

        Args:
            specs: List of STL specifications to combine
            name: Optional custom name

        Returns:
            Combined STLSpecification instance
        """
        assert len(specs) >= 2, "At least 2 specifications required for OR"

        formulas = [f"({spec.formula})" for spec in specs]
        combined_formula = " or ".join(formulas)

        all_signals = []
        for spec in specs:
            all_signals.extend(spec.signal_names)
        all_signals = list(set(all_signals))  # Remove duplicates

        spec_name = name if name else "custom_or_constraint"

        return STLSpecification(
            formula=combined_formula,
            signal_names=all_signals,
            name=spec_name
        )
