"""
Pre-defined STL constraints for power and energy metrics.
"""

from ..core.specification import STLSpecification
from typing import Optional


class PowerConstraints:
    """
    Factory class for power and energy-related STL constraints.
    """

    @staticmethod
    def max_energy(
        threshold_joules: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Total energy consumption must always be below threshold.

        Formula: G(energy < threshold)

        Args:
            threshold_joules: Maximum allowed energy in picojoules
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = f"always(energy < {threshold_joules})"
        spec_name = name if name else f"max_energy_{threshold_joules}pJ"

        return STLSpecification(
            formula=formula,
            signal_names=['energy'],
            name=spec_name
        )

    @staticmethod
    def max_edp_latency(
        threshold: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Energy-Delay Product (with latency) must be below threshold.

        Formula: G(edp_latency < threshold)

        Args:
            threshold: Maximum EDP threshold
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = f"always(edp_latency < {threshold})"
        spec_name = name if name else f"max_edp_latency_{threshold}"

        return STLSpecification(
            formula=formula,
            signal_names=['edp_latency'],
            name=spec_name
        )

    @staticmethod
    def max_edp_cycles(
        threshold: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Energy-Delay Product (with cycles) must be below threshold.

        Formula: G(edp_cycles < threshold)

        Args:
            threshold: Maximum EDP threshold
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = f"always(edp_cycles < {threshold})"
        spec_name = name if name else f"max_edp_cycles_{threshold}"

        return STLSpecification(
            formula=formula,
            signal_names=['edp_cycles'],
            name=spec_name
        )

    @staticmethod
    def energy_efficiency(
        min_energy: float,
        max_energy: float,
        name: Optional[str] = None
    ) -> STLSpecification:
        """
        Energy consumption must be within specified range.

        Formula: G((energy > min_energy) and (energy < max_energy))

        Args:
            min_energy: Minimum energy threshold
            max_energy: Maximum energy threshold
            name: Optional custom name

        Returns:
            STLSpecification instance
        """
        formula = f"always((energy > {min_energy}) and (energy < {max_energy}))"
        spec_name = name if name else f"energy_range_{min_energy}_to_{max_energy}"

        return STLSpecification(
            formula=formula,
            signal_names=['energy'],
            name=spec_name
        )
