"""
Base class for STL monitors.

Provides abstract interface for all STL monitoring implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..core.specification import STLSpecification


class BaseSTLMonitor(ABC):
    """
    Abstract base class for STL monitors.

    All monitor implementations (offline, online, etc.) should inherit
    from this class and implement the evaluate() method.
    """

    def __init__(self, specifications: List[STLSpecification]):
        """
        Initialize the monitor with STL specifications.

        Args:
            specifications: List of STL specifications to monitor
        """
        assert specifications, "At least one specification must be provided"
        assert all(isinstance(spec, STLSpecification) for spec in specifications), \
            "All specifications must be STLSpecification instances"

        self.specifications = specifications
        self.results = None
        self._evaluated = False

    def __str__(self):
        return (f"{self.__class__.__name__}"
                f"(num_specs={len(self.specifications)})")

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def evaluate(self, input_data: Any) -> List[Dict]:
        """
        Evaluate all specifications against input data.

        Args:
            input_data: Input data (format depends on monitor type)

        Returns:
            List of evaluation results, one per specification
        """
        pass

    def get_violations(self) -> List[Dict]:
        """
        Get list of violated specifications.

        Returns:
            List of results where robustness < 0 (violated)

        Raises:
            RuntimeError: If evaluate() hasn't been called yet
        """
        if not self._evaluated:
            raise RuntimeError("Call evaluate() before getting violations")

        return [
            result for result in self.results
            if result['robustness'] < 0
        ]

    def get_satisfied(self) -> List[Dict]:
        """
        Get list of satisfied specifications.

        Returns:
            List of results where robustness >= 0 (satisfied)

        Raises:
            RuntimeError: If evaluate() hasn't been called yet
        """
        if not self._evaluated:
            raise RuntimeError("Call evaluate() before getting satisfied specs")

        return [
            result for result in self.results
            if result['robustness'] >= 0
        ]

    def all_satisfied(self) -> bool:
        """
        Check if all specifications are satisfied.

        Returns:
            True if all specifications have robustness >= 0

        Raises:
            RuntimeError: If evaluate() hasn't been called yet
        """
        if not self._evaluated:
            raise RuntimeError("Call evaluate() before checking satisfaction")

        return all(result['satisfied'] for result in self.results)

    def any_violated(self) -> bool:
        """
        Check if any specification is violated.

        Returns:
            True if at least one specification has robustness < 0

        Raises:
            RuntimeError: If evaluate() hasn't been called yet
        """
        if not self._evaluated:
            raise RuntimeError("Call evaluate() before checking violations")

        return any(not result['satisfied'] for result in self.results)

    def get_min_robustness(self) -> float:
        """
        Get minimum robustness across all specifications.

        The minimum robustness represents the "weakest link" - the constraint
        that is closest to being violated (or most violated).

        Returns:
            Minimum robustness value

        Raises:
            RuntimeError: If evaluate() hasn't been called yet
        """
        if not self._evaluated:
            raise RuntimeError("Call evaluate() before getting robustness")

        return min(result['robustness'] for result in self.results)

    def get_max_robustness(self) -> float:
        """
        Get maximum robustness across all specifications.

        Returns:
            Maximum robustness value

        Raises:
            RuntimeError: If evaluate() hasn't been called yet
        """
        if not self._evaluated:
            raise RuntimeError("Call evaluate() before getting robustness")

        return max(result['robustness'] for result in self.results)

    def get_avg_robustness(self) -> float:
        """
        Get average robustness across all specifications.

        Returns:
            Average robustness value

        Raises:
            RuntimeError: If evaluate() hasn't been called yet
        """
        if not self._evaluated:
            raise RuntimeError("Call evaluate() before getting robustness")

        robustness_values = [result['robustness'] for result in self.results]
        return sum(robustness_values) / len(robustness_values)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of evaluation results.

        Returns:
            Dictionary with summary statistics

        Raises:
            RuntimeError: If evaluate() hasn't been called yet
        """
        if not self._evaluated:
            raise RuntimeError("Call evaluate() before getting summary")

        return {
            'total_specifications': len(self.specifications),
            'satisfied': len(self.get_satisfied()),
            'violated': len(self.get_violations()),
            'min_robustness': self.get_min_robustness(),
            'max_robustness': self.get_max_robustness(),
            'avg_robustness': self.get_avg_robustness(),
            'all_satisfied': self.all_satisfied(),
        }

    def print_summary(self):
        """Print a human-readable summary of evaluation results."""
        if not self._evaluated:
            print("Monitor not yet evaluated. Call evaluate() first.")
            return

        summary = self.get_summary()

        print("=" * 60)
        print(f"STL Monitor Summary - {self.__class__.__name__}")
        print("=" * 60)
        print(f"Total specifications: {summary['total_specifications']}")
        print(f"Satisfied: {summary['satisfied']}")
        print(f"Violated: {summary['violated']}")
        print(f"All satisfied: {summary['all_satisfied']}")
        print(f"\nRobustness Statistics:")
        print(f"  Min: {summary['min_robustness']:.6f}")
        print(f"  Max: {summary['max_robustness']:.6f}")
        print(f"  Avg: {summary['avg_robustness']:.6f}")
        print("=" * 60)

        if summary['violated'] > 0:
            print("\nViolated Specifications:")
            for result in self.get_violations():
                print(f"  - {result['name']}: robustness = {result['robustness']:.6f}")
                print(f"    Formula: {result['specification']}")

    def reset(self):
        """Reset the monitor state."""
        self.results = None
        self._evaluated = False
