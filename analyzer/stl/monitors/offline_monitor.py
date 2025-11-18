"""
Offline STL monitor for post-simulation constraint checking.

Evaluates STL specifications after simulation completes, using
the statistics dictionary from accelerator.get_statistics().
"""

from typing import Dict, List
from .base_monitor import BaseSTLMonitor
from ..core.specification import STLSpecification
from ..signals.signal_extractor import SignalExtractor


class OfflineSTLMonitor(BaseSTLMonitor):
    """
    Post-simulation STL monitoring.

    Evaluates STL constraints after a simulation run completes, using
    the aggregate statistics dictionary.

    This is the simplest form of monitoring and doesn't require any
    changes to the simulation engine.

    Example:
        # Define constraints
        constraints = [
            PerformanceConstraints.max_latency(10e-3),
            PerformanceConstraints.min_utilization(0.7)
        ]

        # Create monitor
        monitor = OfflineSTLMonitor(constraints, hw_arch=accelerator)

        # Run simulation
        analyzer.run_simulation_analysis()
        stats = accelerator.get_statistics()

        # Evaluate constraints
        results = monitor.evaluate(stats)
    """

    def __init__(
        self,
        specifications: List[STLSpecification],
        hw_arch=None
    ):
        """
        Initialize offline STL monitor.

        Args:
            specifications: List of STL specifications to monitor
            hw_arch: Optional hardware architecture (for signal extraction)
        """
        super().__init__(specifications)
        self.hw_arch = hw_arch
        self.extractor = SignalExtractor(hw_arch)

    def evaluate(self, stats_dict: Dict) -> List[Dict]:
        """
        Evaluate all STL specifications on completed simulation.

        Args:
            stats_dict: Statistics dictionary from accelerator.get_statistics()

        Returns:
            List of evaluation results, one per specification.
            Each result contains:
            - specification: Formula string
            - name: Specification name
            - robustness: Robustness degree (>= 0: satisfied, < 0: violated)
            - satisfied: Boolean satisfaction flag
            - signals_used: List of signal names used

        Raises:
            ValueError: If required signals are not available
        """
        # Extract all signals from statistics
        all_signals = self.extractor.extract_signals(stats_dict)

        # Evaluate each specification
        results = []
        for spec in self.specifications:
            try:
                # Get required signals for this specification
                required_signals = {
                    name: all_signals[name]
                    for name in spec.signal_names
                }

                # Evaluate specification
                robustness = spec.evaluate(required_signals)

                # Store result
                result = {
                    'specification': spec.formula,
                    'name': spec.name,
                    'robustness': robustness,
                    'satisfied': robustness >= 0,
                    'signals_used': spec.signal_names,
                    'spec_object': spec
                }

                results.append(result)

            except KeyError as e:
                raise ValueError(
                    f"Required signal not available for specification '{spec.name}': {e}\n"
                    f"Available signals: {list(all_signals.keys())}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to evaluate specification '{spec.name}': {e}"
                )

        # Store results and mark as evaluated
        self.results = results
        self._evaluated = True

        return results

    def evaluate_and_report(self, stats_dict: Dict) -> Dict:
        """
        Evaluate specifications and return a comprehensive report.

        Args:
            stats_dict: Statistics dictionary from accelerator.get_statistics()

        Returns:
            Dictionary containing:
            - results: List of individual specification results
            - summary: Summary statistics
            - all_satisfied: Boolean flag
        """
        results = self.evaluate(stats_dict)
        summary = self.get_summary()

        return {
            'results': results,
            'summary': summary,
            'all_satisfied': summary['all_satisfied'],
            'violations': self.get_violations(),
            'satisfied': self.get_satisfied()
        }

    def get_available_signals(self, stats_dict: Dict) -> List[str]:
        """
        Get list of available signals from statistics.

        Useful for debugging and understanding what signals can be monitored.

        Args:
            stats_dict: Statistics dictionary

        Returns:
            List of available signal names
        """
        return self.extractor.get_available_signals(stats_dict)

    def validate_specifications(self, stats_dict: Dict) -> Dict[str, bool]:
        """
        Check if all required signals are available for each specification.

        Args:
            stats_dict: Statistics dictionary

        Returns:
            Dictionary mapping specification names to availability flags
        """
        available_signals = set(self.extractor.get_available_signals(stats_dict))

        validation = {}
        for spec in self.specifications:
            required_signals = set(spec.signal_names)
            is_valid = required_signals.issubset(available_signals)
            validation[spec.name] = {
                'valid': is_valid,
                'required_signals': list(required_signals),
                'missing_signals': list(required_signals - available_signals)
            }

        return validation
