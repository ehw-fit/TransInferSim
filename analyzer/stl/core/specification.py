"""
STL Specification class for defining and evaluating Signal Temporal Logic formulas.
"""

from typing import Dict, List, Tuple, Union, Optional
import warnings


class STLSpecification:
    """
    Represents a Signal Temporal Logic (STL) temporal formula.

    STL allows expressing temporal properties over continuous and discrete signals,
    with quantitative robustness semantics that indicate "how much" a specification
    is satisfied or violated.

    Supported operators:
    - G (Globally/Always): Property holds at all times in interval
    - F (Finally/Eventually): Property holds at some time in interval
    - U (Until): First property holds until second becomes true
    - & (And): Both properties hold
    - | (Or): At least one property holds
    - -> (Implies): If first holds, then second must hold

    Examples:
        # Latency always below 10ms
        spec = STLSpecification(
            formula="always(latency < 0.01)",
            signal_names=['latency']
        )

        # Utilization eventually exceeds 70%
        spec = STLSpecification(
            formula="eventually(utilization > 0.7)",
            signal_names=['utilization']
        )
    """

    def __init__(
        self,
        formula: str,
        signal_names: List[str],
        time_bounds: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize an STL specification.

        Args:
            formula: STL formula as a string (e.g., "always(latency < 0.01)")
            signal_names: List of signal variable names used in the formula
            time_bounds: Optional (min_time, max_time) bounds for temporal operators
            name: Optional descriptive name for the specification
        """
        self.formula = formula
        self.signal_names = signal_names
        self.time_bounds = time_bounds if time_bounds else (0, float('inf'))
        self.name = name if name else formula

        # Parsed specification (populated when using rtamt)
        self._spec = None
        self._parsed = False

    def __str__(self):
        return f"STLSpec(name='{self.name}', formula='{self.formula}')"

    def __repr__(self):
        return self.__str__()

    def parse(self):
        """
        Parse the STL formula using rtamt library.

        Returns:
            bool: True if parsing succeeded, False otherwise
        """
        try:
            import rtamt

            # Create discrete-time specification
            spec = rtamt.STLDiscreteTimeSpecification()

            # Declare all signal variables
            for signal_name in self.signal_names:
                spec.declare_var(signal_name, 'float')

            # Set the specification formula
            spec.spec = self.formula

            # Parse the specification
            spec.parse()

            self._spec = spec
            self._parsed = True
            return True

        except ImportError:
            warnings.warn(
                "rtamt not installed. STL parsing unavailable. "
                "Install with: pip install rtamt"
            )
            return False
        except Exception as e:
            warnings.warn(f"Failed to parse STL formula '{self.formula}': {e}")
            return False

    def evaluate(
        self,
        signals: Dict[str, List[Tuple[float, float]]]
    ) -> float:
        """
        Evaluate the STL specification against provided signals.

        Args:
            signals: Dictionary mapping signal names to time-series data
                    Each signal is a list of (time, value) tuples

        Returns:
            float: Robustness degree
                   > 0: Specification is satisfied (larger = more robust)
                   < 0: Specification is violated (more negative = worse violation)
                   = 0: Borderline case

        Raises:
            ValueError: If required signals are missing or rtamt not available
        """
        # Check that all required signals are provided
        for signal_name in self.signal_names:
            if signal_name not in signals:
                raise ValueError(
                    f"Required signal '{signal_name}' not provided. "
                    f"Available signals: {list(signals.keys())}"
                )

        # Parse if not already done
        if not self._parsed:
            success = self.parse()
            if not success:
                raise RuntimeError(
                    "Cannot evaluate STL specification: parsing failed. "
                    "Ensure rtamt is installed."
                )

        # Evaluate using rtamt
        try:
            # Build argument list: alternating signal names and data
            eval_args = []
            for signal_name in self.signal_names:
                eval_args.append(signal_name)
                eval_args.append(signals[signal_name])

            robustness = self._spec.evaluate(*eval_args)

            return robustness

        except Exception as e:
            raise RuntimeError(f"STL evaluation failed: {e}")

    def evaluate_simple(
        self,
        signals: Dict[str, List[Tuple[float, float]]]
    ) -> Dict[str, Union[float, bool]]:
        """
        Simplified evaluation that returns both robustness and satisfaction.

        Args:
            signals: Dictionary mapping signal names to time-series data

        Returns:
            dict: {
                'robustness': float,
                'satisfied': bool,
                'specification': str
            }
        """
        robustness = self.evaluate(signals)

        return {
            'robustness': robustness,
            'satisfied': robustness >= 0,
            'specification': self.formula,
            'name': self.name
        }

    @staticmethod
    def from_simple_predicate(
        signal_name: str,
        operator: str,
        threshold: float,
        temporal_op: str = 'always',
        time_interval: Optional[Tuple[int, int]] = None
    ):
        """
        Create a simple STL specification from basic components.

        Args:
            signal_name: Name of the signal to monitor
            operator: Comparison operator ('<', '<=', '>', '>=', '==', '!=')
            threshold: Threshold value for comparison
            temporal_op: Temporal operator ('always', 'eventually')
            time_interval: Optional (start, end) time interval

        Returns:
            STLSpecification instance

        Example:
            # Create: always(latency < 0.01)
            spec = STLSpecification.from_simple_predicate(
                'latency', '<', 0.01, 'always'
            )
        """
        # Build predicate
        predicate = f"{signal_name} {operator} {threshold}"

        # Add temporal operator
        if time_interval:
            start, end = time_interval
            formula = f"{temporal_op}[{start}:{end}]({predicate})"
        else:
            formula = f"{temporal_op}({predicate})"

        return STLSpecification(
            formula=formula,
            signal_names=[signal_name],
            time_bounds=time_interval,
            name=f"{temporal_op}_{signal_name}_{operator}_{threshold}"
        )
