"""
Ranking and selection utilities based on STL robustness.
"""

from typing import List, Dict, Callable, Optional


class RobustnessRanker:
    """
    Rank and select hardware configurations based on STL robustness metrics.
    """

    @staticmethod
    def rank_by_min_robustness(
        results: List[Dict],
        ascending: bool = False
    ) -> List[Dict]:
        """
        Rank configurations by minimum robustness.

        The minimum robustness represents the "weakest link" - the constraint
        that is closest to being violated.

        Args:
            results: List of DSE results
            ascending: If True, rank from lowest to highest robustness

        Returns:
            Sorted list of results
        """
        return sorted(
            results,
            key=lambda x: x.get('min_robustness', float('-inf')),
            reverse=not ascending
        )

    @staticmethod
    def rank_by_avg_robustness(
        results: List[Dict],
        ascending: bool = False
    ) -> List[Dict]:
        """
        Rank configurations by average robustness.

        Args:
            results: List of DSE results
            ascending: If True, rank from lowest to highest

        Returns:
            Sorted list of results
        """
        return sorted(
            results,
            key=lambda x: x.get('avg_robustness', float('-inf')),
            reverse=not ascending
        )

    @staticmethod
    def rank_by_violation_count(
        results: List[Dict],
        ascending: bool = True
    ) -> List[Dict]:
        """
        Rank configurations by number of violations.

        Args:
            results: List of DSE results
            ascending: If True, rank from fewest to most violations

        Returns:
            Sorted list of results
        """
        return sorted(
            results,
            key=lambda x: x.get('num_violations', float('inf')),
            reverse=not ascending
        )

    @staticmethod
    def rank_by_weighted_robustness(
        results: List[Dict],
        spec_weights: Dict[str, float]
    ) -> List[Dict]:
        """
        Rank configurations by weighted robustness.

        Allows prioritizing certain constraints over others.

        Args:
            results: List of DSE results
            spec_weights: Dictionary mapping specification names to weights

        Returns:
            Sorted list of results (highest weighted robustness first)
        """
        def compute_weighted_robustness(result):
            total_weight = 0.0
            weighted_sum = 0.0

            for stl_result in result.get('stl_results', []):
                spec_name = stl_result['name']
                robustness = stl_result['robustness']
                weight = spec_weights.get(spec_name, 1.0)

                weighted_sum += robustness * weight
                total_weight += weight

            return weighted_sum / total_weight if total_weight > 0 else 0.0

        return sorted(
            results,
            key=compute_weighted_robustness,
            reverse=True
        )

    @staticmethod
    def filter_satisfying_all(
        results: List[Dict]
    ) -> List[Dict]:
        """
        Filter to only configurations satisfying all constraints.

        Args:
            results: List of DSE results

        Returns:
            Filtered list
        """
        return [r for r in results if r.get('satisfies_all', False)]

    @staticmethod
    def filter_by_min_robustness(
        results: List[Dict],
        threshold: float
    ) -> List[Dict]:
        """
        Filter configurations with minimum robustness above threshold.

        Args:
            results: List of DSE results
            threshold: Minimum acceptable robustness

        Returns:
            Filtered list
        """
        return [
            r for r in results
            if r.get('min_robustness', float('-inf')) >= threshold
        ]

    @staticmethod
    def filter_by_max_violations(
        results: List[Dict],
        max_violations: int
    ) -> List[Dict]:
        """
        Filter configurations with at most max_violations violations.

        Args:
            results: List of DSE results
            max_violations: Maximum number of allowed violations

        Returns:
            Filtered list
        """
        return [
            r for r in results
            if r.get('num_violations', float('inf')) <= max_violations
        ]

    @staticmethod
    def select_best_tradeoff(
        results: List[Dict],
        robustness_weight: float = 0.5,
        performance_metric: str = 'latency',
        minimize_performance: bool = True
    ) -> Optional[Dict]:
        """
        Select configuration with best robustness-performance tradeoff.

        Args:
            results: List of DSE results
            robustness_weight: Weight for robustness (0.0 to 1.0)
            performance_metric: Performance metric name (e.g., 'latency', 'energy')
            minimize_performance: If True, lower performance values are better

        Returns:
            Best configuration or None if results empty
        """
        if not results:
            return None

        perf_weight = 1.0 - robustness_weight

        # Normalize metrics
        robustness_values = [r.get('min_robustness', 0) for r in results]
        perf_values = [r['stats'].get(performance_metric, 0) for r in results if 'stats' in r]

        if not perf_values:
            # No performance data, rank by robustness only
            return max(results, key=lambda r: r.get('min_robustness', float('-inf')))

        rob_min, rob_max = min(robustness_values), max(robustness_values)
        perf_min, perf_max = min(perf_values), max(perf_values)

        rob_range = rob_max - rob_min if rob_max != rob_min else 1.0
        perf_range = perf_max - perf_min if perf_max != perf_min else 1.0

        def score(result):
            # Normalize robustness (higher is better)
            rob = result.get('min_robustness', rob_min)
            rob_norm = (rob - rob_min) / rob_range

            # Normalize performance
            if 'stats' not in result:
                return robustness_weight * rob_norm

            perf = result['stats'].get(performance_metric, perf_min)
            if minimize_performance:
                perf_norm = 1.0 - (perf - perf_min) / perf_range  # Invert for minimization
            else:
                perf_norm = (perf - perf_min) / perf_range

            return robustness_weight * rob_norm + perf_weight * perf_norm

        return max(results, key=score)

    @staticmethod
    def get_pareto_robustness_performance(
        results: List[Dict],
        performance_metrics: List[str],
        minimize_metrics: List[bool]
    ) -> List[Dict]:
        """
        Get Pareto-optimal configurations considering robustness and performance.

        Args:
            results: List of DSE results
            performance_metrics: List of performance metric names
            minimize_metrics: List of booleans for each metric

        Returns:
            Pareto-optimal configurations
        """
        from .pareto_frontier import ParetoFrontier

        # Add robustness as an objective (maximize)
        objectives = ['min_robustness'] + performance_metrics
        minimize = [False] + minimize_metrics  # Maximize robustness

        def extract_metrics(result):
            metrics = {'min_robustness': result.get('min_robustness', float('-inf'))}
            if 'stats' in result:
                for metric in performance_metrics:
                    metrics[metric] = result['stats'].get(metric, float('inf'))
            return metrics

        return ParetoFrontier.compute_pareto_frontier(
            results,
            objectives,
            minimize,
            extract_metrics
        )
