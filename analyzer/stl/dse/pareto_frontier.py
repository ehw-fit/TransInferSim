"""
Pareto frontier computation for multi-objective design space exploration.
"""

from typing import List, Dict, Tuple, Callable
import math


class ParetoFrontier:
    """
    Compute Pareto-optimal configurations.

    A configuration is Pareto-optimal if no other configuration is better
    in all objectives simultaneously.
    """

    @staticmethod
    def is_dominated(
        point1: Dict[str, float],
        point2: Dict[str, float],
        objectives: List[str],
        minimize: List[bool]
    ) -> bool:
        """
        Check if point1 is dominated by point2.

        Point1 is dominated by point2 if point2 is better or equal in all
        objectives and strictly better in at least one.

        Args:
            point1: First configuration's metrics
            point2: Second configuration's metrics
            objectives: List of objective names
            minimize: List of booleans indicating minimize (True) or maximize (False)

        Returns:
            True if point1 is dominated by point2
        """
        better_or_equal_count = 0
        strictly_better_count = 0

        for obj, is_minimize in zip(objectives, minimize):
            val1 = point1.get(obj, float('inf') if is_minimize else float('-inf'))
            val2 = point2.get(obj, float('inf') if is_minimize else float('-inf'))

            if is_minimize:
                # For minimization: smaller is better
                if val2 < val1:
                    strictly_better_count += 1
                    better_or_equal_count += 1
                elif val2 == val1:
                    better_or_equal_count += 1
            else:
                # For maximization: larger is better
                if val2 > val1:
                    strictly_better_count += 1
                    better_or_equal_count += 1
                elif val2 == val1:
                    better_or_equal_count += 1

        # Dominated if point2 is better/equal in all and strictly better in at least one
        return (better_or_equal_count == len(objectives) and strictly_better_count > 0)

    @staticmethod
    def compute_pareto_frontier(
        configurations: List[Dict],
        objectives: List[str],
        minimize: List[bool],
        extract_metrics: Callable[[Dict], Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Compute Pareto-optimal set of configurations.

        Args:
            configurations: List of configuration dictionaries
            objectives: List of objective metric names
            minimize: List of booleans (True = minimize, False = maximize) for each objective
            extract_metrics: Optional function to extract metrics dict from configuration

        Returns:
            List of Pareto-optimal configurations

        Example:
            # Find Pareto frontier for latency (minimize) and throughput (maximize)
            pareto_configs = ParetoFrontier.compute_pareto_frontier(
                configurations=results,
                objectives=['latency', 'throughput'],
                minimize=[True, False],
                extract_metrics=lambda r: r['stats']
            )
        """
        assert len(objectives) == len(minimize), \
            "objectives and minimize lists must have same length"

        if not configurations:
            return []

        # Extract metrics from configurations
        if extract_metrics:
            points = [extract_metrics(config) for config in configurations]
        else:
            points = configurations

        pareto_optimal = []

        for i, (config, point) in enumerate(zip(configurations, points)):
            is_pareto = True

            # Check if this point is dominated by any other point
            for j, (_, other_point) in enumerate(zip(configurations, points)):
                if i != j:
                    if ParetoFrontier.is_dominated(point, other_point, objectives, minimize):
                        is_pareto = False
                        break

            if is_pareto:
                pareto_optimal.append(config)

        return pareto_optimal

    @staticmethod
    def rank_by_pareto_layers(
        configurations: List[Dict],
        objectives: List[str],
        minimize: List[bool],
        extract_metrics: Callable[[Dict], Dict[str, float]] = None
    ) -> List[List[Dict]]:
        """
        Rank configurations by Pareto layers.

        First layer = Pareto frontier
        Second layer = Pareto frontier after removing first layer
        And so on...

        Args:
            configurations: List of configuration dictionaries
            objectives: List of objective metric names
            minimize: List of booleans for each objective
            extract_metrics: Optional function to extract metrics

        Returns:
            List of layers, where each layer is a list of configurations
        """
        remaining = list(configurations)
        layers = []

        while remaining:
            # Compute Pareto frontier of remaining configurations
            pareto_layer = ParetoFrontier.compute_pareto_frontier(
                remaining, objectives, minimize, extract_metrics
            )

            if not pareto_layer:
                break

            layers.append(pareto_layer)

            # Remove this layer from remaining
            remaining = [c for c in remaining if c not in pareto_layer]

        return layers

    @staticmethod
    def hypervolume(
        pareto_front: List[Dict],
        objectives: List[str],
        reference_point: Dict[str, float],
        extract_metrics: Callable[[Dict], Dict[str, float]] = None
    ) -> float:
        """
        Compute hypervolume indicator (2D only for simplicity).

        Hypervolume measures the volume of objective space dominated by
        the Pareto front. Larger is better.

        Args:
            pareto_front: List of Pareto-optimal configurations
            objectives: List of 2 objective names (only 2D supported)
            reference_point: Reference point (worst acceptable values)
            extract_metrics: Optional function to extract metrics

        Returns:
            Hypervolume value

        Note:
            Only supports 2-objective optimization for now.
        """
        if len(objectives) != 2:
            raise NotImplementedError("Hypervolume only implemented for 2 objectives")

        if not pareto_front:
            return 0.0

        # Extract metrics
        if extract_metrics:
            points = [extract_metrics(config) for config in pareto_front]
        else:
            points = pareto_front

        # Extract 2D points
        obj1, obj2 = objectives
        ref1, ref2 = reference_point[obj1], reference_point[obj2]

        # Sort points by first objective
        sorted_points = sorted(points, key=lambda p: p[obj1])

        # Compute hypervolume
        hv = 0.0
        prev_x = ref1

        for point in sorted_points:
            x = point[obj1]
            y = point[obj2]

            # Width * height
            width = x - prev_x
            height = ref2 - y

            hv += width * height
            prev_x = x

        return hv

    @staticmethod
    def crowding_distance(
        configurations: List[Dict],
        objectives: List[str],
        extract_metrics: Callable[[Dict], Dict[str, float]] = None
    ) -> Dict[int, float]:
        """
        Compute crowding distance for each configuration.

        Crowding distance measures how close a configuration is to its neighbors.
        Higher distance = more isolated = more diverse.

        Args:
            configurations: List of configurations
            objectives: List of objective names
            extract_metrics: Optional function to extract metrics

        Returns:
            Dictionary mapping configuration index to crowding distance
        """
        n = len(configurations)
        if n <= 2:
            return {i: float('inf') for i in range(n)}

        # Extract metrics
        if extract_metrics:
            points = [extract_metrics(config) for config in configurations]
        else:
            points = configurations

        # Initialize crowding distances
        crowding = {i: 0.0 for i in range(n)}

        # For each objective
        for obj in objectives:
            # Sort by this objective
            sorted_indices = sorted(
                range(n),
                key=lambda i: points[i].get(obj, 0)
            )

            # Boundary points get infinite distance
            crowding[sorted_indices[0]] = float('inf')
            crowding[sorted_indices[-1]] = float('inf')

            # Get objective range
            obj_values = [points[i].get(obj, 0) for i in range(n)]
            obj_range = max(obj_values) - min(obj_values)

            if obj_range == 0:
                continue

            # Compute crowding distance for intermediate points
            for i in range(1, n - 1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]

                distance = (points[next_idx][obj] - points[prev_idx][obj]) / obj_range
                crowding[idx] += distance

        return crowding
