"""
STL-guided Design Space Exploration.

Automatically evaluates and ranks hardware configurations based on
STL constraint satisfaction and robustness.
"""

from typing import List, Dict, Callable, Optional, Any
from ..monitors.offline_monitor import OfflineSTLMonitor
from ..core.specification import STLSpecification


class ConstraintBasedDSE:
    """
    STL-guided design space exploration.

    Evaluates multiple hardware configurations against STL constraints,
    filtering and ranking designs by robustness.

    Example:
        # Define constraints
        constraints = [
            PerformanceConstraints.max_latency(10e-3),
            PowerConstraints.max_energy(100e-9)
        ]

        # Create DSE engine
        dse = ConstraintBasedDSE(model=ViTTiny(), constraints=constraints)

        # Generate configurations
        configs = [
            create_accelerator(size=64, mem=16*1024*1024),
            create_accelerator(size=128, mem=32*1024*1024),
            # ...
        ]

        # Explore and rank
        results = dse.explore_design_space(configs)

        # Get best config
        best = results[0]
    """

    def __init__(
        self,
        model,
        constraints: List[STLSpecification],
        data_bitwidth: int = 8
    ):
        """
        Initialize DSE engine.

        Args:
            model: Transformer model to analyze
            constraints: List of STL specifications to enforce
            data_bitwidth: Data bitwidth for analysis (default: 8)
        """
        self.model = model
        self.constraints = constraints
        self.data_bitwidth = data_bitwidth
        self.results_cache = {}

    def explore_design_space(
        self,
        hw_configs: List,
        analyzer_factory: Optional[Callable] = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Evaluate multiple hardware configurations.

        Args:
            hw_configs: List of GenericAccelerator instances
            analyzer_factory: Optional custom analyzer factory function
            verbose: Print progress information

        Returns:
            List of results sorted by robustness (best first)
            Each result contains:
            - config: Hardware configuration
            - stats: Simulation statistics
            - stl_results: STL evaluation results
            - min_robustness: Minimum robustness across all constraints
            - satisfies_all: Boolean flag
        """
        from analyzer.analyzer import Analyzer

        results = []

        for i, config in enumerate(hw_configs):
            if verbose:
                print(f"Evaluating configuration {i+1}/{len(hw_configs)}: {config.name}")

            try:
                # Create analyzer
                if analyzer_factory:
                    analyzer = analyzer_factory(self.model, config, self.data_bitwidth)
                else:
                    analyzer = Analyzer(self.model, config, data_bitwidth=self.data_bitwidth)

                # Run simulation
                analyzer.run_simulation_analysis(verbose=False)
                stats = config.get_statistics()

                # Evaluate STL constraints
                monitor = OfflineSTLMonitor(self.constraints, config)
                stl_results = monitor.evaluate(stats)

                # Compute aggregate robustness
                robustness_values = [r['robustness'] for r in stl_results]
                min_robustness = min(robustness_values) if robustness_values else 0
                avg_robustness = sum(robustness_values) / len(robustness_values) if robustness_values else 0

                result = {
                    'config': config,
                    'config_name': config.name,
                    'stats': stats,
                    'stl_results': stl_results,
                    'min_robustness': min_robustness,
                    'avg_robustness': avg_robustness,
                    'satisfies_all': all(r['satisfied'] for r in stl_results),
                    'num_violations': sum(1 for r in stl_results if not r['satisfied']),
                    'monitor_summary': monitor.get_summary()
                }

                results.append(result)

                # Cache result
                self.results_cache[config.name] = result

                if verbose:
                    print(f"  Min robustness: {min_robustness:.6f}")
                    print(f"  Satisfies all: {result['satisfies_all']}")

            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                results.append({
                    'config': config,
                    'config_name': config.name,
                    'error': str(e),
                    'min_robustness': float('-inf'),
                    'satisfies_all': False
                })

        # Sort by robustness (higher is better)
        results.sort(key=lambda x: x['min_robustness'], reverse=True)

        return results

    def filter_satisfying(
        self,
        results: List[Dict]
    ) -> List[Dict]:
        """
        Filter results to only those satisfying all constraints.

        Args:
            results: List of DSE results

        Returns:
            Filtered list containing only satisfying configurations
        """
        return [r for r in results if r.get('satisfies_all', False)]

    def get_top_k(
        self,
        results: List[Dict],
        k: int,
        metric: str = 'min_robustness'
    ) -> List[Dict]:
        """
        Get top-k configurations by specified metric.

        Args:
            results: List of DSE results
            k: Number of top results to return
            metric: Metric to sort by ('min_robustness', 'avg_robustness', etc.)

        Returns:
            Top-k configurations
        """
        sorted_results = sorted(
            results,
            key=lambda x: x.get(metric, float('-inf')),
            reverse=True
        )
        return sorted_results[:k]

    def compare_configs(
        self,
        config_names: List[str]
    ) -> Dict:
        """
        Compare cached configurations side-by-side.

        Args:
            config_names: List of configuration names to compare

        Returns:
            Comparison dictionary

        Raises:
            ValueError: If configuration not found in cache
        """
        comparison = {
            'configs': config_names,
            'metrics': {}
        }

        for config_name in config_names:
            if config_name not in self.results_cache:
                raise ValueError(f"Configuration '{config_name}' not in cache")

        # Extract common metrics
        metrics_to_compare = [
            'min_robustness',
            'avg_robustness',
            'satisfies_all',
            'num_violations'
        ]

        for metric in metrics_to_compare:
            comparison['metrics'][metric] = {
                config_name: self.results_cache[config_name].get(metric)
                for config_name in config_names
            }

        # Add stats comparison
        stats_metrics = ['latency', 'energy', 'area', 'avg_throughput']
        for metric in stats_metrics:
            comparison['metrics'][metric] = {
                config_name: self.results_cache[config_name]['stats'].get(metric)
                for config_name in config_names
            }

        return comparison

    def export_results(
        self,
        results: List[Dict],
        filename: str,
        format: str = 'txt'
    ):
        """
        Export DSE results to file.

        Args:
            results: List of DSE results
            filename: Output filename
            format: Output format ('txt' or 'csv')
        """
        if format == 'txt':
            self._export_txt(results, filename)
        elif format == 'csv':
            self._export_csv(results, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_txt(self, results: List[Dict], filename: str):
        """Export results as formatted text."""
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STL-Guided Design Space Exploration Results\n")
            f.write("=" * 80 + "\n\n")

            for i, result in enumerate(results):
                f.write(f"Rank {i+1}: {result['config_name']}\n")
                f.write(f"  Min Robustness: {result['min_robustness']:.6f}\n")
                f.write(f"  Avg Robustness: {result['avg_robustness']:.6f}\n")
                f.write(f"  Satisfies All: {result['satisfies_all']}\n")
                f.write(f"  Violations: {result['num_violations']}\n")

                if 'stats' in result:
                    stats = result['stats']
                    f.write(f"  Latency: {stats.get('latency', 'N/A')} s\n")
                    f.write(f"  Energy: {stats.get('energy', 'N/A')} pJ\n")
                    f.write(f"  Area: {stats.get('area', 'N/A')} mm²\n")

                f.write("\n")

    def _export_csv(self, results: List[Dict], filename: str):
        """Export results as CSV."""
        import csv

        with open(filename, 'w', newline='') as f:
            fieldnames = [
                'rank', 'config_name', 'min_robustness', 'avg_robustness',
                'satisfies_all', 'num_violations', 'latency', 'energy', 'area'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for i, result in enumerate(results):
                row = {
                    'rank': i + 1,
                    'config_name': result['config_name'],
                    'min_robustness': result['min_robustness'],
                    'avg_robustness': result['avg_robustness'],
                    'satisfies_all': result['satisfies_all'],
                    'num_violations': result['num_violations'],
                }

                if 'stats' in result:
                    stats = result['stats']
                    row['latency'] = stats.get('latency', '')
                    row['energy'] = stats.get('energy', '')
                    row['area'] = stats.get('area', '')

                writer.writerow(row)
