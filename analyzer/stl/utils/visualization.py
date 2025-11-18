"""
Visualization utilities for STL monitoring and analysis.

Provides functions for plotting signals, robustness values, and DSE results.
"""

from typing import List, Tuple, Dict, Optional
import warnings


def plot_signals(
    signals: Dict[str, List[Tuple[float, float]]],
    title: str = "Signal Plot",
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot time-series signals.

    Args:
        signals: Dictionary mapping signal names to [(time, value), ...] data
        title: Plot title
        filename: Optional filename to save plot (if None, displays interactively)
        figsize: Figure size (width, height)

    Note:
        Requires matplotlib. If not installed, prints warning and returns.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed. Cannot plot signals.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    for signal_name, data in signals.items():
        times = [t for t, _ in data]
        values = [v for _, v in data]
        ax.plot(times, values, label=signal_name, marker='o', markersize=2)

    ax.set_xlabel('Time (cycles)')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    else:
        plt.show()

    plt.close()


def plot_robustness(
    results: List[Dict],
    title: str = "STL Robustness Values",
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot robustness values for multiple specifications.

    Args:
        results: List of STL evaluation results
        title: Plot title
        filename: Optional filename to save plot
        figsize: Figure size

    Note:
        Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed. Cannot plot robustness.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    spec_names = [r['name'] for r in results]
    robustness_values = [r['robustness'] for r in results]
    colors = ['green' if r >= 0 else 'red' for r in robustness_values]

    bars = ax.bar(range(len(spec_names)), robustness_values, color=colors, alpha=0.7)

    ax.set_xlabel('Specification')
    ax.set_ylabel('Robustness')
    ax.set_title(title)
    ax.set_xticks(range(len(spec_names)))
    ax.set_xticklabels(spec_names, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, robustness_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=8)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    else:
        plt.show()

    plt.close()


def plot_dse_comparison(
    dse_results: List[Dict],
    x_metric: str = 'latency',
    y_metric: str = 'energy',
    color_by: str = 'min_robustness',
    top_k: Optional[int] = None,
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot DSE results in 2D objective space.

    Args:
        dse_results: List of DSE results
        x_metric: Metric for x-axis (from stats)
        y_metric: Metric for y-axis (from stats)
        color_by: Metric for color coding ('min_robustness' or 'satisfies_all')
        top_k: If specified, only plot top-k configurations
        filename: Optional filename to save plot
        figsize: Figure size

    Note:
        Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
    except ImportError:
        warnings.warn("matplotlib not installed. Cannot plot DSE comparison.")
        return

    # Filter to top-k if specified
    if top_k:
        dse_results = dse_results[:top_k]

    # Extract data
    x_values = [r['stats'].get(x_metric, 0) for r in dse_results if 'stats' in r]
    y_values = [r['stats'].get(y_metric, 0) for r in dse_results if 'stats' in r]
    config_names = [r['config_name'] for r in dse_results if 'stats' in r]

    if color_by == 'min_robustness':
        colors = [r.get('min_robustness', 0) for r in dse_results if 'stats' in r]
        cmap = cm.RdYlGn
        norm = Normalize(vmin=min(colors), vmax=max(colors))
    elif color_by == 'satisfies_all':
        colors = ['green' if r.get('satisfies_all', False) else 'red'
                  for r in dse_results if 'stats' in r]
        cmap = None
        norm = None
    else:
        colors = 'blue'
        cmap = None
        norm = None

    fig, ax = plt.subplots(figsize=figsize)

    if cmap:
        scatter = ax.scatter(x_values, y_values, c=colors, cmap=cmap, norm=norm,
                            s=100, alpha=0.6, edgecolors='black', linewidth=1)
        plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        scatter = ax.scatter(x_values, y_values, c=colors, s=100, alpha=0.6,
                            edgecolors='black', linewidth=1)

    # Annotate points
    for i, name in enumerate(config_names):
        ax.annotate(f"{i+1}", (x_values[i], y_values[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f'Design Space Exploration: {x_metric} vs {y_metric}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    else:
        plt.show()

    plt.close()


def plot_pareto_frontier(
    dse_results: List[Dict],
    pareto_configs: List[Dict],
    x_metric: str = 'latency',
    y_metric: str = 'energy',
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot Pareto frontier overlaid on all configurations.

    Args:
        dse_results: All DSE results
        pareto_configs: Pareto-optimal subset
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        filename: Optional filename to save plot
        figsize: Figure size

    Note:
        Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed. Cannot plot Pareto frontier.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Plot all configurations
    all_x = [r['stats'].get(x_metric, 0) for r in dse_results if 'stats' in r]
    all_y = [r['stats'].get(y_metric, 0) for r in dse_results if 'stats' in r]
    ax.scatter(all_x, all_y, c='lightgray', s=100, alpha=0.5,
               edgecolors='black', linewidth=1, label='All configs')

    # Plot Pareto frontier
    pareto_x = [r['stats'].get(x_metric, 0) for r in pareto_configs if 'stats' in r]
    pareto_y = [r['stats'].get(y_metric, 0) for r in pareto_configs if 'stats' in r]
    ax.scatter(pareto_x, pareto_y, c='red', s=150, alpha=0.8,
               edgecolors='black', linewidth=2, label='Pareto optimal', marker='*')

    # Sort Pareto points and draw line
    if len(pareto_x) > 1:
        pareto_points = sorted(zip(pareto_x, pareto_y))
        pareto_x_sorted = [p[0] for p in pareto_points]
        pareto_y_sorted = [p[1] for p in pareto_points]
        ax.plot(pareto_x_sorted, pareto_y_sorted, 'r--', alpha=0.5, linewidth=1)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f'Pareto Frontier: {x_metric} vs {y_metric}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    else:
        plt.show()

    plt.close()
