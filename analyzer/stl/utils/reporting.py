"""
Reporting utilities for STL monitoring and DSE results.

Generates human-readable reports in various formats.
"""

from typing import List, Dict, Optional
from datetime import datetime


def generate_stl_report(
    results: List[Dict],
    filename: Optional[str] = None,
    format: str = 'txt'
) -> str:
    """
    Generate a comprehensive STL monitoring report.

    Args:
        results: List of STL evaluation results
        filename: Optional output filename (if None, returns string)
        format: Report format ('txt' or 'md' for markdown)

    Returns:
        Report string (if filename is None)
    """
    if format == 'txt':
        report = _generate_text_report(results)
    elif format == 'md':
        report = _generate_markdown_report(results)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if filename:
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Report saved to {filename}")
        return None
    else:
        return report


def _generate_text_report(results: List[Dict]) -> str:
    """Generate plain text report."""
    lines = []

    lines.append("=" * 80)
    lines.append("STL MONITORING REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Specifications: {len(results)}")
    lines.append("")

    # Summary
    satisfied = sum(1 for r in results if r['satisfied'])
    violated = len(results) - satisfied

    lines.append("SUMMARY:")
    lines.append(f"  Satisfied: {satisfied}")
    lines.append(f"  Violated: {violated}")
    lines.append("")

    # Individual results
    lines.append("SPECIFICATION RESULTS:")
    lines.append("-" * 80)

    for i, result in enumerate(results, 1):
        lines.append(f"\n{i}. {result['name']}")
        lines.append(f"   Formula: {result['specification']}")
        lines.append(f"   Robustness: {result['robustness']:.6f}")
        lines.append(f"   Status: {'SATISFIED' if result['satisfied'] else 'VIOLATED'}")
        lines.append(f"   Signals used: {', '.join(result['signals_used'])}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def _generate_markdown_report(results: List[Dict]) -> str:
    """Generate markdown report."""
    lines = []

    lines.append("# STL Monitoring Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Specifications:** {len(results)}")
    lines.append("")

    # Summary
    satisfied = sum(1 for r in results if r['satisfied'])
    violated = len(results) - satisfied

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Satisfied:** {satisfied}")
    lines.append(f"- **Violated:** {violated}")
    lines.append("")

    # Individual results
    lines.append("## Specification Results")
    lines.append("")

    for i, result in enumerate(results, 1):
        status_icon = "✅" if result['satisfied'] else "❌"
        lines.append(f"### {i}. {result['name']} {status_icon}")
        lines.append("")
        lines.append(f"- **Formula:** `{result['specification']}`")
        lines.append(f"- **Robustness:** {result['robustness']:.6f}")
        lines.append(f"- **Status:** {'SATISFIED' if result['satisfied'] else 'VIOLATED'}")
        lines.append(f"- **Signals:** {', '.join(result['signals_used'])}")
        lines.append("")

    return "\n".join(lines)


def generate_dse_report(
    dse_results: List[Dict],
    filename: Optional[str] = None,
    format: str = 'txt',
    include_violations: bool = True
) -> str:
    """
    Generate Design Space Exploration report.

    Args:
        dse_results: List of DSE results
        filename: Optional output filename
        format: Report format ('txt' or 'md')
        include_violations: Include detailed violation information

    Returns:
        Report string (if filename is None)
    """
    if format == 'txt':
        report = _generate_dse_text_report(dse_results, include_violations)
    elif format == 'md':
        report = _generate_dse_markdown_report(dse_results, include_violations)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if filename:
        with open(filename, 'w') as f:
            f.write(report)
        print(f"DSE report saved to {filename}")
        return None
    else:
        return report


def _generate_dse_text_report(
    dse_results: List[Dict],
    include_violations: bool
) -> str:
    """Generate plain text DSE report."""
    lines = []

    lines.append("=" * 80)
    lines.append("DESIGN SPACE EXPLORATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Configurations: {len(dse_results)}")
    lines.append("")

    # Summary
    satisfying = sum(1 for r in dse_results if r.get('satisfies_all', False))
    lines.append("SUMMARY:")
    lines.append(f"  Configurations satisfying all constraints: {satisfying}")
    lines.append(f"  Configurations with violations: {len(dse_results) - satisfying}")
    lines.append("")

    # Top configurations
    lines.append("TOP CONFIGURATIONS (by minimum robustness):")
    lines.append("-" * 80)

    for i, result in enumerate(dse_results[:10], 1):  # Top 10
        lines.append(f"\nRank {i}: {result['config_name']}")
        lines.append(f"  Min Robustness: {result['min_robustness']:.6f}")
        lines.append(f"  Avg Robustness: {result['avg_robustness']:.6f}")
        lines.append(f"  Satisfies All: {result['satisfies_all']}")
        lines.append(f"  Violations: {result['num_violations']}")

        if 'stats' in result:
            stats = result['stats']
            lines.append(f"  Latency: {stats.get('latency', 'N/A'):.6e} s")
            lines.append(f"  Energy: {stats.get('energy', 'N/A'):.6e} pJ")
            lines.append(f"  Area: {stats.get('area', 'N/A'):.2f} mm²")

        if include_violations and result['num_violations'] > 0:
            lines.append("  Violated Constraints:")
            for stl_result in result.get('stl_results', []):
                if not stl_result['satisfied']:
                    lines.append(f"    - {stl_result['name']}: {stl_result['robustness']:.6f}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def _generate_dse_markdown_report(
    dse_results: List[Dict],
    include_violations: bool
) -> str:
    """Generate markdown DSE report."""
    lines = []

    lines.append("# Design Space Exploration Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Configurations:** {len(dse_results)}")
    lines.append("")

    # Summary
    satisfying = sum(1 for r in dse_results if r.get('satisfies_all', False))
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Configurations satisfying all constraints:** {satisfying}")
    lines.append(f"- **Configurations with violations:** {len(dse_results) - satisfying}")
    lines.append("")

    # Top configurations table
    lines.append("## Top Configurations")
    lines.append("")
    lines.append("| Rank | Config | Min Rob. | Avg Rob. | All Sat? | Viol. | Latency | Energy | Area |")
    lines.append("|------|--------|----------|----------|----------|-------|---------|--------|------|")

    for i, result in enumerate(dse_results[:10], 1):  # Top 10
        config_name = result['config_name']
        min_rob = f"{result['min_robustness']:.4f}"
        avg_rob = f"{result['avg_robustness']:.4f}"
        all_sat = "✅" if result['satisfies_all'] else "❌"
        viol = str(result['num_violations'])

        if 'stats' in result:
            stats = result['stats']
            latency = f"{stats.get('latency', 0):.2e}"
            energy = f"{stats.get('energy', 0):.2e}"
            area = f"{stats.get('area', 0):.2f}"
        else:
            latency = energy = area = "N/A"

        lines.append(f"| {i} | {config_name} | {min_rob} | {avg_rob} | {all_sat} | {viol} | {latency} | {energy} | {area} |")

    lines.append("")

    # Detailed violations (if requested)
    if include_violations:
        lines.append("## Detailed Constraint Violations")
        lines.append("")

        for i, result in enumerate(dse_results[:5], 1):  # Top 5
            if result['num_violations'] > 0:
                lines.append(f"### {i}. {result['config_name']}")
                lines.append("")
                for stl_result in result.get('stl_results', []):
                    if not stl_result['satisfied']:
                        lines.append(f"- **{stl_result['name']}**: {stl_result['robustness']:.6f}")
                        lines.append(f"  - Formula: `{stl_result['specification']}`")
                lines.append("")

    return "\n".join(lines)


def print_comparison_table(
    comparison: Dict,
    config_names: Optional[List[str]] = None
):
    """
    Print a formatted comparison table of configurations.

    Args:
        comparison: Comparison dictionary from ConstraintBasedDSE.compare_configs()
        config_names: Optional list of config names (uses all if None)
    """
    if config_names is None:
        config_names = comparison['configs']

    print("\n" + "=" * 100)
    print("CONFIGURATION COMPARISON")
    print("=" * 100)

    # Print header
    header = f"{'Metric':<25} | " + " | ".join(f"{name:<15}" for name in config_names)
    print(header)
    print("-" * 100)

    # Print metrics
    for metric, values in comparison['metrics'].items():
        row = f"{metric:<25} | "
        for config_name in config_names:
            value = values.get(config_name, 'N/A')
            if isinstance(value, float):
                row += f"{value:<15.6e} | "
            elif isinstance(value, bool):
                row += f"{str(value):<15} | "
            else:
                row += f"{str(value):<15} | "
        print(row)

    print("=" * 100 + "\n")
