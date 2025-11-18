"""
Example: STL-Guided Design Space Exploration

This example demonstrates how to use STL constraints to automatically
explore and rank hardware configurations for transformer inference.

The design space includes:
- Different systolic array sizes
- Different memory configurations
- Different memory hierarchy designs

STL constraints guide the exploration by filtering and ranking configs
based on formal temporal requirements.
"""

from analyzer.core.hardware.accelerator import GenericAccelerator
from analyzer.core.hardware.matmul import MatmulArray
from analyzer.hardware_components.memories.dedicated import DedicatedMemory
from analyzer.hardware_components.memories.offchip import OffChipMemory
from analyzer.hardware_components.memories.shared import SharedMemory
from analyzer.model_architectures.transformers.models import ViTTiny
from analyzer.analyzer import Analyzer

# Import STL DSE components
from analyzer.stl import (
    ConstraintBasedDSE,
    PerformanceConstraints,
    PowerConstraints,
    ResourceConstraints,
    STL_AVAILABLE
)
from analyzer.stl.dse import ParetoFrontier, RobustnessRanker
from analyzer.stl.utils import generate_dse_report


def create_accelerator_config(
    name: str,
    matmul_size: int,
    num_matmul: int,
    shared_mem_depth: int,
    dedicated_mem_depth: int
):
    """
    Factory function to create accelerator configurations.

    Args:
        name: Configuration name
        matmul_size: Size of systolic array (rows = columns)
        num_matmul: Number of systolic arrays
        shared_mem_depth: Depth of shared memory
        dedicated_mem_depth: Depth of dedicated memories

    Returns:
        GenericAccelerator instance
    """
    # Create DRAM (fixed across configs)
    dram = OffChipMemory(
        name="offchip_mem_1",
        width=1024,
        depth=4096000,
        action_latency=70e-9,
        cycle_time=5e-9,
        bus_clock_hz=200e6,
        bus_bitwidth=32,
        ports=num_matmul,
        prefetch_factor=2,
        burst_length=4
    )

    # Create accelerator
    accelerator = GenericAccelerator(
        name=name,
        cycle_time=5e-9,
        auto_interconnect=True,
        dram=dram
    )

    # Create compute blocks
    for i in range(num_matmul):
        comp_block = MatmulArray(
            rows=matmul_size,
            columns=matmul_size,
            data_bitwidth=8,
            buffer_length=16,
            cycle_time=accelerator.cycle_time,
            name=f"comp_block{i}",
            num_pipeline_stages=1,
            cycles_per_mac=1
        )
        accelerator.add_matmul_block(comp_block)

    # Create shared memory
    shared_mem = SharedMemory(
        name="shared_mem_1",
        width=2048,
        depth=shared_mem_depth,
        cycle_time=accelerator.cycle_time,
        action_latency=5e-9,
        ports=num_matmul * 2,
        bus_bitwidth=32,
        word_size=8,
        replacement_strategy="lru"
    )
    accelerator.add_memory_block(shared_mem)

    # Create dedicated memories
    for i in range(num_matmul):
        dedicated_mem = DedicatedMemory(
            name=f"dedicated_mem_{i}",
            width=1024,
            depth=dedicated_mem_depth,
            cycle_time=accelerator.cycle_time,
            action_latency=5e-9,
            ports=2,
            bus_bitwidth=32,
            word_size=8,
            replacement_strategy="lru"
        )
        accelerator.add_memory_block(dedicated_mem)

    return accelerator


def main():
    print("=" * 80)
    print("STL-Guided Design Space Exploration Example")
    print("=" * 80)
    print()

    if not STL_AVAILABLE:
        print("WARNING: rtamt library not installed.")
        print("Install with: pip install rtamt")
        print()

    # =========================================================================
    # 1. DEFINE MODEL
    # =========================================================================
    print("1. Model: ViT-Tiny")
    model = ViTTiny()
    print()

    # =========================================================================
    # 2. DEFINE STL CONSTRAINTS (Design Requirements)
    # =========================================================================
    print("2. Defining design requirements as STL constraints...")
    print()

    constraints = [
        # Hard constraints (must satisfy)
        PerformanceConstraints.max_latency(
            threshold_seconds=8e-3,  # Max 8ms latency
            name="max_latency_8ms"
        ),

        PowerConstraints.max_energy(
            threshold_joules=150e-9,  # Max 150pJ energy
            name="max_energy_150pJ"
        ),

        ResourceConstraints.max_area(
            threshold_mm2=80.0,  # Max 80 mm²
            name="max_area_80mm2"
        ),

        # Soft constraints (optimize for)
        PerformanceConstraints.min_utilization(
            threshold_percent=0.6,  # Target 60% utilization
            name="target_utilization_60pct"
        ),

        ResourceConstraints.min_cache_hit_rate(
            memory_name="shared_mem_1",
            threshold=0.75,  # Target 75% hit rate
            name="target_hit_rate_75pct"
        ),
    ]

    print(f"   Defined {len(constraints)} constraints:")
    for spec in constraints:
        print(f"   - {spec.name}")
    print()

    # =========================================================================
    # 3. GENERATE DESIGN SPACE
    # =========================================================================
    print("3. Generating hardware configurations...")
    print()

    configs = []

    # Vary systolic array size and memory
    config_params = [
        # (name, matmul_size, num_matmul, shared_depth, dedicated_depth)
        ("small_config", 32, 2, 4096, 2048),
        ("medium_config_a", 64, 2, 8192, 4096),
        ("medium_config_b", 48, 3, 8192, 4096),
        ("large_config_a", 64, 3, 16384, 8192),
        ("large_config_b", 128, 2, 16384, 8192),
        ("xlarge_config", 128, 3, 32768, 16384),
    ]

    for params in config_params:
        name, matmul_size, num_matmul, shared_depth, dedicated_depth = params
        config = create_accelerator_config(name, matmul_size, num_matmul, shared_depth, dedicated_depth)
        configs.append(config)
        print(f"   Created: {name}")
        print(f"     - Matmul: {num_matmul}x {matmul_size}×{matmul_size}")
        print(f"     - Shared mem: {shared_depth * 2048 * 8 / (1024**2):.1f} MB")

    print()
    print(f"   Total configurations: {len(configs)}")
    print()

    # =========================================================================
    # 4. RUN STL-GUIDED DSE
    # =========================================================================
    print("4. Running STL-guided design space exploration...")
    print("   (This will simulate all configurations)")
    print()

    dse = ConstraintBasedDSE(
        model=model,
        constraints=constraints,
        data_bitwidth=8
    )

    # Explore design space
    results = dse.explore_design_space(configs, verbose=True)

    print()
    print("   Exploration completed!")
    print()

    # =========================================================================
    # 5. ANALYZE RESULTS
    # =========================================================================
    print("5. Analyzing results...")
    print()

    # Filter to satisfying configurations
    satisfying = dse.filter_satisfying(results)
    print(f"   Configurations satisfying all constraints: {len(satisfying)}/{len(results)}")
    print()

    # Show top 3 by robustness
    print("   Top 3 configurations (by minimum robustness):")
    print("   " + "-" * 76)
    for i, result in enumerate(results[:3], 1):
        print(f"   {i}. {result['config_name']}")
        print(f"      Min Robustness: {result['min_robustness']:.6f}")
        print(f"      Avg Robustness: {result['avg_robustness']:.6f}")
        print(f"      Satisfies all:  {result['satisfies_all']}")
        print(f"      Violations:     {result['num_violations']}")
        if 'stats' in result:
            print(f"      Latency:        {result['stats']['latency']:.6e} s")
            print(f"      Energy:         {result['stats']['energy']:.6e} pJ")
            print(f"      Area:           {result['stats']['area']:.2f} mm²")
        print()

    # =========================================================================
    # 6. PARETO FRONTIER ANALYSIS
    # =========================================================================
    print("6. Computing Pareto frontier (latency vs energy)...")
    print()

    pareto_configs = ParetoFrontier.compute_pareto_frontier(
        configurations=results,
        objectives=['latency', 'energy'],
        minimize=[True, True],
        extract_metrics=lambda r: r['stats'] if 'stats' in r else {}
    )

    print(f"   Pareto-optimal configurations: {len(pareto_configs)}")
    for i, result in enumerate(pareto_configs, 1):
        print(f"   {i}. {result['config_name']}")
        print(f"      Latency: {result['stats']['latency']:.6e} s")
        print(f"      Energy:  {result['stats']['energy']:.6e} pJ")
        print(f"      Min Robustness: {result['min_robustness']:.6f}")
    print()

    # =========================================================================
    # 7. BEST TRADEOFF SELECTION
    # =========================================================================
    print("7. Selecting best robustness-performance tradeoff...")
    print()

    best_tradeoff = RobustnessRanker.select_best_tradeoff(
        results=results,
        robustness_weight=0.6,  # 60% weight on robustness
        performance_metric='latency',
        minimize_performance=True
    )

    if best_tradeoff:
        print(f"   Best configuration: {best_tradeoff['config_name']}")
        print(f"   Min Robustness: {best_tradeoff['min_robustness']:.6f}")
        print(f"   Latency: {best_tradeoff['stats']['latency']:.6e} s")
        print(f"   Energy: {best_tradeoff['stats']['energy']:.6e} pJ")
        print(f"   Area: {best_tradeoff['stats']['area']:.2f} mm²")
        print()

    # =========================================================================
    # 8. COMPARE SPECIFIC CONFIGS
    # =========================================================================
    print("8. Comparing top 3 configurations...")
    print()

    top_3_names = [r['config_name'] for r in results[:3]]
    comparison = dse.compare_configs(top_3_names)

    from analyzer.stl.utils.reporting import print_comparison_table
    print_comparison_table(comparison)

    # =========================================================================
    # 9. EXPORT RESULTS
    # =========================================================================
    print("9. Exporting results...")
    print()

    generate_dse_report(results, filename="dse_results.txt", format='txt')
    generate_dse_report(results, filename="dse_results.md", format='md')
    dse.export_results(results, filename="dse_results.csv", format='csv')

    print("   Reports generated:")
    print("   - dse_results.txt")
    print("   - dse_results.md")
    print("   - dse_results.csv")
    print()

    # =========================================================================
    # 10. CONCLUSION
    # =========================================================================
    print("=" * 80)
    print("Design Space Exploration completed!")
    print()
    print("Key Insights:")
    print(f"- Evaluated {len(configs)} hardware configurations")
    print(f"- {len(satisfying)} configurations satisfy all constraints")
    print(f"- {len(pareto_configs)} configurations on Pareto frontier")
    print(f"- Best tradeoff: {best_tradeoff['config_name'] if best_tradeoff else 'N/A'}")
    print()
    print("STL-Guided DSE Benefits:")
    print("- Automated filtering of infeasible designs")
    print("- Quantitative ranking by robustness")
    print("- Multi-objective optimization support")
    print("- Formal specification of design requirements")
    print("=" * 80)


if __name__ == "__main__":
    main()
