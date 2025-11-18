"""
Example: Using STL Constraints with TransInferSim

This example demonstrates how to use Signal Temporal Logic (STL) constraints
to monitor hardware performance metrics during transformer inference simulation.

STL provides a formal way to specify temporal requirements (e.g., "latency must
always be below 10ms") and quantify how well they are satisfied through robustness values.
"""

from analyzer.core.hardware.accelerator import GenericAccelerator
from analyzer.core.hardware.matmul import MatmulArray
from analyzer.hardware_components.memories.dedicated import DedicatedMemory
from analyzer.hardware_components.memories.offchip import OffChipMemory
from analyzer.hardware_components.memories.shared import SharedMemory
from analyzer.model_architectures.transformers.models import ViTTiny
from analyzer.analyzer import Analyzer

# Import STL components
from analyzer.stl import (
    OfflineSTLMonitor,
    PerformanceConstraints,
    PowerConstraints,
    ResourceConstraints,
    CompositeConstraints,
    STL_AVAILABLE
)
from analyzer.stl.utils import generate_stl_report

def main():
    print("=" * 80)
    print("TransInferSim with STL Constraints Example")
    print("=" * 80)
    print()

    # Check if STL is available
    if not STL_AVAILABLE:
        print("WARNING: rtamt library not installed. STL monitoring will have limited functionality.")
        print("Install with: pip install rtamt")
        print()

    # =========================================================================
    # 1. MODEL SPECIFICATION
    # =========================================================================
    print("1. Creating Transformer model (ViT-Tiny)...")
    model = ViTTiny()
    print(f"   Model: {model}")
    print()

    # =========================================================================
    # 2. HARDWARE SPECIFICATION (Same as example.py)
    # =========================================================================
    print("2. Defining hardware architecture...")

    # Create DRAM
    dram = OffChipMemory(
        name="offchip_mem_1",
        width=1024,
        depth=4096000,
        action_latency=70e-9,
        cycle_time=5e-9,
        bus_clock_hz=200e6,
        bus_bitwidth=32,
        ports=3,
        prefetch_factor=2,
        burst_length=4
    )

    # Create accelerator with auto-interconnect
    accelerator = GenericAccelerator(
        name="my_accelerator",
        cycle_time=5e-9,
        auto_interconnect=True,
        dram=dram
    )

    # Define 3 computational blocks (64x64 systolic arrays)
    comp_blocks = []
    for i in range(3):
        comp_blocks.append(
            MatmulArray(
                rows=64,
                columns=64,
                data_bitwidth=8,
                buffer_length=16,
                cycle_time=accelerator.cycle_time,
                name=f"comp_block{i}",
                num_pipeline_stages=1,
                cycles_per_mac=1
            )
        )

    # 16MB shared memory
    mem_block_1 = SharedMemory(
        name="shared_mem_1",
        width=2048,
        depth=8192,
        cycle_time=accelerator.cycle_time,
        action_latency=5e-9,
        ports=6,
        bus_bitwidth=32,
        word_size=8,
        replacement_strategy="lru"
    )

    # 4MB dedicated memories for weights
    dedicated_mems = []
    for i in range(3):
        dedicated_mems.append(
            DedicatedMemory(
                name=f"dedicated_mem_{i}",
                width=1024,
                depth=4096,
                cycle_time=accelerator.cycle_time,
                action_latency=5e-9,
                ports=2,
                bus_bitwidth=32,
                word_size=8,
                replacement_strategy="lru"
            )
        )

    # Add components to accelerator
    for c in comp_blocks:
        accelerator.add_matmul_block(c)
    accelerator.add_memory_block(mem_block_1)
    for m in dedicated_mems:
        accelerator.add_memory_block(m)

    print(f"   Accelerator: {accelerator.name}")
    print(f"   Compute blocks: {len(comp_blocks)}")
    print(f"   Memory blocks: {len(accelerator.memory_blocks)}")
    print()

    # =========================================================================
    # 3. DEFINE STL CONSTRAINTS
    # =========================================================================
    print("3. Defining STL temporal constraints...")
    print()

    constraints = [
        # Performance constraints
        PerformanceConstraints.max_latency(
            threshold_seconds=10e-3,  # Max 10ms latency
            name="max_latency_10ms"
        ),

        PerformanceConstraints.min_utilization(
            threshold_percent=0.5,  # Min 50% utilization
            name="min_utilization_50pct"
        ),

        # Power constraints
        PowerConstraints.max_energy(
            threshold_joules=200e-9,  # Max 200pJ energy
            name="max_energy_200pJ"
        ),

        PowerConstraints.max_edp_latency(
            threshold=2e-12,  # Max EDP
            name="max_edp_2e-12"
        ),

        # Resource constraints
        ResourceConstraints.max_area(
            threshold_mm2=100.0,  # Max 100 mm²
            name="max_area_100mm2"
        ),

        ResourceConstraints.min_cache_hit_rate(
            memory_name="shared_mem_1",
            threshold=0.7,  # Min 70% hit rate
            name="min_shared_mem_hit_rate_70pct"
        ),

        # Composite constraint
        CompositeConstraints.pareto_optimal(
            max_latency=15e-3,
            max_energy=250e-9,
            max_area=150.0,
            name="pareto_frontier_constraint"
        ),
    ]

    print(f"   Defined {len(constraints)} STL constraints:")
    for i, spec in enumerate(constraints, 1):
        print(f"   {i}. {spec.name}")
        print(f"      Formula: {spec.formula}")
    print()

    # =========================================================================
    # 4. RUN SIMULATION (Standard TransInferSim workflow)
    # =========================================================================
    print("4. Running cycle-accurate simulation...")
    analyzer = Analyzer(model, accelerator, data_bitwidth=8)
    analyzer.run_simulation_analysis(verbose=False, permutation_seed=42)
    stats = accelerator.get_statistics(log_mem_contents=False)

    print(f"   Simulation completed:")
    print(f"   - Latency: {stats['latency']:.6e} s")
    print(f"   - Energy: {stats['energy']:.6e} pJ")
    print(f"   - Area: {stats['area']:.2f} mm²")
    print(f"   - Avg Utilization: {stats['avg_utilization']:.2%}")
    print()

    # =========================================================================
    # 5. STL MONITORING (NEW!)
    # =========================================================================
    print("5. Evaluating STL constraints...")
    print()

    # Create STL monitor
    monitor = OfflineSTLMonitor(constraints, accelerator)

    # List available signals
    available_signals = monitor.get_available_signals(stats)
    print(f"   Available signals for monitoring: {len(available_signals)}")
    # print(f"   Signals: {', '.join(available_signals[:10])}...")  # Show first 10
    print()

    # Validate that all constraints can be evaluated
    validation = monitor.validate_specifications(stats)
    print("   Constraint validation:")
    for spec_name, val_info in validation.items():
        status = "✓" if val_info['valid'] else "✗"
        print(f"   {status} {spec_name}")
        if not val_info['valid']:
            print(f"      Missing signals: {val_info['missing_signals']}")
    print()

    # Evaluate constraints
    print("   Evaluating constraints against simulation results...")
    results = monitor.evaluate(stats)

    # =========================================================================
    # 6. ANALYZE RESULTS
    # =========================================================================
    print()
    print("6. STL Evaluation Results:")
    print("=" * 80)
    print()

    for result in results:
        status_icon = "✅" if result['satisfied'] else "❌"
        print(f"{status_icon} {result['name']}")
        print(f"   Formula: {result['specification']}")
        print(f"   Robustness: {result['robustness']:.6f}")
        print(f"   Status: {'SATISFIED' if result['satisfied'] else 'VIOLATED'}")
        print()

    # Print summary
    print("=" * 80)
    monitor.print_summary()

    # =========================================================================
    # 7. GENERATE REPORT
    # =========================================================================
    print()
    print("7. Generating STL report...")
    generate_stl_report(results, filename="stl_monitoring_report.txt", format='txt')
    generate_stl_report(results, filename="stl_monitoring_report.md", format='md')
    print("   Reports saved:")
    print("   - stl_monitoring_report.txt")
    print("   - stl_monitoring_report.md")
    print()

    # =========================================================================
    # 8. CONCLUSION
    # =========================================================================
    print("=" * 80)
    print("Example completed!")
    print()
    print("Key Takeaways:")
    print("- STL provides formal temporal constraint specification")
    print("- Robustness values quantify 'how well' constraints are satisfied")
    print("- Positive robustness = satisfied, negative = violated")
    print("- Magnitude indicates distance from threshold")
    print()
    print("Next steps:")
    print("- Modify constraints to match your design requirements")
    print("- Use STL for automated design space exploration (see example_dse.py)")
    print("- Integrate with optimization loops for robustness-guided tuning")
    print("=" * 80)


if __name__ == "__main__":
    main()
