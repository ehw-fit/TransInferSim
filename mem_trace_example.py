"""
Memory trace extraction:
  1. Define HW + workload (RobertaBase here)
  2. Run static simulation with log_mem_trace=True
  3. Visualize per-memory occupancy (needed / obsolete / free)

Run from project root:
  python3 mem_trace_example.py
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzer.core.hardware.accelerator import GenericAccelerator
from analyzer.core.hardware.matmul import MatmulArray
from analyzer.hardware_components.memories.offchip import OffChipMemory
from analyzer.hardware_components.memories.shared import SharedMemory
from analyzer.analyzer import Analyzer
from analyzer.model_architectures.transformers.models import RobertaBase

MEM_TRACE_CSV  = "mem_trace_roberta_base.csv"
OUTPUT_SVG     = "mem_trace_roberta_base.svg"
WORD_THRESHOLD = 4096   # skip tiny SA feeder buffers
NCOLS          = 3

# Model
# RobertaBase: 12 layers, seq=256, embed=768, FFN=3072, heads=12, ~6.75 MB weights/layer, ~81 MB total
model = RobertaBase(sequence_length=256)

# HW
# DRAM: 256 MB, DDR-1600, 64-bit channel
dram = OffChipMemory(
    name="dram",
    width=512, depth=4_194_304,
    action_latency=80e-9,
    channel_bus_bitwidth=64,
    cycle_time=1e-9,
    bus_clock_hz=1600e6,
    word_size=8,
    ports=2,
    prefetch_factor=2,
    burst_length=8,
)

hw = GenericAccelerator(
    name="roberta_base_sim",
    cycle_time=1e-9,
    tech_node="45nm",
    auto_interconnect=True,
    dram=dram,
)

# 2× 64×64 SAs (head-dim = 768/12 = 64)
for i in range(2):
    hw.add_matmul_block(MatmulArray(
        rows=64, columns=64,
        data_bitwidth=8,
        cycle_time=1e-9,
        action_latency=1e-9,
        buffer_length=16,
        name=f"sa_{i}",
    ))

# Shared SRAM: 4 MB
shared = SharedMemory(
    name="sram_shared",
    width=512, depth=65_536,
    cycle_time=1e-9,
    action_latency=10e-9,
    bus_bitwidth=512,
    word_size=8,
    ports=4,
    replacement_strategy="lru",
)
hw.add_memory_block(shared)

# Static simulation with memory trace logging
analyzer = Analyzer(model, hw, data_bitwidth=8, num_subops=1)
analyzer.run_simulation_analysis(
    engine_type="static",
    log_mem_trace=True,
    mem_trace_path=MEM_TRACE_CSV,
)
stats = hw.get_statistics(log_mem_contents=False)
print(f"Model        : RobertaBase  ({model.num_macs:,} MACs)")
print(f"Total cycles : {stats['global_cycles']:,}")
print(f"Energy       : {stats['energy']:.4e} J")

# Visualization of memory occupancy traces
df = pd.read_csv(MEM_TRACE_CSV)

visible = [
    m for m in df["memory_name"].unique()
    if df[df["memory_name"] == m]["capacity_words"].iloc[0] > WORD_THRESHOLD
]
print(f"\nMemories plotted ({len(visible)}): {visible}")

ncols = min(len(visible), 3)
nrows = (len(visible) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
fig.suptitle(f"Memory occupancy — {hw.name}  ({model.name})", fontsize=13)

legend_handles = None

for idx, mem_name in enumerate(visible):
    ax = axes[idx // ncols][idx % ncols]
    sub = df[df["memory_name"] == mem_name].sort_values("op_end_time").copy()

    init = sub.iloc[0].copy()
    init["op_end_time"]  = 0
    init["needed_pct"]   = 0.0
    init["obsolete_pct"] = 0.0
    init["free_pct"]     = 1.0
    sub = pd.concat([pd.DataFrame([init]), sub], ignore_index=True)

    x        = sub["op_end_time"].to_numpy()
    needed   = sub["needed_pct"].to_numpy()   * 100
    obsolete = sub["obsolete_pct"].to_numpy() * 100
    used     = needed + obsolete

    h1 = ax.fill_between(x, 0,      needed, color="#4CAF50", alpha=0.9,  label="Needed")
    h2 = ax.fill_between(x, needed, used,   color="#F44336", alpha=0.85, label="Obsolete")
    h3 = ax.fill_between(x, used,   100,    color="#BDBDBD", alpha=0.4,  hatch="//", label="Free")
    ax.axhline(100, color="black", linewidth=1.2, zorder=10)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel("End cycle")
    ax.set_ylabel("Capacity [%]")
    ax.set_title(mem_name, fontsize=10)

    if legend_handles is None:
        legend_handles = [h1, h2, h3]

for idx in range(len(visible), nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

fig.legend(
    handles=legend_handles,
    labels=["Needed", "Obsolete", "Free"],
    loc="lower center",
    ncol=3,
    fontsize=10,
    frameon=True,
    bbox_to_anchor=(0.5, 0.0),
)

fig.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig(OUTPUT_SVG, bbox_inches="tight")
print(f"\nSaved {OUTPUT_SVG}")
