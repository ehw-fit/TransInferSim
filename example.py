from analyzer.core.hardware.accelerator import GenericAccelerator
from analyzer.core.hardware.generic_memory import GenericMemory
from analyzer.core.hardware.matmul import MatmulArray
from analyzer.hardware_components.memories.dedicated import DedicatedMemory
from analyzer.hardware_components.memories.offchip import OffChipMemory
from analyzer.hardware_components.memories.shared import SharedMemory

from analyzer.model_architectures.transformers.models import RobertaBase, RobertaLarge, ViTBase, ViTLarge, ViTSmall, ViTTiny, DeiTTiny
from analyzer.model_architectures.transformers.layers import MultiHeadSelfAttention, SelfAttention, FeedForwardNetwork, Encoder, Test
from analyzer.analyzer import Analyzer

if __name__ == "__main__":
    """ MODEL/LAYER Specification """
    # NOTE: Currently no batch and bias supported during analysis
    model = ViTTiny()  #  ViTTiny chosen for faster execution here, change with any model/layer
    print(model)

    """ HW Specification """
    # Below is a drawn view of the HW setup described below:
    """                                 
    ┌──────────────────────────┐            
    │┌──────┐ ┌──────┐ ┌──────┐│            
    ││Matmul│ │Matmul│ │Matmul││            
    │└─▲──▲─┘ └─▲──▲─┘ └─▲──▲─┘│            
    │  │  │     │  │     │  │  │            
    │┌────▼─┐ ┌────▼─┐ ┌─▼────┐│            
    ││DedMem│ │DedMem│ │DedMem││            
    │└────▲─┘ └────▲─┘ └─▲────┘│            
    │  │  │     │  │     │  │  │            
    │  │  │   ┌─▼──▼─┐   │  │  │            
    │  │  └───►Shared◄───┘  │  │
    │  └──────►Memory◄──────┘  │            
    │         └─▲▲▲──┘         │            
    └───────────┼┼┼────────────┘            
              ┌─▼▼▼──┐                      
              │ DRAM │                      
              └──────┘                      
    """
    # Create DRAM, accelerator, and all its subcomponents instances (ALL NAMES MUST BE UNIQUE!)

    # NOTE: Offchip DRAM width and depth is not important for the final area. However, its size determines the energy-per-access into DRAM, since the size affects the energy connected with address generation.
    # Minimum width and depth of a memory is 64 to avoid Accelergy errors (CACTI wants at least 64x64 size)
    # the cycle time should match the accelerator cycle time (1/cycle_time gives frequency)
    dram = OffChipMemory(name="offchip_mem_1", width=1024, depth=4096000, action_latency=70e-9, cycle_time=5e-9, bus_bitwidth=32, ports=3)
    # Tech node is set to 45nm for Accelergy (if different is used, it may cause Accelergy errors, resulting in no energy/area stats)
    
    # NOTE: auto interconnect is used to automatically connect matmul blocks with memories based on the lists of components (if set to true, it overrides USER DEFINED INTERCONNECTIONS!)
    accelerator = GenericAccelerator(name="my_accelerator", cycle_time=5e-9, auto_interconnect=True, dram=dram)

    # Define computational blocks (MatmulArray); NOTE: Their sizes can be different
    comp_blocks = []
    for i in range(3):
        comp_blocks.append(MatmulArray(rows=64, columns=64, data_bitwidth=8, cycle_time=accelerator.cycle_time, name=f"comp_block{i}", num_pipeline_stages=1, cycles_per_mac=1))
    
    # 16MB shared memory (used for connection to DRAM and to other memories/matmuls blocks and storing all data types (static and dynamic))
    # Replacement strategies supported: random, lru, lfu, mru, fifo
    # NOTE: LRU is set here because it is efficient and assures deterministic behavior for this scenario
    mem_block_1 = SharedMemory(name="shared_mem_1", width=2048, depth=8192, cycle_time=accelerator.cycle_time, action_latency=5e-9, ports=6, bus_bitwidth=32, word_size=8, replacement_strategy="lru")

    # 4MB dedicated memories are intended for storing static params (i.e. weights) and preferably be uniqely assigned to one matmul block
    dedicated_mems = []
    for i in range(3):
        dedicated_mems.append(DedicatedMemory(name=f"dedicated_mem_{i}", width=1024, depth=4096, cycle_time=accelerator.cycle_time, action_latency=5e-9, ports=2, bus_bitwidth=32, word_size=8, replacement_strategy="lru"))

    # Add the components to the accelerator
    for c in comp_blocks:
        accelerator.add_matmul_block(c)

    accelerator.add_memory_block(mem_block_1)

    for m in dedicated_mems:
        accelerator.add_memory_block(m)

    # OPTIONAL: Interconnect the components (NOTE: auto_interconnect must be set to False during accelerator instantiation to make this take effect)
    
    # The auto_interconnect works as follows:
    # It automatically connects the shared memories in a cascade like manner: [DRAM] <-> [SHARED MEM 1] <-> ... <-> [SHARED MEM N] 
    # Then to the last shared mem N it connects dedicated memories and matmuls (all dedicated mems and all matmuls to that shared mem N, since it provides a place to store dynamic params for matmuls and for fetching of static params for dedicated memories)
    # If dedicated memories are present, always one is attempted to be assigned and connected with one matmul block (one ded. mem per matmul block) and are also connected to the upper shared mem (if none exists, then straight to DRAM) and to matmul.

    # NOTE: auto_interconnect WAS NOT TESTED FOR TOO COMPLEX HIERARCHIES, users are encouraged to set the connections manually for such cases
    # The following setting exactly matches the auto_interconnect setting, which the engine would do automatically
    
    # Shared mem 1 gets data from dram
    mem_block_1.set_upper_level_memory(dram)

    # Connect dedicated memories (holding static params) to upper memory for data retrieval (shared mem 1) and to the individual assigned matmul arrays
    dedicated_mems[0].set_upper_level_memory(mem_block_1)
    dedicated_mems[1].set_upper_level_memory(mem_block_1)
    dedicated_mems[2].set_upper_level_memory(mem_block_1)

    dedicated_mems[0].add_associated_matmul(comp_blocks[0])
    dedicated_mems[1].add_associated_matmul(comp_blocks[1])
    dedicated_mems[2].add_associated_matmul(comp_blocks[2])

    # Connect memories to comp blocks (for retrieving input static params and retrieving/storing the input/output dynamic params)
    comp_blocks[0].assign_static_params_memory(dedicated_mems[0])
    comp_blocks[0].assign_dynamic_params_memory(mem_block_1)
    
    comp_blocks[1].assign_static_params_memory(dedicated_mems[1])
    comp_blocks[1].assign_dynamic_params_memory(mem_block_1)
    
    comp_blocks[2].assign_static_params_memory(dedicated_mems[2])
    comp_blocks[2].assign_dynamic_params_memory(mem_block_1)

    """ ANALYZER """
    analyzer = Analyzer(model, accelerator, data_bitwidth=8)  # data bitwidth is currently considered uniform for all data tensors
    analyzer.visualize_graph()

    """ SIMULATION """
    # None scheduling key == uniform distribution of compute operations across compute units
    # NOTE: Turn on verbose to see detailed reports of the simulation run, but be aware that it is written to STDOUT now, so a redirection to a file is recommended
    analyzer.run_simulation_analysis(verbose=False, permutation_seed=42, scheduling_seed=None, engine_type="static")

    """ METRICS RETRIEVAL """
    # NOTE: To see the contents of individual memories (per-tensor read/write log data) turn to True, but it may generate a lot of details
    stats = accelerator.get_statistics(log_mem_contents=False)
    GenericAccelerator.pretty_print_stats(stats, verbose=False, file_path="stats_out.txt")
