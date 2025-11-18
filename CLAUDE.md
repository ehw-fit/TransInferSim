# CLAUDE.md - TransInferSim Codebase Guide for AI Assistants

## Project Overview

**TransInferSim** is a cycle-accurate simulator for analyzing the hardware performance of Transformer neural network inference on custom systolic-array accelerators. It integrates with Accelergy to provide comprehensive performance metrics including latency, energy consumption, area, and efficiency metrics.

**Key Features:**
- Cycle-accurate simulation of Transformer NN inference
- Integration with Accelergy for energy/area estimation
- Support for custom hardware configurations (systolic arrays, memory hierarchies)
- Cache policy analysis and memory hierarchy optimization
- Hardware design-space exploration
- Exportable execution plans for RTL validation

**Project Metadata:**
- **License:** MIT
- **Author:** Jan Klhůfek (iklhufek@fit.vut.cz)
- **Repository:** https://github.com/ehw-fit/TransInferSim
- **Version:** 0.1
- **Python Requirement:** 3.8+
- **Paper:** IEEE Access 2025, Vol. 13, pp. 177215-177226

---

## Codebase Structure

### Directory Layout

```
TransInferSim/
├── analyzer/                           # Main Python package (~6,097 LOC)
│   ├── core/                          # Core framework components
│   │   ├── hardware/                  # Hardware abstraction layer
│   │   │   ├── accelerator.py        # GenericAccelerator class
│   │   │   ├── matmul.py             # MatmulArray (systolic array)
│   │   │   └── generic_memory.py     # GenericMemory base class
│   │   ├── model_architectures/       # Model base classes
│   │   │   ├── transformer_blocks/   # TransformerModel, TransformerLayer
│   │   │   └── convolutional_blocks/ # CNN support (minimal)
│   │   └── simulator/                # Simulation engine
│   │       ├── simulator.py          # StaticSimulationEngine, EventScheduler
│   │       ├── events.py             # Event-driven simulation events
│   │       └── utils.py              # Simulation utilities
│   ├── hardware_components/           # Concrete hardware implementations
│   │   └── memories/                 # Memory hierarchy components
│   │       ├── shared.py             # SharedMemory (SRAM)
│   │       ├── dedicated.py          # DedicatedMemory (weights)
│   │       ├── offchip.py            # OffChipMemory (DRAM)
│   │       └── buffer.py             # BufferStack (register files)
│   ├── model_architectures/           # Concrete model implementations
│   │   ├── transformers/             # Transformer architectures
│   │   │   ├── models/               # ViT, RoBERTa, DeiT, DeepSeekV2
│   │   │   └── layers/               # Attention, FFN, Encoder
│   │   └── cnns/                     # CNN support (placeholder)
│   ├── utils/                        # Utility functions
│   │   └── utils.py                  # Tiling, tensor tracking
│   └── analyzer.py                   # Main Analyzer orchestrator
├── accelergy/                         # Git submodule - Accelergy core
├── accelergy_plugins/                 # Git submodules - Energy plugins
│   ├── accelergy-aladdin-plug-in/
│   ├── accelergy-library-plug-in/
│   ├── accelergy-cacti-plug-in/      # Requires C++ compilation
│   ├── accelergy-adc-plug-in/
│   └── accelergy-neurosim-plugin/    # Requires C++ extension build
├── scripts/
│   └── setup_submodules.sh           # Submodule initialization script
├── .github/workflows/
│   └── python-test.yml               # CI/CD pipeline
├── compound_components.yaml           # Accelergy component definitions
├── example.py                         # Example usage script
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
└── README.md                         # User documentation
```

### Package Structure

The `analyzer` package follows a layered architecture:
1. **Core Layer** (`analyzer/core/`): Abstract base classes and framework
2. **Components Layer** (`analyzer/hardware_components/`, `analyzer/model_architectures/`): Concrete implementations
3. **Orchestration Layer** (`analyzer/analyzer.py`): High-level analysis coordination
4. **Utilities Layer** (`analyzer/utils/`): Helper functions

---

## Key Components and Architecture

### 1. Hardware Abstraction Layer

**GenericAccelerator** (`analyzer/core/hardware/accelerator.py`):
- Main accelerator class managing computational and memory blocks
- Handles cycle-accurate simulation and component interconnection
- Exports to Accelergy YAML format for energy/area estimation
- Supports auto-interconnect feature for simple topologies

**MatmulArray** (`analyzer/core/hardware/matmul.py`):
- Systolic array implementation for matrix multiplication
- Configurable: rows×columns, pipeline stages, cycles per MAC
- Includes row/column buffers (BufferStack) for data feeding
- Tracks computational/idle cycles, FLOP counts, peak throughput

**GenericMemory** (`analyzer/core/hardware/generic_memory.py`):
- Abstract base for memory hierarchy
- Supports multiple ports, configurable replacement strategies:
  - LRU (Least Recently Used)
  - LFU (Least Frequently Used)
  - FIFO (First In First Out)
  - MRU (Most Recently Used)
  - Random
- Tracks cache statistics, bandwidth utilization

### 2. Memory Hierarchy Components

**SharedMemory** (`analyzer/hardware_components/memories/shared.py`):
- SRAM-based shared memory (smartbuffer_SRAM in Accelergy)
- Stores both static and dynamic parameters
- Typically connected to DRAM and compute blocks

**DedicatedMemory** (`analyzer/hardware_components/memories/dedicated.py`):
- Dedicated memory for static parameters (weights)
- Usually uniquely assigned to one matmul block
- Reduces contention for shared resources

**OffChipMemory** (`analyzer/hardware_components/memories/offchip.py`):
- DRAM with DDR characteristics
- Supports burst transfers and prefetch
- Action latency and bus clock configurable

**BufferStack** (`analyzer/hardware_components/memories/buffer.py`):
- Register file buffers for systolic array feeding
- Row and column buffers with configurable depth

### 3. Simulation Engine

**StaticSimulationEngine** (`analyzer/core/simulator/simulator.py`):
- Static scheduling simulation (currently implemented)
- Event-driven using discrete event scheduler (heapq-based)
- Supports deterministic and non-deterministic execution

**Event-Driven Architecture** (`analyzer/core/simulator/events.py`):
- MatmulStartEvent, TemporalTileCompleteEvent, SpatialTileCompleteEvent
- MemoryReadEvent, MemoryWriteEvent, MemoryFetchEvent, MemoryWriteBackEvent
- MemoryReadCompleteEvent, MemoryWriteCompleteEvent

**Simulation Utilities** (`analyzer/core/simulator/utils.py`):
- `eval_operation_duration()`: Calculate matmul cycles with spatial/temporal tiling
- `eval_mem_time_cycles()`: Calculate memory transfer cycles
- `get_memory_paths()`: Build memory hierarchy paths

### 4. Model Architectures

**Transformer Models** (`analyzer/model_architectures/transformers/models/`):
- **ViT variants:** ViTTiny (12 layers, 192 dim, 3 heads), ViTSmall, ViTBase, ViTLarge
- **DeiT:** DeiTTiny (data-efficient image transformer)
- **RoBERTa:** RobertaBase, RobertaLarge (NLP)
- **DeepSeekV2:** 60 layers, 5120 dim, 40 heads, supports Multi-Head Latent Attention

**Transformer Layers** (`analyzer/model_architectures/transformers/layers/`):
- `SelfAttention`: Standard self-attention mechanism
- `MultiHeadSelfAttention`: Multi-head attention with parallel Q/K/V projections
- `LatentAttentionHead`: Low-rank latent attention (DeepSeekV2)
- `MultiHeadLatentAttention`: Multi-head version with compression
- `FeedForwardNetwork`: Two-layer MLP
- `Encoder`: Complete encoder block (attention + FFN)

### 5. Analysis Orchestrator

**Analyzer** (`analyzer/analyzer.py`):
- Builds execution graph (DAG) from model execution plan
- ExecutionGraph: Stores operation dependencies
- ExecutionNode: Individual operation with inputs/outputs/dependencies
- Manages hardware-software mapping
- Runs cycle-accurate simulation
- Visualizes execution DAG using Graphviz
- Supports operation tiling and scheduling

---

## Development Workflow

### Initial Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/ehw-fit/TransInferSim
cd TransInferSim

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and tools
pip install --upgrade pip wheel setuptools

# Build Accelergy submodules (includes C++ compilation)
./scripts/setup_submodules.sh

# Install TransInferSim
pip install .
```

### Running Example

```bash
# Run the example script
python example.py

# Output: stats_out.txt with performance metrics
```

### Development Dependencies

**System Requirements:**
- Graphviz (for visualization): `sudo apt-get install graphviz`
- make, g++ (for Accelergy plugins)
- Python 3.8+ (tested on 3.8, 3.11, 3.x)

**Python Dependencies:**
- graphviz==0.20.3 (Python bindings)
- numpy>=1.24,<2.0 (numerical operations)
- PyYAML>=6.0 (configuration parsing)

### Git Workflow

**Submodule Management:**
```bash
# Initialize/update submodules
git submodule update --init --recursive

# Pull latest changes in submodules
git submodule update --remote
```

**Branch Strategy:**
- Main branch: `main` (stable releases)
- Development: Feature branches
- CI/CD: GitHub Actions on all pushes/PRs

---

## Coding Conventions

### Naming Conventions

**Classes:**
- PascalCase for class names: `GenericAccelerator`, `MatmulArray`, `SharedMemory`
- Descriptive names indicating purpose

**Methods/Functions:**
- snake_case for methods: `add_matmul_block()`, `get_statistics()`, `eval_operation_duration()`
- Prefix private methods with underscore: `_params`, `_is_done`, `_auto_interconnect_set`

**Variables:**
- snake_case: `cycle_time`, `data_bitwidth`, `num_pipeline_stages`
- Descriptive names avoiding abbreviations unless standard (e.g., `mem`, `hw`)

**Components:**
- All hardware component instances **MUST have unique names**
- Example: `comp_block0`, `comp_block1`, `shared_mem_1`, `dedicated_mem_0`

### Code Style

**Documentation:**
- Extensive docstrings for classes (triple quotes)
- Parameter descriptions in `__init__` methods
- TODO comments for future work: `# TODO make PE its own object type...`

**Assertions:**
- Heavy use of assertions for input validation
- Example:
  ```python
  assert rows > 0, "Number of rows must be positive."
  assert len(name) > 0, "Name must be a non-empty string."
  ```

**Type Hints:**
- Partial type hints used (primarily in function signatures)
- Example: `def __init__(self, name: str, data_bitwidth: int, cycle_time: float)`

**String Formatting:**
- f-strings for formatting: `f"comp_block{i}"`, `f"{name}_rowbuf"`
- Multi-line f-strings for complex output

**Imports:**
- Standard library first, then third-party, then local
- Absolute imports preferred: `from analyzer.core.hardware.matmul import MatmulArray`

### Architecture Patterns

**Abstract Base Classes:**
- Core framework uses abstract base classes
- Concrete implementations in `hardware_components/` and `model_architectures/`
- Example: `GenericMemory` (abstract) → `SharedMemory`, `DedicatedMemory` (concrete)

**Composition Over Inheritance:**
- Hardware components composed of sub-components
- Example: `MatmulArray` contains `BufferStack` instances

**Event-Driven Simulation:**
- Events inherit from base event classes
- Scheduler manages event queue using heapq (priority queue)

**Builder Pattern:**
- Accelerator built incrementally: add matmul blocks, add memory blocks, interconnect
- Example pattern:
  ```python
  accelerator = GenericAccelerator(...)
  accelerator.add_matmul_block(comp_block)
  accelerator.add_memory_block(mem_block)
  comp_block.assign_static_params_memory(dedicated_mem)
  ```

---

## Common Development Tasks

### Adding a New Transformer Model

1. **Create model file** in `analyzer/model_architectures/transformers/models/`
2. **Inherit from** `TransformerModel` base class
3. **Define parameters** (num_layers, embedding_dim, num_heads, etc.)
4. **Implement** `forward()` method defining execution plan
5. **Update** `__init__.py` to export the new model
6. **Test** with example.py

Example structure:
```python
from analyzer.core.model_architectures.transformer_blocks.transformer_model import TransformerModel

class MyTransformer(TransformerModel):
    def __init__(self, num_layers=12, embedding_dim=768, ...):
        # Define parameters
        self.parameters = {...}
        super().__init__(name="MyTransformer")

    def forward(self):
        # Define execution plan
        ...
```

### Adding a New Memory Type

1. **Create memory file** in `analyzer/hardware_components/memories/`
2. **Inherit from** `GenericMemory` base class
3. **Override** necessary methods (if any)
4. **Set** Accelergy component name in constructor
5. **Update** `__init__.py` to export

### Modifying Hardware Configurations

**In example.py or custom script:**
```python
# Modify matmul array size
comp_block = MatmulArray(rows=128, columns=128, ...)  # Larger array

# Change memory size/hierarchy
shared_mem = SharedMemory(width=4096, depth=16384, ...)  # 32MB instead of 16MB

# Adjust memory replacement strategy
shared_mem = SharedMemory(..., replacement_strategy="lfu")  # Use LFU instead of LRU

# Configure DRAM characteristics
dram = OffChipMemory(..., action_latency=100e-9, burst_length=8)
```

### Adding Visualization

```python
analyzer = Analyzer(model, accelerator, data_bitwidth=8)

# Visualize execution graph
analyzer.visualize_graph("my_graph_name")  # Generates my_graph_name.png/pdf
```

### Running Simulations

```python
# Static simulation engine
analyzer.run_simulation_analysis(
    verbose=False,           # Set True for detailed logs (STDOUT)
    permutation_seed=42,     # Seed for reproducibility
    scheduling_seed=None,    # None = uniform distribution across compute units
    engine_type="static"     # Currently only "static" fully implemented
)

# Retrieve statistics
stats = accelerator.get_statistics(log_mem_contents=False)  # Set True for per-tensor logs

# Pretty print to file
GenericAccelerator.pretty_print_stats(stats, verbose=False, file_path="stats_out.txt")
```

---

## Testing and CI/CD

### Current Testing Strategy

**No formal unit test suite** - testing is integration-based:
- CI/CD runs `example.py` as integration test
- Validates across Python 3.8, 3.11, and 3.x
- Checks build success and output generation

**GitHub Actions CI/CD** (`.github/workflows/python-test.yml`):
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.11', '3.x']
    steps:
      - Checkout with submodules
      - Install system dependencies (graphviz, make, g++)
      - Build Accelergy submodules
      - Install TransInferSim
      - Run example.py
      - Upload stats_out.txt as artifact
```

### Testing Locally

```bash
# Run example script
python example.py

# Check stats output
cat stats_out.txt

# Verify Accelergy integration
ls -la *.yaml  # Should see generated Accelergy YAML files
```

### Adding Tests (Future)

If adding unit tests:
1. Create `tests/` directory
2. Use pytest framework (industry standard)
3. Test individual components in isolation
4. Update CI/CD to run pytest

---

## Important Patterns and Conventions

### Memory Hierarchy Setup

**Auto-Interconnect (Recommended for Simple Topologies):**
```python
accelerator = GenericAccelerator(name="my_accelerator", cycle_time=5e-9,
                                  auto_interconnect=True, dram=dram)
# Add components - they will be auto-connected
```

**Auto-interconnect logic:**
- Connects shared memories in cascade: DRAM ↔ SharedMem1 ↔ ... ↔ SharedMemN
- Connects dedicated memories and matmuls to last shared memory
- Assigns one dedicated memory per matmul block (if available)

**Manual Interconnect (Complex Hierarchies):**
```python
accelerator = GenericAccelerator(..., auto_interconnect=False, dram=dram)

# Explicitly set memory hierarchy
mem_block_1.set_upper_level_memory(dram)
dedicated_mem_0.set_upper_level_memory(mem_block_1)

# Assign memories to compute blocks
comp_block_0.assign_static_params_memory(dedicated_mem_0)
comp_block_0.assign_dynamic_params_memory(mem_block_1)
```

### Component Naming Requirements

**CRITICAL:** All component names must be unique within an accelerator.

Bad:
```python
comp_block = MatmulArray(name="comp_block", ...)
comp_block = MatmulArray(name="comp_block", ...)  # CONFLICT!
```

Good:
```python
for i in range(3):
    comp_blocks.append(MatmulArray(name=f"comp_block{i}", ...))
```

### Accelergy Configuration

**Technology Node:** Default is 45nm
- Changing may cause Accelergy errors
- Defined in `compound_components.yaml`

**Minimum Memory Size:** 64×64 bits
- CACTI requires at least 64 width and 64 depth
- Smaller sizes will cause Accelergy errors

**Component Mapping:**
- `MatmulArray` → `mac` component in Accelergy
- `SharedMemory` → `smartbuffer_SRAM`
- `DedicatedMemory` → `smartbuffer_SRAM`
- `OffChipMemory` → `offchip_DRAM`
- `BufferStack` → `smartbuffer_RF`

### Execution Plan Pattern

Models define execution plans as dictionaries:
```python
def forward(self):
    execution_plan = {
        'operator_type': 'matmul',
        'name': 'unique_operation_name',
        'computation': {'A_shape': ..., 'B_shape': ..., 'C_shape': ...},
        'input_data': {'A': 'tensor_name', 'B': 'tensor_name'},
        'output_data': {'C': 'tensor_name'},
        'batches': batch_size,
        'dependant_on': [list_of_prerequisite_operations]
    }
    return execution_plan
```

---

## Known Limitations and TODOs

### Current Limitations

**Model Support:**
- No batch processing support in analysis (batch size ignored in some contexts)
- No bias addition in operations
- Softmax and LayerNorm ignored in performance analysis
- CNN support is minimal/placeholder

**Hardware:**
- Processing elements (PEs) not separate object types (part of MatmulArray)
- Auto-interconnect not thoroughly tested for complex hierarchies

**Simulation:**
- Only static scheduling engine fully implemented
- Dynamic scheduling engine: TODO
- Genetic algorithm optimization: TODO

**Data Types:**
- Uniform bitwidth across all tensors (non-uniform bitwidth TODO)
- All tensors use same data_bitwidth parameter

### Common TODO Comments in Code

From codebase analysis:
- `# TODO make PE its own object type...` (matmul.py)
- `# TODO ADD POSSIBLE BIAS ADDITION LOGIC` (matmul.py)
- `# TODO Uniform for simplicity.. could be tied to individual data tensors` (analyzer.py)
- `# TODO make automated.. Number of sub operations` (analyzer.py)
- `# TODO ... HERE.. STATIC CHECKS FOR MEMORY SIZES` (analyzer.py)
- `# TODO CHECK AUTO INTERCONNECTION FOR COMPLEX ARCHITECTURES` (analyzer.py)
- `# TODO future!!! FOR SUPPORT OF NONUNIFORM BITWIDTH` (analyzer.py)

---

## Working with the Codebase

### File Modification Guidelines

**When modifying core classes:**
1. Preserve existing interfaces (breaking changes affect all users)
2. Add assertions for new parameters
3. Update docstrings
4. Maintain backward compatibility where possible

**When adding features:**
1. Follow existing patterns (base class → concrete implementation)
2. Update corresponding `__init__.py` files
3. Add example usage in comments or separate example file
4. Consider Accelergy integration (YAML export)

### .gitignore Patterns

**Important exclusions:**
- `*.yaml` - Accelergy output files (generated)
- `*.png`, `*.pdf` - Visualization outputs (generated)
- `*.csv` - Statistics exports (generated)
- `analyzer_*.py`, `analyze*.py`, `experiment*.py`, `test*.py` - User test scripts

**Tracked files:**
- `compound_components.yaml` - Accelergy component definitions (tracked)
- `example.py` - Official example (tracked)
- All source code in `analyzer/` package

### Debugging Tips

**Verbose Mode:**
```python
analyzer.run_simulation_analysis(verbose=True, ...)
# WARNING: Produces extensive STDOUT output - redirect to file recommended
```

**Memory Contents Logging:**
```python
stats = accelerator.get_statistics(log_mem_contents=True)
# Shows per-tensor read/write access patterns
```

**Execution Graph Visualization:**
```python
analyzer.visualize_graph("debug_graph")
# Visual inspection of operation dependencies
```

**Component Statistics:**
```python
GenericAccelerator.pretty_print_stats(stats, verbose=True, file_path="debug.txt")
# Detailed per-component breakdown
```

### Performance Considerations

**Simulation Performance:**
- Larger models (e.g., ViTLarge) take longer to simulate
- Use ViTTiny for quick testing/development
- Static engine is deterministic (same seed = same results)

**Memory Simulation:**
- LRU replacement is deterministic and generally efficient
- Random replacement is non-deterministic (use seed for reproducibility)
- Larger cache sizes reduce simulation time (fewer evictions)

**Accelergy Integration:**
- First run builds plugin components (slow)
- Subsequent runs use cached components (faster)
- Energy/area estimation time depends on component complexity

---

## Quick Reference

### Essential Imports

```python
# Hardware components
from analyzer.core.hardware.accelerator import GenericAccelerator
from analyzer.core.hardware.matmul import MatmulArray
from analyzer.hardware_components.memories.shared import SharedMemory
from analyzer.hardware_components.memories.dedicated import DedicatedMemory
from analyzer.hardware_components.memories.offchip import OffChipMemory

# Models
from analyzer.model_architectures.transformers.models import (
    ViTTiny, ViTSmall, ViTBase, ViTLarge,
    RobertaBase, RobertaLarge, DeiTTiny
)

# Layers
from analyzer.model_architectures.transformers.layers import (
    SelfAttention, MultiHeadSelfAttention,
    FeedForwardNetwork, Encoder
)

# Analyzer
from analyzer.analyzer import Analyzer
```

### Typical Workflow

```python
# 1. Define model
model = ViTTiny()

# 2. Create hardware
dram = OffChipMemory(...)
accelerator = GenericAccelerator(..., auto_interconnect=True, dram=dram)
comp_blocks = [MatmulArray(...) for i in range(3)]
mem_blocks = [SharedMemory(...), DedicatedMemory(...), ...]

# 3. Add components
for cb in comp_blocks:
    accelerator.add_matmul_block(cb)
for mb in mem_blocks:
    accelerator.add_memory_block(mb)

# 4. Create analyzer
analyzer = Analyzer(model, accelerator, data_bitwidth=8)

# 5. Visualize (optional)
analyzer.visualize_graph("my_graph")

# 6. Simulate
analyzer.run_simulation_analysis(verbose=False, engine_type="static")

# 7. Get results
stats = accelerator.get_statistics()
GenericAccelerator.pretty_print_stats(stats, file_path="results.txt")
```

### Key File References

| File | Purpose |
|------|---------|
| `example.py` | Complete usage example |
| `analyzer/analyzer.py` | Main orchestrator (383 LOC) |
| `analyzer/core/hardware/accelerator.py` | Accelerator class |
| `analyzer/core/hardware/matmul.py` | Systolic array |
| `analyzer/core/simulator/simulator.py` | Simulation engine |
| `compound_components.yaml` | Accelergy component definitions |
| `setup.py` | Package metadata and dependencies |
| `scripts/setup_submodules.sh` | Submodule build script |

---

## Additional Resources

**Paper Reference:**
```
J. Klhufek, A. Marchisio, V. Mrazek, L. Sekanina and M. Shafique,
"TransInferSim: Toward Fast and Accurate Evaluation of Embedded Hardware
Accelerators for Transformer Networks," in IEEE Access, vol. 13,
pp. 177215-177226, 2025, doi: 10.1109/ACCESS.2025.3621062.
```

**Accelergy Documentation:**
- https://github.com/Accelergy-Project/accelergy
- Required for understanding energy/area estimation

**Related Tools:**
- CACTI: Memory/cache modeling (integrated via plugin)
- Graphviz: DAG visualization
- NeuroSim: Emerging memory technologies

---

## AI Assistant Best Practices

**When asked to modify the codebase:**
1. Always check existing patterns in similar files first
2. Preserve unique naming requirements for components
3. Update `__init__.py` files when adding new modules
4. Maintain compatibility with Accelergy (45nm tech node, minimum memory sizes)
5. Follow the abstract → concrete class hierarchy

**When debugging issues:**
1. Check component name uniqueness first
2. Verify memory sizes meet minimums (64×64)
3. Ensure auto_interconnect is set correctly
4. Check that all matmul blocks have assigned memories

**When adding features:**
1. Start with abstract base class if needed
2. Add concrete implementation
3. Update imports in `__init__.py`
4. Create or update example usage
5. Consider CI/CD impact (does example.py still pass?)

**When answering questions:**
1. Reference specific file paths with line numbers where possible
2. Use the Quick Reference section for common tasks
3. Point to example.py for usage patterns
4. Mention known limitations from TODOs if relevant

---

**Last Updated:** 2025-11-18
**Codebase Version:** 0.1 (commit: 9acde4e)
