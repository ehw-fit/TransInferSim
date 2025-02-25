from analyzer.hardware_components.memories.dedicated import DedicatedMemory
from analyzer.hardware_components.memories.shared import SharedMemory
import math

class MatmulArray():
    """Generic class for a matrix multiplication spatial array in a hardware accelerator.

    TODO revise
    Each processing element is thought to be composed of a MAC unit and small local register.

    Args:
        name (str): The name of the matmul array.
        rows (int): The number of rows in the matmul array.
        columns (int): The number of columns in the matmul array.
        data_bitwidth (int): The bitwidth of the data processed by the array.
        cycle_time (float): The time per cycle in seconds (e.g., 7e-9 for 7ns).
        cycles_per_mac (int): The number of cycles required to complete a single MAC operation. (Defaults to 1)
        num_pipeline_stages (int): The number of pipeline stages in the MAC unit. (Defaults to 1)
    """

    def __init__(self, rows: int, columns: int, data_bitwidth: int, cycle_time: float, cycles_per_mac: int = 1, num_pipeline_stages: int = 1, name: str = "spatial_array", parent_component=None):
        assert rows > 0, "Number of rows must be positive."
        assert columns > 0, "Number of columns must be positive."
        assert data_bitwidth > 0, "Data bitwidth must be positive."
        assert cycle_time > 0, "Time required for one cycle must be positive."
        assert cycles_per_mac > 0, "Number of cycles per MAC must be positive."
        assert num_pipeline_stages > 0, "Number of pipeline stages must be positive."
        assert len(name) > 0, "Name must be a non-empty string."
        assert num_pipeline_stages <= cycles_per_mac, "Number of pipeline stages cannot exceed the cycles per MAC operation."

        # TODO ADD POSSIBLE BIAS ADDITION LOGIC
        self.name = name
        self.rows = rows
        self.columns = columns
        self.pes = rows * columns
        self.cycle_time = cycle_time  # Accelerator cycle time
        self.parent_component = parent_component
        self._auto_interconnect_set = False  # Used for auto interconnection feature between other HW components to avoid repated interconnetct

        # Functional characteristics
        self.data_bitwidth = data_bitwidth
        self.cycles_per_mac = cycles_per_mac
        self.num_pipeline_stages = num_pipeline_stages
        self.peak_macs = (self.pes * num_pipeline_stages) / (self.cycle_time * self.cycles_per_mac)        
        self.peak_flops = 2 * self.peak_macs  # Peak throughput

        # Accelergy attributes
        self._area = 0
        self._energy = 0

        # Simulation attributes
        self.is_busy = False
        self.current_operation_event = None

        self.plan = []  # List of operations planned for execution on this unit (populated during simulation analysis)
        self.static_param_memory = None # Memory for storing this matmul's static params if it has any
        self.dynamic_param_memory = None # Memory for storing this matmul's dynamic params if it has any

        self._global_cycles = 0
        self._pes_computational_cycles = 0
        self._pes_idle_cycles = 0
        self._total_flop_computes = 0

        self._accumulator_reads = 0
        self._accumulator_writes = 0

    def __str__(self):
        return (f"<class={self.__class__.__name__} name={self.name}| "
                f"Dimensions: ({self.rows}x{self.columns}) "
                f"Processing Elements: {self.pes} "
                f"Bitwidth: {self.data_bitwidth} "
                f"Peak FLOPs/s: {self.peak_flops:.2e} "
                f"Peak MACs/s: {self.peak_macs:.2e} "
                f"Cycles per MAC: {self.cycles_per_mac} "
                f"Num Pipeline Stages: {self.num_pipeline_stages} "
                f"Cycle time: {self.cycle_time} s>")

    # Attributes for analysis stats
    @property
    def area(self):
        return self._area
    
    @area.setter
    def area(self, value):
        self._area = value

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value):
        self._energy = value

    @property
    def global_cycles(self):
        return self._global_cycles

    @global_cycles.setter
    def global_cycles(self, value):
        self._global_cycles = value

    @property
    def pes_computational_cycles(self):
        return self._pes_computational_cycles

    @pes_computational_cycles.setter
    def pes_computational_cycles(self, value):
        self._pes_computational_cycles = value

    @property
    def pes_idle_cycles(self):
        return self._pes_idle_cycles

    @pes_idle_cycles.setter
    def pes_idle_cycles(self, value):
        self._pes_idle_cycles = value

    @property
    def total_flop_computes(self):
        return self._total_flop_computes

    @total_flop_computes.setter
    def total_flop_computes(self, value):
        self._total_flop_computes = value

    @property
    def accumulator_reads(self):
        return self._accumulator_reads

    @accumulator_reads.setter
    def accumulator_reads(self, value):
        self._accumulator_reads = value

    @property
    def accumulator_writes(self):
        return self._accumulator_writes

    @accumulator_writes.setter
    def accumulator_writes(self, value):
        self._accumulator_writes = value

    # Control methods during simulation analysis
    def assign_static_params_memory(self, memory):
        self.static_param_memory = memory
        
    def assign_dynamic_params_memory(self, memory):
        self.dynamic_param_memory = memory

    # TODO REWORK MEMORY ASSIGNMENT LOGIC?
    def find_and_assign_memories(self):
        self._auto_interconnect_set = True
        accelerator = self.parent_component

        # Get available memories
        dram = accelerator.dram

        # First, check and reset the associated_matmuls and upper memories if they were assigned prior to the simulation.. but auto interconnect should be used (overwrite the setting)
        for m in accelerator.memory_blocks:
            if m.parent_component.auto_interconnect and not m._auto_interconnect_set:
                m.associated_matmuls = []
                m.upper_level_memory = None

        shared_memories = [m for m in accelerator.memory_blocks if isinstance(m, SharedMemory)]
        dedicated_memories = [m for m in accelerator.memory_blocks if isinstance(m, DedicatedMemory) and len(m.associated_matmuls) == 0]

        # TODO check and adjust with more complex mem hierarchies..
        if dedicated_memories:
            # Assign a dedicated memory to this matmul for static parameters
            dedicated_memory = dedicated_memories.pop(0)
            dedicated_memory._auto_interconnect_set = True
            dedicated_memory.add_associated_matmul(self)
            self.assign_static_params_memory(dedicated_memory)

            if shared_memories:
                # Use the first shared memory for dynamic parameters
                shared_memory = shared_memories.pop(-1)
                shared_memory._auto_interconnect_set = True
                self.assign_dynamic_params_memory(shared_memory)
                
                # Set the shared memory as the upper level for the dedicated memory (and vice versa, the dedicated memory as the shared memory's lower level)
                dedicated_memory.set_upper_level_memory(shared_memory)

                # If there are more shared memories, link them in hierarchy
                if shared_memories:
                    next_shared_memory = shared_memories.pop(-1)
                    next_shared_memory._auto_interconnect_set = True
                    shared_memory.set_upper_level_memory(next_shared_memory)

                    # Link remaining shared memories sequentially to the upper levels
                    while shared_memories:
                        upper_memory = shared_memories.pop(-1)
                        upper_memory._auto_interconnect_set = True
                        next_shared_memory.set_upper_level_memory(upper_memory)
                        next_shared_memory = upper_memory

                    # Finally, set DRAM as the upper level for the last shared memory (and vice versa, the last shared memory as DRAM's lower memory)
                    next_shared_memory.set_upper_level_memory(dram)
                else:
                    # If no additional shared memories, set DRAM as the upper level (and vice versa assign lower memory to DRAM)
                    shared_memory.set_upper_level_memory(dram)
            else:
                # If no shared memories, assign dynamic parameters to DRAM (and vice versa assign lower memory to DRAM)
                self.assign_dynamic_params_memory(dram)
                dedicated_memory.set_upper_level_memory(dram)

        elif shared_memories:
            # If only shared memories are available, use the first one for both static and dynamic parameters
            shared_memory = shared_memories.pop(-1)
            shared_memory._auto_interconnect_set = True
            shared_memory.add_associated_matmul(self)
            self.assign_static_params_memory(shared_memory)
            self.assign_dynamic_params_memory(shared_memory)

            # Link the remaining shared memories if available
            if shared_memories:
                next_shared_memory = shared_memories.pop(-1)
                shared_memory.set_upper_level_memory(next_shared_memory)

                # Chain remaining shared memories sequentially
                while shared_memories:
                    upper_memory = shared_memories.pop(-1)
                    upper_memory._auto_interconnect_set = True
                    next_shared_memory.set_upper_level_memory(upper_memory)
                    next_shared_memory = upper_memory

                # Finally, set DRAM as the upper level for the last shared memory (and vice versa, the last shared memory as DRAM's lower memory)
                next_shared_memory.set_upper_level_memory(dram)
            else:
                # If no additional shared memories, set DRAM as the upper level (and vice versa assign lower memory to DRAM)
                shared_memory.set_upper_level_memory(dram)
        else:
            # If no dedicated or shared memories, assign everything to DRAM
            self.assign_static_params_memory(dram)
            self.assign_dynamic_params_memory(dram)
            dram.add_associated_matmul(self)
            dram._auto_interconnect_set = True

    def compute(self, dim_m, dim_k, dim_n):
        self.is_busy = True
        num_mac_operations = dim_m * dim_k * dim_n
        self.total_flop_computes += num_mac_operations * 2

        # Common size of data tensors x cycles per MAC operation
        # Basically these match: (cycles_per_mac_compute * (dim_m * dim_n)) == (num_mac_operations * self.cycles_per_mac)
        cycles_per_mac_compute = dim_k * self.cycles_per_mac

        # Pipeline fill delay added to each pe computation (since it accounts to cycles when some stages are already computing)
        pipeline_latency = self.num_pipeline_stages - 1
        
        # Effective cycles per compute considering the overlap due to pipelining
        effective_cycles_per_compute = math.ceil(cycles_per_mac_compute / self.num_pipeline_stages) + pipeline_latency

        # Total cycles for the entire array to finish the computation
        # Each PE waits for ((dim_m - 1) + (dim_n - 1)) cycles to either get its data for calculating partial results or to wait on other PEs computations to finish
        total_cycles = effective_cycles_per_compute + (dim_m - 1) + (dim_n - 1)
        
        self.accumulator_reads += ((effective_cycles_per_compute - pipeline_latency - 1) * (dim_m * dim_n))  # -1 to account for the first read of 0 partial result
        self.accumulator_writes += ((effective_cycles_per_compute - pipeline_latency) * (dim_m * dim_n))

        # PEs computational cycles (NOTE: only those cycles that PEs actually compute results are accounted for) and idle cycles
        pes_computational_cycles = effective_cycles_per_compute * (dim_m * dim_n)
        self.pes_computational_cycles += pes_computational_cycles
        # PEs idle cycles (total PE cycles - computational cycles)
        pes_idle_cycles = (total_cycles * self.pes) - pes_computational_cycles
        self.pes_idle_cycles += pes_idle_cycles

        assert pes_computational_cycles + pes_idle_cycles == total_cycles * self.pes
        self.global_cycles += total_cycles
        return total_cycles

    def compute_done(self):
        self.current_operation_event = None
        self.is_busy = False

    # Methods used for stats retrieval
    def get_pes_total_cycles(self):
        return self.global_cycles * self.pes

    def calculate_utilization(self):
        """Calculates and returns the utilization based on current stats."""
        return self.pes_computational_cycles / self.get_pes_total_cycles() if self.get_pes_total_cycles() > 0 else 0

    def calculate_throughput(self):
        """Calculates and returns the throughput based on current stats."""
        effective_utilization = self.calculate_utilization()
        return self.peak_flops * effective_utilization

    def get_stats(self):
        """Updates and returns the current statistics."""
        stats = {
            'name': self.name,
            'dimensions': (self.rows, self.columns),
            'PEs': self.pes,
            'pipeline_stages': self.num_pipeline_stages,
            'cycles_per_mac': self.cycles_per_mac,
            'peak_flops': self.peak_flops,
            'peak_macs': self.peak_macs,
            'cycle_time': self.cycle_time,
            'cycles': self.global_cycles,
            'latency': self.global_cycles * self.cycle_time,
            'energy': self.energy * 1e-12,
            'edp_latency': (self.energy * 1e-12) * self.global_cycles * self.cycle_time,
            'edp_cycles': (self.energy * 1e-12) * self.global_cycles,
            'area': self.area,
            'pes_total_cycles': self.get_pes_total_cycles(),
            'pes_computational_cycles': self.pes_computational_cycles,
            'pes_idle_cycles': self.pes_idle_cycles,
            'pes_accumulator_reads': self.accumulator_reads,
            'pes_accumulator_writes': self.accumulator_writes,
            'total_mac_computes': self.total_flop_computes // 2,
            'throughput': self.calculate_throughput(),
            'utilization': self.calculate_utilization()
        }
        return stats

    # Accelergy export methods
    def get_accelergy_description(self):
        return {
            "name": self.name,
            "attributes": {
                "n_instances": self.rows * self.columns,
                "meshX": self.rows,
                "meshY": self.columns
            },
            "local": [
                {
                    "name": "mac",
                    "class": "mac",
                    "attributes": {
                        "datawidth": self.data_bitwidth,
                        "num_pipeline_stages": self.num_pipeline_stages
                    },
                },
                {
                    "name": "buffer",
                    "class": "smartbuffer_RF",
                    "attributes": {
                        "width": 2 * self.data_bitwidth,
                        "depth": 1
                    },
                },
            ]
        }

    def generate_action_counts(self):
        return {
            'name': self.name,
            'local': [
                {
                    'name': 'mac',
                    'action_counts': [
                        {
                            'name': 'mac_random',
                            'counts': self.pes_computational_cycles
                        },
                        {
                            'name': 'idle',
                            'counts': self.pes_idle_cycles
                        }
                    ]
                },
                {
                    'name': 'buffer',
                    'action_counts': [
                        {
                            'name': 'read',
                            'counts': self.accumulator_reads
                        },
                        {
                            'name': 'write',
                            'counts': self.accumulator_writes
                        }
                    ]
                }
            ]
        }