from .matmul import MatmulArray
from .generic_memory import GenericMemory
from analyzer.hardware_components.memories.offchip import OffChipMemory
import math
import yaml
import re

# Definition of SafeDumper for proper generation of quoted strings as requiered by accelergy (technology and voltage values must be quoted)
def quoted_str_presenter(dumper, data):
    if isinstance(data, str):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, quoted_str_presenter, Dumper=yaml.SafeDumper)


class GenericAccelerator():
    """Generic class for a hardware accelerator.

    This class represents a generic hardware accelerator that can contains
    multiple computational and memory blocks, allowing for creation of
    a hierarchical structure.
    
    It also allows to export the accelerator description to yaml format used by Accelergy for energy and area estimation.

    Args:
        cycle_time (float): The cycle time of the accelerator.
        name (str): The name of the accelerator.
        tech_node (str): The technology node of the accelerator.
        dram (OffChipMemory): The off-chip memory used by the accelerator.
    """

    def __init__(self, cycle_time: float = 5e-09, name: str = "accelerator", tech_node: str = "45nm", auto_interconnect: bool = True, dram: OffChipMemory = None):
        assert dram, "DRAM must be provided for the accelerator!"
        self.name = name
        self.tech_node = tech_node
        self.cycle_time = cycle_time
        self.frequency = math.ceil(1 / cycle_time / 1e6)
        self.matmul_blocks = []  # List of on-chip computational spatial array components
        self.memory_blocks = []  # List of on-chip memory subcomponents like SRAMs, caches, etc.
        self.dram = dram
        self.auto_interconnect = auto_interconnect
        self.comp_names_map = {self.dram.name: self.dram}  # Map of component names to their respective object (Used for mapping the component to Accelergy stats)
        self.dram.parent_component = self
        self.global_cycles = 0  # Global cycle count for discrete simulation analysis
        self._analysis_done = False
        
        # Accelergy attributes
        self._area = 0
        self._energy = 0

        # Set to keep track of all block subcomponent names to ensure uniqueness
        self._names = set()

    def __str__(self):
        matmul_block_str = "\n".join([f"    {str(block)}" for block in self.matmul_blocks])
        memory_block_str = "\n".join([f"    {str(block)}" for block in self.memory_blocks])
        return (f"<class={self.__class__.__name__} name={self.name}| "
                f"Frequency: {self.frequency} MHz "
                f"Cycle Time: {self.cycle_time} s "
                f"Matmul Blocks: {len(self.matmul_blocks)} "
                f"Memory Blocks: {len(self.memory_blocks)} "
                f"DRAM: {str(self.dram)}\n"
                f"  Matmul Blocks:\n{matmul_block_str}\n"
                f"  Memory Blocks:\n{memory_block_str}\n>")

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
    def analysis_done(self):
        """Getter for the analysis done flag."""
        return self._analysis_done

    @analysis_done.setter
    def analysis_done(self, value: bool):
        """Setter for the analysis done flag."""
        if not isinstance(value, bool):
            raise ValueError("Value for analysis_done flag must be a boolean!")
        self._analysis_done = value

    # Accelerator component addition methods
    def add_matmul_block(self, block: MatmulArray):
        """Adds a matmul spatial array block to the accelerator.

        Args:
            block: A matmul spatial array block (instance of MatmulArray).
        """
        assert isinstance(block, MatmulArray), f"The inserted object '{block}' is not a MatmulArray instance."
        self.check_valid_block_name(block)  # Check for duplicate block names within the design
        block.parent_component = self
        self.comp_names_map[block.name] = block
        self.matmul_blocks.append(block)
    
    def add_memory_block(self, block: GenericMemory):
        """Adds a memory block to the accelerator.

        Args:
            block: A memory block (instance of GenericMemory).
        """
        assert isinstance(block, GenericMemory), f"The inserted object '{block}' is not a GenericMemory instance."
        self.check_valid_block_name(block)  # Check for duplicate block names within the design
        block.parent_component = self
        self.comp_names_map[block.name] = block
        self.memory_blocks.append(block)

    def check_valid_block_name(self, block):
        """
        Recursively collects names of all blocks in the design, ensuring all names are unique.
        
        Args:
            block: Computational or memory block to be added to the accelerator hierarchy.
        
        Raises:
            ValueError: If duplicate block names are found.
        """
        if block.name in self._names:
            raise ValueError(f"Duplicate HW component name detected: {block.name}")
        self._names.add(block.name)

    # Discrete simulation methods
    def set_replacement_strategy(self, replacement_strategy):
        for mem in self.memory_blocks:
            mem.set_replacement_strategy(replacement_strategy)
        
    def calculate_avg_throughput(self):
        """Calculates and returns the average throughput for the accelerator's computational blocks."""
        if not self.matmul_blocks:
            return 0

        # Calculate the average throughput of all computational blocks
        total_throughput = sum(block.calculate_throughput() for block in self.matmul_blocks)
        avg_throughput = total_throughput / len(self.matmul_blocks)
        return avg_throughput

    def calculate_min_utilization(self):
        """Calculates and returns the minimum utilization for the accelerator."""
        utilizations = [block.calculate_utilization() for block in self.matmul_blocks if self.global_cycles > 0]
        return min(utilizations) if utilizations else 0

    def calculate_max_utilization(self):
        """Calculates and returns the maximum utilization for the accelerator."""
        utilizations = [block.calculate_utilization() for block in self.matmul_blocks if self.global_cycles > 0]
        return max(utilizations) if utilizations else 0

    def update_global_cycles(self, cycles):
        """Updates the global cycle count for the accelerator."""
        self.global_cycles += cycles

    # Statistics report methods
    def get_statistics(self, log_mem_contents=False):
        """Returns statistics for the accelerator."""
        stats = {
            'name': self.name,
            'frequency': self.frequency,
            'cycle_time': self.cycle_time,
            'memories': len(self.memory_blocks),
            'spatial_arrays': len(self.matmul_blocks),
            'global_cycles': self.global_cycles,
            'latency': self.global_cycles * self.cycle_time,
            'energy': self.energy * 1e-12,
            'edp_latency': (self.energy * 1e-12) * self.global_cycles * self.cycle_time,
            'edp_cycles': (self.energy * 1e-12) * self.global_cycles,
            'area': self.area,
            'avg_throughput': self.calculate_avg_throughput(),
            'min_utilization': self.calculate_min_utilization(),
            'max_utilization': self.calculate_max_utilization(),
            'memory_stats': [mem.get_stats(log_mem_contents) for mem in self.memory_blocks],
            'compute_stats': [comp.get_stats() for comp in self.matmul_blocks],
            'dram_stats':  self.dram.get_stats(log_mem_contents)
        }
        return stats

    @staticmethod
    def format_flops(value, precision):
        """Formats FLOPs into a more human-readable format with appropriate units."""
        if value >= 1e12:
            return f"{value / 1e12:.{precision}f} TFLOPs"
        elif value >= 1e9:
            return f"{value / 1e9:.{precision}f} GFLOPs"
        elif value >= 1e6:
            return f"{value / 1e6:.{precision}f} MFLOPs"
        elif value >= 1e3:
            return f"{value / 1e3:.{precision}f} KFLOPs"
        else:
            return f"{value:.{precision}f} FLOPs"

    @staticmethod
    def format_macs(value, precision):
        """Formats MACs into a more human-readable format with appropriate units."""
        if value >= 1e12:
            return f"{value / 1e12:.{precision}f} TMACs"
        elif value >= 1e9:
            return f"{value / 1e9:.{precision}f} GMACs"
        elif value >= 1e6:
            return f"{value / 1e6:.{precision}f} MMACs"
        elif value >= 1e3:
            return f"{value / 1e3:.{precision}f} KMACs"
        else:
            return f"{value:.{precision}f} MACs"

    @staticmethod
    def pretty_print_stats(stats, verbose=True, file_path=None, precision=6):
        output = []

        output.append("STATS AFTER SIMULATION ANALYSIS")
        output.append("–––––––––––––––––––––––––––––––")
        output.append(f"Accelerator Name: {stats['name']}")
        output.append(f"Frequency: {stats['frequency']} Hz")
        output.append(f"Cycle Time: {stats['cycle_time']:.{precision}e} s")
        output.append(f"Number of Memory Blocks: {stats['memories']}")
        output.append(f"Number of Matmul Blocks: {stats['spatial_arrays']}")
        output.append(f"Global Cycles: {stats['global_cycles']:.{precision}e}")
        output.append(f"Global Latency: {stats['latency']:.{precision}} s")
        output.append(f"Total Energy: {stats['energy']:.{precision}} J")
        output.append(f"Total EDP (cycles): {stats['edp_cycles']:.{precision}} J.cycles")
        output.append(f"Total EDP (latency): {stats['edp_latency']:.{precision}} J.s")
        output.append(f"Total Area: {float(stats['area']):.{precision}} um^2")
        output.append(f"Average Throughput: {GenericAccelerator.format_flops(stats['avg_throughput'], precision=precision)}/cycle")
        output.append(f"Minimum Utilization: {stats['min_utilization']:.2%}")
        output.append(f"Maximum Utilization: {stats['max_utilization']:.2%}")

        output.append("\nMemory Block Statistics:")
        for mem_stat in stats['memory_stats']:
            output.append(f"  Name: {mem_stat['name']}")
            output.append(f"    Size: {mem_stat['size']}")
            output.append(f"    Width: {mem_stat['width']}")
            output.append(f"    Depth: {mem_stat['depth']}")
            output.append(f"    Word Size: {mem_stat['word_size']}")
            output.append(f"    Ports: {mem_stat['ports']}")
            output.append(f"    Bus Bitwidth: {mem_stat['bus_bitwidth']}")
            output.append(f"    Bandwidth per port: {mem_stat['bandwidth_per_port'] / 1e9:.{precision}e} Gbps")
            output.append(f"    Total Bandwidth: {mem_stat['total_bandwidth'] / 1e9:.{precision}e} Gbps")
            output.append(f"    Current Usage: {mem_stat['current_usage']}")
            output.append(f"    Utilized Capacity: {(mem_stat['current_usage'] / mem_stat['size']):.2%}")
            output.append(f"    Action Latency: {mem_stat['action_latency']:.{precision}e}")
            output.append(f"    Cycles per Access: {mem_stat['cycles_per_access']}")
            output.append(f"    Data Read Count: {mem_stat['data_read_count']}")
            output.append(f"    Memory Block Read Count: {mem_stat['mem_read_count']}")
            output.append(f"    Data Write Count: {mem_stat['data_write_count']}")
            output.append(f"    Memory Block Write Count: {mem_stat['mem_write_count']}")
            output.append(f"    Word Read Count: {mem_stat['word_read_count']}")
            output.append(f"    Word Write Count: {mem_stat['word_write_count']}")
            output.append(f"    Cache Miss Count: {mem_stat['cache_miss_count']}")
            output.append(f"    Cache Miss Rate: {mem_stat['cache_miss_rate']}")
            output.append(f"    Cache Hit Rate: {mem_stat['cache_hit_rate']}")
            output.append(f"    Fragmented Bits: {mem_stat['fragmented_bits']}")
            output.append(f"    Replacement Strategy: {mem_stat['replacement_strategy']}")
            output.append(f"    Energy: {mem_stat['energy']:.{precision}} J")
            output.append(f"    Area: {float(mem_stat['area']):.{precision}} um^2")
            output.append("    Ports Utilization:")
            for port_id, port_util in mem_stat['ports_utilization'].items():
                output.append(f"      Port ID: {port_id}")
                output.append(f"        Global Cycles: {port_util['global_cycles']}")
                output.append(f"        Idle Cycles: {port_util['idle_cycles']}")
                output.append(f"        Utilization: {port_util['utilization']}")
            if mem_stat['contents'] is not None:
                output.append("    Contents:")
                for data_id, content in mem_stat['contents'].items():
                    output.append(f"      Data ID: {data_id}")
                    output.append(f"        Data Amount: {content['data_amount']}")
                    output.append(f"        Data Read Count: {content['data_read_count']}")
                    output.append(f"        Memory Read Count: {content['mem_read_count']}")
                    output.append(f"        Data Write Count: {content['data_write_count']}")
                    output.append(f"        Memory Write Count: {content['mem_write_count']}")
                    output.append(f"        Word Amount: {content['word_amount']}")
                    output.append(f"        Word Read Count: {content['word_read_count']}")
                    output.append(f"        Word Write Count: {content['word_write_count']}")
                    output.append(f"        Data Bitwidth: {content['data_bitwidth']}")
                    output.append(f"        Bit Packing Used: {content['data_bit_packing']}")
                    output.append(f"        Fragmented Bits in Words: {content['fragmented_bits']}")
                    output.append(f"        Data Cache Misses: {content['cache_miss_count']}")

        if 'dram_stats' in stats:
            output.append("\nDRAM Statistics:")
            dram_stat = stats['dram_stats']
            output.append(f"  Name: {dram_stat['name']}")
            output.append(f"    Size: -")
            output.append(f"    Word Size: {dram_stat['word_size']}")
            output.append(f"    Ports: {dram_stat['ports']}")
            output.append(f"    Bus Bitwidth: {dram_stat['bus_bitwidth']}")
            output.append(f"    Bandwidth per port: {dram_stat['bandwidth_per_port'] / 1e9:.{precision}e} Gbps")
            output.append(f"    Total Bandwidth: {dram_stat['total_bandwidth'] / 1e9:.{precision}e} Gbps")
            output.append(f"    Current Usage: -")
            output.append(f"    Utilized Capacity: -")
            output.append(f"    Action Latency: {dram_stat['action_latency']:.{precision}e}")
            output.append(f"    Cycles per Access: {dram_stat['cycles_per_access']}")
            output.append(f"    Data Read Count: {dram_stat['data_read_count']}")
            output.append(f"    Memory Block Read Count: {dram_stat['mem_read_count']}")
            output.append(f"    Data Write Count: {dram_stat['data_write_count']}")
            output.append(f"    Memory Block Write Count: {dram_stat['mem_write_count']}")
            output.append(f"    Word Read Count: {dram_stat['word_read_count']}")
            output.append(f"    Word Write Count: {dram_stat['word_write_count']}")
            output.append(f"    Fragmented Bits: -")
            output.append(f"    Energy: {dram_stat['energy']:.{precision}} J")
            output.append(f"    Area: {dram_stat['area']} um^2")
            output.append("    Ports Utilization:")
            for port_id, port_util in dram_stat['ports_utilization'].items():
                output.append(f"      Port ID: {port_id}")
                output.append(f"        Global Cycles: {port_util['global_cycles']}")
                output.append(f"        Idle Cycles: {port_util['idle_cycles']}")
                output.append(f"        Utilization: {port_util['utilization']}")
            if dram_stat['contents'] is not None:
                output.append("    Contents:")
                for data_id, content in dram_stat['contents'].items():
                    output.append(f"      Data ID: {data_id}")
                    output.append(f"        Data Amount: {content['data_amount']}")
                    output.append(f"        Data Read Count: {content['data_read_count']}")
                    output.append(f"        Memory Read Count: {content['mem_read_count']}")
                    output.append(f"        Data Write Count: {content['data_write_count']}")
                    output.append(f"        Memory Write Count: {content['mem_write_count']}")
                    output.append(f"        Word Amount: {content['word_amount']}")
                    output.append(f"        Word Read Count: {content['word_read_count']}")
                    output.append(f"        Word Write Count: {content['word_write_count']}")
                    output.append(f"        Data Bitwidth: {content['data_bitwidth']}")
                    output.append(f"        Bit Packing Used: {content['data_bit_packing']}")
                    output.append(f"        Fragmented Bits in Words: {content['fragmented_bits']}")

        output.append("\nMatmul Block Statistics:")
        for comp_stat in stats['compute_stats']:
            output.append(f"  Name: {comp_stat['name']}")
            output.append(f"    Dimensions: {comp_stat['dimensions']}")
            output.append(f"    PEs: {comp_stat['PEs']}")
            output.append(f"    Pipeline Stages: {comp_stat['pipeline_stages']}")
            output.append(f"    Cycles per MAC: {comp_stat['cycles_per_mac']}")
            output.append(f"    Peak FLOPs: {GenericAccelerator.format_flops(comp_stat['peak_flops'], precision=precision)}")
            output.append(f"    Peak MACs: {GenericAccelerator.format_macs(comp_stat['peak_macs'], precision=precision)}")
            output.append(f"    Operational Cycles: {comp_stat['cycles']:.{precision}e}")
            output.append(f"    Latency: {(comp_stat['latency']):.{precision}} s")
            output.append(f"    Energy: {comp_stat['energy']:.{precision}} J")
            output.append(f"    EDP (cycles): {stats['edp_cycles']:.{precision}} J.cycles")
            output.append(f"    EDP (latency): {stats['edp_latency']:.{precision}} J.s")            
            output.append(f"    Area: {float(comp_stat['area']):.{precision}} um^2")
            output.append(f"    PEs Total Cycles: {comp_stat['pes_total_cycles']:.{precision}e}")
            output.append(f"    PEs Computational Cycles: {comp_stat['pes_computational_cycles']:.{precision}e}")
            output.append(f"    PEs Idle Cycles: {comp_stat['pes_idle_cycles']:.{precision}e}")
            output.append(f"    PEs Accumulator Reads: {comp_stat['pes_accumulator_reads']:.{precision}e}")
            output.append(f"    PEs Accumulator Writes: {comp_stat['pes_accumulator_writes']:.{precision}e}")
            output.append(f"    Total MAC Computes: {comp_stat['total_mac_computes']}")
            output.append(f"    Throughput: {GenericAccelerator.format_flops(comp_stat['throughput'], precision=precision)}")
            output.append(f"    Utilization: {(comp_stat['utilization']):.{precision}%}")

        if verbose:
            for line in output:
                print(line)

        if file_path:
            with open(file_path, "w") as file:
                for line in output:
                    file.write(line + "\n")

    # Accelergy export methods
    def get_accelergy_description(self, out_fname: str = None):
        description = {
            "architecture": {
                "version": 0.4,
                "subtree": [
                    {
                        "name": "system",
                        "attributes": {
                            "technology": self.tech_node,
                            "global_cycle_seconds": self.cycle_time,
                            "voltage": "1V",
                        },
                        "local": [],
                        "subtree": [{
                                "name": self.name,
                                "local": [],
                                "subtree": []
                            }
                        ],
                    }
                ]
            }
        }
        description["architecture"]["subtree"][0]["local"].append(self.dram.get_accelergy_description())
        
        for block in self.memory_blocks:
            description["architecture"]["subtree"][0]["subtree"][0]["local"].append(block.get_accelergy_description())

        for block in self.matmul_blocks:
            description["architecture"]["subtree"][0]["subtree"][0]["subtree"].append(block.get_accelergy_description())

        yaml_content = yaml.dump(description, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper)
        # To ensure the quotes are removed from keys but not the values (quotes at strings needed for correct Accelergy parsing)
        yaml_content = re.sub(r'(\s|^)(\'|")([a-zA-Z0-9_]+)(\'|"):', r'\1\3:', yaml_content)

        if out_fname:
            with open(f"{out_fname}.yaml", 'w') as file:
                file.write(yaml_content)
        else:
            return yaml_content

    def generate_action_counts(self, out_fname: str = None):
        action_counts = {
            'action_counts': {
                "version": 0.4,
                "subtree": [
                    {
                        "name": "system",
                        "local": [],
                        "subtree": [{
                                "name": self.name,
                                "local": [],
                                "subtree": []
                            }
                        ],
                    }
                ]
            }
        }
        action_counts["action_counts"]["subtree"][0]["local"].append(self.dram.generate_action_counts())
        
        for block in self.memory_blocks:
            action_counts["action_counts"]["subtree"][0]["subtree"][0]["local"].append(block.generate_action_counts())

        for block in self.matmul_blocks:
            action_counts["action_counts"]["subtree"][0]["subtree"][0]["subtree"].append(block.generate_action_counts())

        # Convert dictionary to YAML
        yaml_content = yaml.dump(action_counts, default_flow_style=False, sort_keys=False)

        if out_fname:
            with open(f"{out_fname}.yaml", 'w') as file:
                file.write(yaml_content)
        else:
            return yaml_content
