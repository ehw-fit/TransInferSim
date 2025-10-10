from analyzer.core.hardware.generic_memory import GenericMemory
import math


class SharedMemory(GenericMemory):
    def __init__(self, name: str, width: int, depth: int, action_latency: float, bus_bitwidth: int, cycle_time: float, word_size: int, power_gating: bool = False, banks: int = 1, ports: int = 2, replacement_strategy="random", parent_component = None):
        super().__init__(width=width, depth=depth, bus_bitwidth=bus_bitwidth, action_latency=action_latency, cycle_time=cycle_time, word_size=word_size, ports=ports, name=name, replacement_strategy=replacement_strategy, parent_component=parent_component, accelergy_class="smartbuffer_SRAM")
        assert banks > 0, f"Banks must be a positive integer for {self.__class__.__name__}."
        self.power_gating = power_gating  # TODO Currently does nothing
        self.banks = banks  # TODO Currently does nothing
        # TODO add more fine-grained support later

    def __str__(self):
        base_str = super().__str__()
        return (f"{base_str[:-1]} Power gating: {self.power_gating} "
                f"Banks: {self.banks}>")
    
    # Accelergy export methods
    def get_accelergy_description(self):
        # TODO ADD ALL NECESSARY MEMORY TYPE PARAMS
        return {
            "name": self.name,
            "class": self.accelergy_class,
            "attributes": {
                "depth": self.depth,
                "width": self.width,
                "block_size": self.row_words,  # Number of words per row as per Accelergy
                "cluster_size": 1,
                "n_banks": self.banks,
                "datawidth": self.word_size,
                "read_bandwidth": math.ceil(self.bus_bitwidth / self.word_size),
                "write_bandwidth": math.ceil(self.bus_bitwidth / self.word_size),
                "n_rdwr_ports": self.ports,
                "action_latency_cycles": self.mem_access_cycles
            }
        }
