# buffer_stack.py
from typing import Any
import numpy as np
from analyzer.core.hardware import GenericMemory     # adjust the import to your package layout


class BufferStack(GenericMemory):
    """Systolic-array streaming buffer bank.

    Args:
        num_buffers (int): Number of independent FIFOs in the bank (typically one per PE row or column). Equals `depth` for generic memory instantiation.
        buffer_length (int): Depth of each buffer FIFO, (in elements per FIFO).
        is_it_row_buffer (bool): Specification whether the buffer represents a row buffer or column buffer (for internal data organization).
        element_size (int): Width of each element inside the buffer, (in bits per element). (Defaults to 8)
        name (str): Name identifier for the buffer bank. (Defaults to "buffer_stack")
        **kwargs: Additional keyword args forwarded to `GenericMemory`.
    """

    def __init__(self, num_buffers: int, bus_bitwidth: int, action_latency: float, cycle_time: float, buffer_length: int, is_it_row_buffer: bool, power_gating: bool = False, element_size: int = 8, name: str = "buffer_stack", replacement_strategy = "fifo", **kwargs: Any):
        # Map to GenericMemory semantics
        self.buffer_length = buffer_length
        self.num_buffers  = num_buffers
        self.power_gating = power_gating  # TODO Currently does nothing
        width  = buffer_length * element_size
        self._type = "row" if is_it_row_buffer else "column"
        accelergy_cls  = kwargs.pop("accelergy_class", "smartbuffer_RF")

        super().__init__(
            width=width,
            depth=self.num_buffers,
            bus_bitwidth=bus_bitwidth,
            action_latency=action_latency,
            cycle_time=cycle_time,
            ports=1,
            word_size=element_size,
            banks=1,
            name=name,
            replacement_strategy=replacement_strategy,
            parent_component=kwargs.pop("parent_component", None),
            accelergy_class=accelergy_cls,
        )

    def __str__(self):
        return (f"<class={self.__class__.__name__} name={self.name}| "
                f"Total: {self.num_buffers * self.buffer_length} elements/words ({self.num_buffers} buffers × {self.buffer_length} w/buf) "
                f"Size: {self.size} b "
                f"Word-size: {self.word_size} b "
                f"Power gating: {self.power_gating}>")

    # Accelergy export methods
    def get_accelergy_description(self):
        # TODO!!
        return {
            "name": self.name,
            "class": self.accelergy_class,
            "attributes": {
                "depth": self.depth,
                "width": self.width,
                "datawidth": self.word_size,
                "action_latency_cycles": self.mem_access_cycles,
                "n_banks": self.banks,
                "has_power_gating": self.power_gating
            }
        }
