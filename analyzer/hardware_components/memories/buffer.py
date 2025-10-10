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

    def __init__(self, num_buffers: int, buffer_length: int, is_it_row_buffer: bool, element_size: int = 8, name: str = "buffer_stack", **kwargs: Any):
        # Map to GenericMemory semantics
        self.buffer_length = buffer_length
        self.num_buffers  = num_buffers
        width  = buffer_length * element_size
        self._type = "row" if is_it_row_buffer else "column"

        # GenericMemory still insists on a bus, cycle time, etc., passing dummy but legal values so all asserts pass.
        dummy_bus      = max(element_size, kwargs.pop("bus_bitwidth", element_size))
        dummy_latency  = kwargs.pop("action_latency", 1)
        dummy_cycle    = kwargs.pop("cycle_time", 1)
        dummy_ports    = kwargs.pop("ports", 1)          # ≥ 1 to satisfy the assert
        dummy_repl     = kwargs.pop("replacement_strategy", "fifo")
        accelergy_cls  = kwargs.pop("accelergy_class", "smartbuffer_RF")

        super().__init__(
            width=width,
            depth=self.num_buffers,
            bus_bitwidth=dummy_bus,
            action_latency=dummy_latency,
            cycle_time=dummy_cycle,
            ports=dummy_ports,
            word_size=element_size,
            name=name,
            replacement_strategy=dummy_repl,
            parent_component=kwargs.pop("parent_component", None),
            accelergy_class=accelergy_cls,
        )

    def __str__(self):
        return (f"<class={self.__class__.__name__} name={self.name}| "
                f"Total: {self.num_buffers * self.buffer_length} elements/words ({self.num_buffers} buffers × {self.buffer_length} w/buf) "
                f"Size: {self.size} b "
                f"Word-size: {self.word_size} b>")

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
                "action_latency_cycles": self.mem_access_cycles
            }
        }
