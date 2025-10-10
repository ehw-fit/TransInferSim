from abc import ABC, abstractmethod
from analyzer.hardware_components.memories.offchip import OffChipMemory
import math


""" Abstract base class for Event """
class AbstractEvent(ABC):
    def __init__(self, verbose: bool = False):
        self._event_time = 0
        self.verbose = verbose

    @abstractmethod
    def execute(self, scheduler):
        pass

    def __lt__(self, other):
        return self.event_time < other.event_time

    @property
    def event_time(self):
        return self._event_time
    
    @event_time.setter
    def event_time(self, value):
        self._event_time = value


class MemoryThrashingError(Exception):
    """Custom exception for memory thrashing issues."""
    pass


""" MEMORY EVENTS """
class MemoryReadEvent(AbstractEvent):
    def __init__(self, data_id, data_category, data_bitwidth, tensor_shape, tile_shape, offset, memory, final_operation = None, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.data_id = data_id
        self.data_category = data_category
        self.data_bitwidth = data_bitwidth
        self.tensor_shape = tensor_shape
        self.tile_shape = tile_shape
        self.offset = offset
        self.memory = memory
        self.memory_port = None
        self.final_operation = final_operation
        self.next_event = next_event

    def __str__(self):
        return f"MemoryReadEvent(data_id={self.data_id}, data_category={self.data_category}, data_bitwidth={self.data_bitwidth}, tensor_shape={self.tensor_shape}, tile_shape={self.tile_shape}, offset={self.offset}, memory={self.memory.name}, final_operation={self.final_operation}, next_event={self.next_event})"

    def execute(self, scheduler):
        # Check if the data is already being fetched by another event
        if self.data_id in self.memory.fetch_locks:
            # Reschedule this event for after the fetch completes (the fetch event must reschedule this event to a time after the fetch completes)
            self.memory.fetch_locks[self.data_id].append(self)
            if self.verbose:
                print(f"Fetch in progress for {self.data_id} from {self.memory.name}. Adding read to wait list.")
            return

        if not self.memory.available_ports:
            self.memory.pending_actions.append(self)
            if self.verbose:
                print(f"Memory {self.memory.name} ports are busy. Rescheduling read of {self.data_id} to pending actions.")
            return
        self.memory_port = self.memory.get_available_port(scheduler.deterministic_key)

        success, read_info = self.memory.read(read_mode="read_tile", data_id=self.data_id, data_category=self.data_category, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.memory_port, tile_shape=self.tile_shape, offset=self.offset)
        if success:
            cycles = read_info
            self.memory.update_per_ports_cycles(self.memory_port, cycles, update_idle=False)
            if self.verbose:
                print(f"Read {self.data_id} from {self.memory.name} in {cycles} cycles and in global time {self.event_time}.")
            scheduler.schedule_event(self.event_time + cycles, MemoryReadCompleteEvent(self.data_id, self.data_bitwidth, self.memory, self.memory_port, self.tensor_shape, self.tile_shape, self.offset, self.final_operation, cycles, self.next_event, verbose=self.verbose))
        else:
            missing_data, missing_data_offset = read_info

            # Free the port
            self.memory.available_ports.append(self.memory_port)
            self.memory_port = None
            while self.memory.pending_actions:
                memory_op = self.memory.pending_actions.pop(0)
                scheduler.schedule_event(self.event_time, memory_op)

            mem_found, mem_miss_data = self.memory.query_lower_memories(read_mode="read_tile", data_id=self.data_id, tile_shape=missing_data, offset=missing_data_offset)
            missing_data, missing_data_offset = mem_miss_data

            if mem_found:
                scheduler.schedule_event(self.event_time, MemoryFetchEvent(self.data_id, self.data_category, self.data_bitwidth, self.tensor_shape, missing_data, missing_data_offset, mem_found, mem_found.upper_level_memory, self.final_operation, self, verbose=self.verbose))
            # If not recursively present in any lower memory, try accessing the data from higher-level memory
            else:
                if self.memory.upper_level_memory:
                    if self.verbose:
                        print(f"Scheduled fetch of {self.data_id} from {self.memory.upper_level_memory.name} to {self.memory.name}.")
                    scheduler.schedule_event(self.event_time, MemoryFetchEvent(self.data_id, self.data_category, self.data_bitwidth, self.tensor_shape, missing_data, missing_data_offset, self.memory.upper_level_memory, self.memory, self.final_operation, self, verbose=self.verbose))
                else:
                    if self.verbose:
                        print(f"Failed to fetch {self.data_id} from {self.memory.name}'s upper memory - no upper memory found containing this data.")


class MemoryReadCompleteEvent(AbstractEvent):
    def __init__(self, data_id, data_bitwidth, memory, memory_port, tensor_shape, tile_shape, offset, final_operation, cycles, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.data_id = data_id
        self.data_bitwidth = data_bitwidth
        self.memory = memory
        self.memory_port = memory_port
        self.tensor_shape = tensor_shape
        self.tile_shape = tile_shape
        self.offset = offset
        self.final_operation = final_operation
        self.cycles = cycles
        self.next_event = next_event

    def __str__(self):
        return f"MemoryReadCompleteEvent(data_id={self.data_id}, memory={self.memory.name}, memory_port={self.memory_port}, tensor_shape={self.tensor_shape}, tile_shape={self.tile_shape}, offset={self.offset}, final_operation={self.final_operation}, cycles={self.cycles}, next_event={self.next_event})"

    def execute(self, scheduler):
        self.memory.available_ports.append(self.memory_port)
        self.memory_port = None
        while self.memory.pending_actions:
            memory_op = self.memory.pending_actions.pop(0)
            scheduler.schedule_event(self.event_time, memory_op)

        if len(self.memory.available_ports) == self.memory.ports:
            self.memory.synchronize_per_ports_cycles()

        if self.verbose:
            print(f"Completed read of {self.data_id} from {self.memory.name}.")
        if self.next_event:
            scheduler.schedule_event(self.event_time, self.next_event)
            


class MemoryFetchEvent(AbstractEvent):
    def __init__(self, data_id, data_category, data_bitwidth, tensor_shape, tile_shape, offset, src_memory, dest_memory, final_operation, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.data_id = data_id
        self.data_category = data_category
        self.data_bitwidth = data_bitwidth
        self.tensor_shape = tensor_shape
        self.tile_shape = tile_shape
        self.offset = offset
        self.src_memory = src_memory
        self.src_port = None
        self.dest_memory = dest_memory
        self.dest_port = None
        self.final_operation = final_operation
        self.next_event = next_event

    def __str__(self):
        return f"MemoryFetchEvent(data_id={self.data_id}, data_category={self.data_category}, data_bitwidth={self.data_bitwidth}, tensor_shape={self.tensor_shape}, tile_shape={self.tile_shape}, offset={self.offset}, src_memory={self.src_memory.name}, dest_memory={self.dest_memory.name}, final_operation={self.final_operation}, next_event={self.next_event})"

    def execute(self, scheduler):
        if not self.src_memory.available_ports or not self.dest_memory.available_ports:
            self.dest_memory.pending_actions.append(self)
            if self.verbose:
                print(f"Ports of memories (src) {self.src_memory.name} and (dest) {self.dest_memory.name} are busy. Rescheduling fetch of {self.data_id} to pending actions.")
            return
        self.src_port = self.src_memory.get_available_port(scheduler.deterministic_key)
        self.dest_port = self.dest_memory.get_available_port(scheduler.deterministic_key)

        read_success, read_info = self.src_memory.read(read_mode="read_tile", data_id=self.data_id, data_category=self.data_category, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.src_port, tile_shape=self.tile_shape, offset=self.offset)
        
        if self.src_memory.detect_memory_thrashing(self.data_id, self.tensor_shape, self.tile_shape, self.offset):
            tensor_info_list = [f"Data ID: {k}, Tensor Shape: {v['tensor_shape']}, Tile Shape: {v['tile_shape']}, Offset: {v['offset']}" for k, v in self.src_memory.last_reads.items()]
            print(f"Memory thrashing detected: All tensors ({'; '.join(tensor_info_list)}) to be fetched need to free up space, but that would cause an infinite loop of free and fetch.")
            print(f"The current memory size for {self.dest_memory.name} of {self.dest_memory.size} words may be too small.")
            raise MemoryThrashingError("Memory thrashing detected and simulation aborted.")

        if read_success:
            read_cycles = read_info
            write_success, write_info = self.dest_memory.write(write_mode="write_tile", data_id=self.data_id, data_category=self.data_category, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.dest_port, tensors_needed_info=scheduler.execution_graph.tensors_needed, tile_shape=self.tile_shape, offset=self.offset, deterministic_key=scheduler.deterministic_key, verbose=self.verbose)
            if write_success:
                #write_cycles = 0  # Latency already counted for read
                # TODO VALIDATE PORTS CYCLES
                # Update action cycles during read/write
                self.src_memory.update_per_ports_cycles(self.src_port, read_cycles, update_idle=False)
                self.dest_memory.update_per_ports_cycles(self.dest_port, read_cycles, update_idle=False)
                # Update idle cycles during read/write
                self.src_memory.update_per_ports_cycles(self.src_port, read_cycles, update_idle=True)
                self.dest_memory.update_per_ports_cycles(self.dest_port, read_cycles, update_idle=True)
                if self.verbose:
                    print(f"Fetched {self.data_id} from {self.src_memory.name} to {self.dest_memory.name} in {read_cycles} cycles and in global time {self.event_time + read_cycles}.")
                scheduler.schedule_event(self.event_time + read_cycles, MemoryFetchCompleteEvent(self.data_id, self.data_bitwidth, self.src_memory, self.src_port, self.dest_memory, self.dest_port, self.tensor_shape, self.tile_shape, self.offset, self.final_operation, read_cycles, self.next_event, verbose=self.verbose))
            else:  # Read from upper memory was successful, but write to lower memory failed because of not enough space → we thus first need to free data in the lower memory (i.e. do a write-back to upper memory and then fetch)
                # Free the ports
                self.src_memory.available_ports.append(self.src_port)
                self.src_port = None
                while self.src_memory.pending_actions:
                    src_memory_op = self.src_memory.pending_actions.pop(0)
                    scheduler.schedule_event(self.event_time, src_memory_op)

                self.dest_memory.available_ports.append(self.dest_port)
                self.dest_port = None
                while self.dest_memory.pending_actions:
                    dest_memory_op = self.dest_memory.pending_actions.pop(0)
                    scheduler.schedule_event(self.event_time, dest_memory_op)

                # Check if the operation could not proceed because different write-back/fetch operation is currently in action for the chosen data id
                if write_info is None:
                    # Check for possible memory thrashing -> loop where two data tensors need space.. but at the same time cannot free each other
                    tensors_with_data = {k for k, v in self.dest_memory.contents.items() if v['word_amount'] > 0}  # All data ids present in memory
                    locked_tensors = self.dest_memory.fetch_locks.keys()  # All currently fetched data ids from the memory
                    if tensors_with_data == locked_tensors or (self.data_id in locked_tensors and len(tensors_with_data) == 1):
                        tensor_info_list = [f"Data ID: {t}, Data Bitwidth: {self.dest_memory.contents[t]['data_bitwidth']}, Tensor Shape: {self.dest_memory.contents[t]['presence_matrix'].shape}" for t in tensors_with_data]
                        # TODO investigate further..
                        print(f"Memory thrashing detected: All tensors ({'; '.join(tensor_info_list)}) to be fetched need to free up space, but that would cause an infinite loop of free and fetch.")
                        print(f"The current memory size for {self.dest_memory.name} of {self.dest_memory.size} words may be too small.")
                        raise MemoryThrashingError("Memory thrashing detected and simulation aborted.")

                    # Reschedule this event
                    self.src_memory.pending_actions.append(self)
                    if self.verbose:
                        print(f"Write back or fetch in progress for {self.data_id}. Adding fetch to wait list.")
                    return
                
                # Else proceed to write-back
                victim_data_id, victim_data_category, victim_tensor_shape, victim_layout_info, victim_layout, victim_offset, victim_data_bitwidth = write_info

                # First write back the chosen victim data to the upper memory
                if self.verbose:
                    print(f"Scheduled write-back of {victim_data_id}'s {victim_layout} elements from {self.dest_memory.name} to {self.src_memory.name} to free space.")
                scheduler.schedule_event(self.event_time, MemoryWriteBackEvent(victim_layout_info, victim_data_id, victim_data_category, victim_data_bitwidth, victim_tensor_shape, victim_layout, victim_offset, self.dest_memory, self.src_memory, self.final_operation, next_event=self, verbose=self.verbose))
        else:  # Read from upper memory failed.. schedule another fetch from the upper memory of this upper memory
            missing_data, missing_data_offset = read_info

            # Free the ports
            self.src_memory.available_ports.append(self.src_port)
            self.src_port = None
            while self.src_memory.pending_actions:
                src_memory_op = self.src_memory.pending_actions.pop(0)
                scheduler.schedule_event(self.event_time, src_memory_op)

            self.dest_memory.available_ports.append(self.dest_port)
            self.dest_port = None
            while self.dest_memory.pending_actions:
                dest_memory_op = self.dest_memory.pending_actions.pop(0)
                scheduler.schedule_event(self.event_time, dest_memory_op)

            mem_found, mem_miss_data = self.src_memory.query_lower_memories(read_mode="read_tile", data_id=self.data_id, tile_shape=missing_data, offset=missing_data_offset)
            missing_data, missing_data_offset = mem_miss_data

            if mem_found:
                scheduler.schedule_event(self.event_time, MemoryFetchEvent(self.data_id, self.data_category, self.data_bitwidth, self.tensor_shape, missing_data, missing_data_offset, mem_found, mem_found.upper_level_memory, self.final_operation, self, verbose=self.verbose))
            # If not recursively present in any lower memory, try accessing the data from higher-level memory
            else:
                if self.src_memory.upper_level_memory:
                    if self.verbose:
                        print(f"Scheduled fetch of {self.data_id} from {self.src_memory.upper_level_memory.name} to {self.src_memory.name}.")
                    scheduler.schedule_event(self.event_time, MemoryFetchEvent(self.data_id, self.data_category, self.data_bitwidth, self.tensor_shape, missing_data, missing_data_offset, self.src_memory.upper_level_memory, self.src_memory, self.final_operation, self, verbose=self.verbose))
                else:
                    if self.verbose:
                        print(f"Failed to fetch {self.data_id} from {self.src_memory.name}'s upper memory - no upper memory found containing this data.")


class MemoryFetchCompleteEvent(AbstractEvent):
    def __init__(self, data_id, data_bitwidth, src_memory, src_port, dest_memory, dest_port, tensor_shape, tile_shape, offset, final_operation, cycles, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.data_id = data_id
        self.data_bitwidth = data_bitwidth
        self.src_memory = src_memory
        self.src_port = src_port
        self.dest_memory = dest_memory
        self.dest_port = dest_port
        self.tensor_shape = tensor_shape
        self.tile_shape = tile_shape
        self.offset = offset
        self.final_operation = final_operation
        self.cycles = cycles
        self.next_event = next_event

    def __str__(self):
        return f"MemoryFetchCompleteEvent(data_id={self.data_id}, data_bitwidth={self.data_bitwidth}, src_memory={self.src_memory.name}, src_port={self.src_port}, dest_memory={self.dest_memory.name}, dest_port={self.dest_port}, tensor_shape={self.tensor_shape}, tile_shape={self.tile_shape}, offset={self.offset}, final_operation={self.final_operation}, cycles={self.cycles}, next_event={self.next_event})"

    def execute(self, scheduler):
        self.src_memory.available_ports.append(self.src_port)
        self.src_port = None
        while self.src_memory.pending_actions:
            src_memory_op = self.src_memory.pending_actions.pop(0)
            scheduler.schedule_event(self.event_time, src_memory_op)

        self.dest_memory.available_ports.append(self.dest_port)
        self.dest_port = None
        while self.dest_memory.pending_actions:
            dest_memory_op = self.dest_memory.pending_actions.pop(0)
            scheduler.schedule_event(self.event_time, dest_memory_op)
        
        if len(self.src_memory.available_ports) == self.src_memory.ports:
            self.src_memory.synchronize_per_ports_cycles()
        if len(self.dest_memory.available_ports) == self.dest_memory.ports:
            self.dest_memory.synchronize_per_ports_cycles()

        # Unlocking the fetch and processing queued reads
        waiting_reads = self.dest_memory.fetch_locks.pop(self.data_id, [])
        self.dest_memory.unlock_fetch_lock(self.data_id)
        for read_event in waiting_reads:
            if self.verbose:
                print(f"Rescheduled waiting read of {self.data_id} from {self.dest_memory.name}.")
            scheduler.schedule_event(self.event_time, read_event)
        if self.verbose:
            print(f"Completed fetch of {self.data_id} from {self.src_memory.name} to {self.dest_memory.name}.")
        if self.next_event:
            # If the next event is TemporalTileComputeEvent, ensure all reads are complete before triggering it
            if isinstance(self.next_event, TemporalTileComputeEvent):
                self.next_event.pending_reads -= 1
                if self.next_event.pending_reads == 0:
                    scheduler.schedule_event(self.event_time, self.next_event)
            else:
                scheduler.schedule_event(self.event_time, self.next_event)


class MemoryWriteEvent(AbstractEvent):
    def __init__(self, data_id, data_category, data_bitwidth, tensor_shape, tile_shape, offset, memory, final_operation, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.data_id = data_id
        self.data_category = data_category
        self.data_bitwidth = data_bitwidth
        self.tensor_shape = tensor_shape
        self.tile_shape = tile_shape
        self.offset = offset
        self.memory = memory
        self.memory_port = None
        self.final_operation = final_operation
        self.next_event = next_event

    def __str__(self):
        return f"MemoryWriteEvent(data_id={self.data_id}, data_category={self.data_category}, data_bitwidth={self.data_bitwidth}, tensor_shape={self.tensor_shape}, tile_shape={self.tile_shape}, offset={self.offset}, memory={self.memory.name}, final_operation={self.final_operation}, next_event={self.next_event})"

    def execute(self, scheduler):
        if not self.memory.available_ports:
            self.memory.pending_actions.append(self)
            if self.verbose:
                print(f"Memory {self.memory.name} ports are busy. Rescheduling write of {self.data_id} to pending actions.")
            return
        self.memory_port = self.memory.get_available_port(scheduler.deterministic_key)

        success, write_info = self.memory.write(write_mode="write_tile", data_id=self.data_id, data_category=self.data_category, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.memory_port, tensors_needed_info=scheduler.execution_graph.tensors_needed, tile_shape=self.tile_shape, offset=self.offset, deterministic_key=scheduler.deterministic_key, verbose=self.verbose)
        if success:
            cycles = write_info
            self.memory.update_per_ports_cycles(self.memory_port, cycles, update_idle=False)
            if self.verbose:
                print(f"Wrote {self.data_id} to {self.memory.name} in {cycles} cycles and in global time {self.event_time + cycles}.")
            scheduler.schedule_event(self.event_time + cycles, MemoryWriteCompleteEvent(self.data_id, self.data_bitwidth, self.memory, self.memory_port, self.final_operation, self.tensor_shape, self.tile_shape, self.offset, cycles, self.next_event, verbose=self.verbose))

            # Additional write-back to higher-level memory for final output data
            if self.final_operation and not (isinstance(self.memory, OffChipMemory)):
                assert self.memory.upper_level_memory, f"Final operation cannot be written to offchip since {self.memory.name} has no associated upper memory to write to!"
                # TODO layout info!!
                scheduler.schedule_event(self.event_time + cycles, MemoryWriteBackEvent("tile", self.data_id, self.data_category, self.data_bitwidth, self.tensor_shape, self.tile_shape, self.offset, self.memory, self.memory.upper_level_memory, self.final_operation, next_event=self.next_event, verbose=self.verbose))
        else:  # Write failed, we need to proceed with write-back of the chosen victim data to upper memory
            # Free the port
            self.memory.available_ports.append(self.memory_port)
            self.memory_port = None
            while self.memory.pending_actions:
                memory_op = self.memory.pending_actions.pop(0)
                scheduler.schedule_event(self.event_time, memory_op)
            
            # Check if the operation could not proceed because different write-back or fetch operation is currently in action for the chosen data id
            if write_info is None:
                # Reschedule this event
                self.memory.pending_actions.append(self)
                if self.verbose:
                    print(f"Write back or fetch in progress for {self.data_id}. Adding write to wait list.")
                return

            # Else proceed to write-back
            victim_data_id, victim_data_category, victim_tensor_shape, victim_layout_info, victim_layout, victim_offset, victim_data_bitwidth = write_info

            # Write back the chosen victim data to the upper memory first
            if self.memory.upper_level_memory:
                if self.verbose:
                    print(f"Scheduled write-back of {victim_data_id}'s {victim_layout} elements from {self.memory.name} to {self.memory.upper_level_memory.name} to free space.")
                scheduler.schedule_event(self.event_time, MemoryWriteBackEvent(victim_layout_info, victim_data_id, victim_data_category, victim_data_bitwidth, victim_tensor_shape, victim_layout, victim_offset, self.memory, self.memory.upper_level_memory, self.final_operation, next_event=self, verbose=self.verbose))    
            else:
                if self.verbose:
                    print(f"Failed to write-back {victim_data_id} from {self.memory.name} to its upper memory - no attached upper memory found.")


class MemoryWriteCompleteEvent(AbstractEvent):
    def __init__(self, data_id, data_bitwidth, memory, memory_port, final_operation, tensor_shape, tile_shape, offset, cycles, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.data_id = data_id
        self.data_bitwidth = data_bitwidth
        self.tensor_shape = tensor_shape
        self.tile_shape = tile_shape
        self.offset = offset
        self.final_operation = final_operation
        self.memory = memory
        self.memory_port = memory_port
        self.cycles=cycles
        self.next_event = next_event

    def __str__(self):
        return f"MemoryWriteCompleteEvent(data_id={self.data_id}, data_bitwidth={self.data_bitwidth}, memory={self.memory.name}, memory_port={self.memory_port}, final_operation={self.final_operation}, cycles={self.cycles}, next_event={self.next_event})"

    def execute(self, scheduler):
        self.memory.available_ports.append(self.memory_port)
        self.memory_port = None

        while self.memory.pending_actions:
            memory_op = self.memory.pending_actions.pop(0)
            scheduler.schedule_event(self.event_time, memory_op)
        
        if len(self.memory.available_ports) == self.memory.ports:
            self.memory.synchronize_per_ports_cycles()
            
        if self.verbose:
            print(f"Completed write of {self.data_id} into {self.memory.name}.")
        if self.next_event and ((isinstance(self.memory, OffChipMemory) and self.final_operation) or not self.final_operation):
            # We need to modify the event time since the next event is typically a MatmulStartEvent/SpatialTileComputeEvent from which we wish to continue with next tile
            self.next_event.event_time = self.event_time
            self.next_event.schedule_next_tile(scheduler)


class MemoryWriteBackEvent(AbstractEvent):
    def __init__(self, data_layout_info, data_id, data_category, data_bitwidth, tensor_shape, data_layout, offset, src_memory, dest_memory, final_operation, next_event=None, verbose: bool = False):
        super().__init__(verbose)
        self.data_layout_info = data_layout_info
        assert data_layout_info == "contiguous" or data_layout_info == "tile", "Invalid data layout type for MemoryWriteBackEvent. It should be either 'contiguous' or 'tile'."
        self.data_id = data_id
        self.data_category = data_category
        self.data_bitwidth = data_bitwidth
        self.tensor_shape = tensor_shape
        self.data_layout = data_layout
        self.offset = offset
        self.src_memory = src_memory
        self.src_port = None
        self.dest_memory = dest_memory
        self.dest_port = None
        self.final_operation = final_operation
        self.next_event = next_event

    def __str__(self):
        return f"MemoryWriteBackEvent(data_layout_info={self.data_layout_info}, data_id={self.data_id}, data_category={self.data_category}, data_bitwidth={self.data_bitwidth}, tensor_shape={self.tensor_shape}, data_layout={self.data_layout}, offset={self.offset}, src_memory={self.src_memory.name}, dest_memory={self.dest_memory.name}, final_operation={self.final_operation}, next_event={self.next_event})"

    def execute(self, scheduler):
        if not self.src_memory.available_ports or not self.dest_memory.available_ports:
            self.src_memory.pending_actions.append(self)
            if self.verbose:
                print(f"Ports of memories (src) {self.src_memory.name} and (dest) {self.dest_memory.name} are busy. Rescheduling write back of {self.data_id} to pending actions.")
            return
        self.src_port = self.src_memory.get_available_port(scheduler.deterministic_key)
        self.dest_port = self.dest_memory.get_available_port(scheduler.deterministic_key)

        if self.data_layout_info == "tile":
            read_success, read_cycles = self.src_memory.read(read_mode="read_tile", data_id=self.data_id, data_category=self.data_category, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.src_port, tile_shape=self.data_layout, offset=self.offset)
        else:
            read_success, read_cycles = self.src_memory.read(read_mode="read_elements", data_id=self.data_id, data_category=self.data_category, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.src_port, elems_to_read=self.data_layout, offset=self.offset)

        assert read_success, "Failed to read data for write-back from source memory. This should never happen. Probably data was not found in the source memory."

        if self.data_layout_info == "tile":
            write_success, write_info = self.dest_memory.write(write_mode="write_tile", data_id=self.data_id, data_category=self.data_category, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.dest_port, tensors_needed_info=scheduler.execution_graph.tensors_needed, tile_shape=self.data_layout, offset=self.offset, deterministic_key=scheduler.deterministic_key, verbose=self.verbose)
        else: 
            write_success, write_info = self.dest_memory.write(write_mode="write_elements", data_id=self.data_id, data_category=self.data_category, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.dest_port, tensors_needed_info=scheduler.execution_graph.tensors_needed, elems_to_write=self.data_layout, offset=self.offset, deterministic_key=scheduler.deterministic_key, verbose=self.verbose)

        if write_success:
            #write_cycles =  0  # Latency already counted for read

            # TODO VALIDATE PORTS CYCLES
            # Update action cycles during read/write
            self.src_memory.update_per_ports_cycles(self.src_port, read_cycles, update_idle=False)
            self.dest_memory.update_per_ports_cycles(self.dest_port, read_cycles, update_idle=False)
            # Update idle cycles during read/write
            self.src_memory.update_per_ports_cycles(self.src_port, read_cycles, update_idle=True)
            self.dest_memory.update_per_ports_cycles(self.dest_port, read_cycles, update_idle=True)
            if self.verbose:
                print(f"Wrote back {self.data_id} from {self.src_memory.name} to {self.dest_memory.name} in {read_cycles} cycles and in global time {self.event_time + read_cycles}.")
            
            if self.final_operation and self.data_id == scheduler.execution_graph.root.output and not (isinstance(self.dest_memory, OffChipMemory)):  # Final write back must complete all the way to the off-chip memory
                next_write_back_event = MemoryWriteBackEvent(self.data_layout_info, self.data_id, self.data_category, self.data_bitwidth, self.tensor_shape, self.data_layout, self.offset, self.dest_memory, self.dest_memory.upper_level_memory, self.final_operation, next_event=self.next_event, verbose=self.verbose)
                scheduler.schedule_event(self.event_time + read_cycles, MemoryWriteBackCompleteEvent(self.data_id, self.data_bitwidth, self.src_memory, self.src_port, self.dest_memory, self.dest_port, self.tensor_shape, self.data_layout_info, self.offset, self.final_operation, read_cycles, next_write_back_event, verbose=self.verbose))
            else:
                scheduler.schedule_event(self.event_time + read_cycles, MemoryWriteBackCompleteEvent(self.data_id, self.data_bitwidth, self.src_memory, self.src_port, self.dest_memory, self.dest_port, self.tensor_shape, self.data_layout_info, self.offset, self.final_operation, read_cycles, self.next_event, verbose=self.verbose))
        else:  # Cannot write back the data to upper memory, because its memory is full.. we need to write-back the victim data chosen from it and write it back to its upper memory before retrying this write-back
            # Free the ports
            self.src_memory.available_ports.append(self.src_port)
            self.src_port = None
            while self.src_memory.pending_actions:
                src_memory_op = self.src_memory.pending_actions.pop(0)
                scheduler.schedule_event(self.event_time, src_memory_op)

            self.dest_memory.available_ports.append(self.dest_port)
            self.dest_port = None
            while self.dest_memory.pending_actions:
                dest_memory_op = self.dest_memory.pending_actions.pop(0)
                scheduler.schedule_event(self.event_time, dest_memory_op)
            
            # Check if the operation could not proceed because different write-back or fetch operation is currently in action for the chosen dat id
            if write_info is None:
                # Reschedule this event
                self.src_memory.pending_actions.append(self)
                if self.verbose:
                    print(f"Write back or fetch in progress for {self.data_id}. Adding free/write-back to wait list.")
                return

            # Else proceed to write-back
            victim_data_id, victim_data_category, victim_tensor_shape, victim_layout_info, victim_layout, victim_offset, victim_data_bitwidth = write_info
            # TODO REFACTOR STATS INTO TUPLE...
            
            # Write back the victim data from this memory's upper memory to its upper memory first
            if self.dest_memory.upper_level_memory:
                if self.verbose:
                    print(f"Scheduled write-back of {victim_data_id}'s {victim_layout} elements from {self.dest_memory.name} to {self.dest_memory.upper_level_memory.name} to free space.")
                scheduler.schedule_event(self.event_time, MemoryWriteBackEvent(victim_layout_info, victim_data_id, victim_data_category, victim_data_bitwidth, victim_tensor_shape, victim_layout, victim_offset, self.dest_memory, self.dest_memory.upper_level_memory, self.final_operation, next_event=self, verbose=self.verbose))
            else:
                if self.verbose:
                    print(f"Failed to write-back {victim_data_id} from {self.dest_memory.name} to its upper memory - no attached upper memory found.")


class MemoryWriteBackCompleteEvent(AbstractEvent):
    def __init__(self, data_id, data_bitwidth, src_memory, src_port, dest_memory, dest_port, tensor_shape, data_layout_info, offset, final_operation, cycles, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.data_id = data_id
        self.data_bitwidth = data_bitwidth
        self.src_memory = src_memory
        self.src_port = src_port
        self.dest_memory = dest_memory
        self.dest_port = dest_port
        self.tensor_shape = tensor_shape
        self.data_layout_info = data_layout_info
        self.offset = offset
        self.final_operation = final_operation
        self.cycles = cycles
        self.next_event = next_event

    def __str__(self):
        return f"MemoryWriteBackCompleteEvent(data_id={self.data_id}, data_bitwidth={self.data_bitwidth}, src_memory={self.src_memory.name}, src_port={self.src_port}, dest_memory={self.dest_memory.name}, dest_port={self.dest_port}, tensor_shape={self.tensor_shape}, data_layout_info={self.data_layout_info}, offset={self.offset}, final_operation={self.final_operation}, cycles={self.cycles}, next_event={self.next_event})"
    
    def execute(self, scheduler):
        self.src_memory.unlock_free_lock(self.data_id)

        self.src_memory.available_ports.append(self.src_port)
        self.src_port = None
        while self.src_memory.pending_actions:
            src_memory_op = self.src_memory.pending_actions.pop(0)
            scheduler.schedule_event(self.event_time, src_memory_op)

        self.dest_memory.available_ports.append(self.dest_port)
        self.dest_port = None
        while self.dest_memory.pending_actions:
            dest_memory_op = self.dest_memory.pending_actions.pop(0)
            scheduler.schedule_event(self.event_time, dest_memory_op)
        
        if len(self.src_memory.available_ports) == self.src_memory.ports:
            self.src_memory.synchronize_per_ports_cycles()
        if len(self.dest_memory.available_ports) == self.dest_memory.ports:
            self.dest_memory.synchronize_per_ports_cycles()

        if self.verbose:
            print(f"Completed write back of {self.data_id} into {self.dest_memory.name}.")
        if self.next_event:
            if isinstance(self.next_event, MatmulStartEvent) or isinstance(self.next_event, SpatialTileComputeEvent):
                # We need to modify the event time since the next event is typically a MatmulStartEvent/SpatialTileComputeEvent from which we wish to continue with next tile
                self.next_event.event_time = self.event_time
                self.next_event.schedule_next_tile(scheduler)
            else:                
                scheduler.schedule_event(self.event_time, self.next_event)


""" MATMUL EVENTS """
class MatmulStartEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, row_major: bool = True, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.matmul_array = matmul_array
        self.current_tile = 0
        self.num_row_tiles = 0
        self.num_col_tiles = 0
        self.row_major = row_major  # Specifies order of processing subtiles (row/col major)
        self._cycles = 0

    def __str__(self):
        return f"MatmulStartEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name}, row_major={self.row_major})"

    def execute(self, scheduler):
        # Check if the operation has any children (operations that depend on the output of this operation),
        # if yes, postpone the start event until all children are done (the last child will reschedule this event)
        if self.operation.children:
            self.operation.pending_event = self
            return
        self.matmul_array.current_operation_event = self  # Set the current start event associated to the matmul array component

        if self.operation.computation.startswith('concat'):
            # Process concatenation operation (rearrange memory data element names)
            self.process_concatenation(scheduler)

            # Remove the operation from the parents children (dependencies) list (list of operations which produce input data for the parent)
            self.operation.remove_dependency()
            for p in self.operation.parents:
                # If this operation is the last child of the parent, and the parent's event is pending, finally reschedule it
                if not p.children and p.pending_event is not None:
                    scheduler.schedule_event(self.event_time, p.pending_event)

            if self.verbose:
                print(f"Skipping concatenation operation: {self.operation.name}")
            if self.matmul_array.plan:
                next_operation = self.matmul_array.plan.pop(0)
                # TODO add and test row/col majorness
                next_start_event = MatmulStartEvent(next_operation, self.matmul_array, row_major=True, verbose=self.verbose)
                scheduler.schedule_event(self.event_time, next_start_event)
            return

        if not self.matmul_array.is_busy:
            dim_m, _, dim_n = get_matmul_dimensions(self.operation.input_data)

            self.num_row_tiles = math.ceil(dim_m / self.matmul_array.rows)
            self.num_col_tiles = math.ceil(dim_n / self.matmul_array.columns)

            if self.matmul_array.parent_component.auto_interconnect is True and self.matmul_array._auto_interconnect_set is False:
                self.matmul_array.find_and_assign_memories()

            if self.verbose:
                print(f"Splitting computation into {self.num_row_tiles * self.num_col_tiles} subtiles.")
            self.schedule_next_tile(scheduler)
        else:
            if self.verbose:
                print(f"{self.matmul_array.name} is busy, could not start {self.operation.name}.")

    def schedule_next_tile(self, scheduler):
        if self.current_tile < self.num_row_tiles * self.num_col_tiles:
            tile_idx = self.current_tile
            dim_m, dim_k, dim_n = get_matmul_dimensions(self.operation.input_data)

            if self.row_major:
                row_tile_idx = tile_idx // self.num_col_tiles
                col_tile_idx = tile_idx % self.num_col_tiles
            else:
                col_tile_idx = tile_idx // self.num_row_tiles
                row_tile_idx = tile_idx %  self.num_row_tiles

            # The last tile in a row or column may be smaller than the rest (if the dimensions are not divisible by the tile size)
            row_tile_dim = min(dim_m - row_tile_idx * self.matmul_array.rows, self.matmul_array.rows)
            col_tile_dim = min(dim_n - col_tile_idx * self.matmul_array.columns, self.matmul_array.columns)

            # Event scheduling (computation after both reads succeed)            
            compute_event = SpatialTileComputeEvent(self.operation, self.matmul_array, (row_tile_idx, col_tile_idx), (row_tile_dim, col_tile_dim), dim_k, tile_idx, self.num_row_tiles * self.num_col_tiles, parent_event=self, verbose=self.verbose)
            scheduler.schedule_event(self.event_time, compute_event)                

            self.current_tile += 1
        else:
            if self.verbose:
                print(f"Computed matmul operation '{self.operation.name}' on {self.matmul_array.name} in {self._cycles} cycles and in global time {self.event_time}.")
            scheduler.schedule_event(self.event_time, MatmulCompleteEvent(self.operation, self.matmul_array, self._cycles, verbose=self.verbose))

    def process_concatenation(self, scheduler):
        # TODO SHOULD BE ITSELF AN OP!
        def compute_tile_offsets(tile_shapes, final_shape):
            offsets = {}
            row_cursor = 0
            col_cursor = 0
            max_rows, max_cols = final_shape

            for data_id, (tile_rows, tile_cols) in tile_shapes.items():
                # Wrap to next row if needed
                if col_cursor + tile_cols > max_cols:
                    row_cursor += tile_rows
                    col_cursor = 0

                # Error if tile doesn't fit vertically
                if row_cursor + tile_rows > max_rows:
                    raise ValueError(f"Tile {data_id} does not fit in output matrix.")
                offsets[data_id] = (row_cursor, col_cursor)
                col_cursor += tile_cols
            return offsets

        # Retrieve list of all memory blocks and update the concatenad blocks in all of them (some parts may have already been written to upper memory)
        accelerator = self.matmul_array.parent_component
        memory_blocks = accelerator.memory_blocks + [accelerator.dram]
        tile_shapes = {data_id: self.operation.input_data[data_id]['dimensions'] for data_id in self.operation.input_data}

        for mem in memory_blocks:
            mem_contents = mem.contents 
            if any(key in mem_contents for key in self.operation.input_data.keys()):
                subtiles_for_writeback = {}
                
                # Initialize variables for the new concatenated matrix
                new_data_id = list(self.operation.output_data.keys())[0]
                presence_matrix = mem._initialize_presence_matrix(self.operation.output_data[new_data_id]['dimensions'], 0)
                data_category = self.operation.output_data[new_data_id]['data_category']
                data_amount = 0
                data_read_count = 0
                mem_read_count = 0
                data_write_count = 0
                mem_write_count = 0
                word_amount = 0
                word_read_count = 0
                word_write_count = 0
                data_bit_packing = False
                data_bitwidth = 0
                fragmented_bits = 0
                insertion_time = float('inf')
                last_access_time = float('-inf')
                cache_miss_count = 0

                # Iterate over the submatrix data IDs and accumulate stats values
                tile_offsets = compute_tile_offsets(tile_shapes, presence_matrix.shape)

                # TODO refactor with offsets (should be more robust and better for irregular subtiles) info from operation splitting? should be faster and more elegant
                for data_id in self.operation.input_data.keys():
                    if data_id in mem_contents:
                        row_off, col_off = tile_offsets[data_id]
                        tile_rows, tile_cols = self.operation.input_data[data_id]['dimensions']
                        if (scheduler.execution_graph.tensors_needed.get()[data_id]["count"] > 1):
                            submatrix = mem_contents[data_id]
                        else:
                            submatrix = mem_contents.pop(data_id)  # Remove the submatrix entry if it is not needed anymore

                        tile = submatrix['presence_matrix']
                        subtiles_for_writeback[data_id] = {"tile": tile.shape, "offset": (row_off, col_off)}

                        assert presence_matrix[row_off:row_off+tile_rows, col_off:col_off+tile_cols].shape == tile.shape, "Error during concatenation operation, some submatrix does not fit into the final matrix shape!"
                        presence_matrix[row_off:row_off+tile_rows, col_off:col_off+tile_cols] = tile
                        data_amount += submatrix['data_amount']
                        data_read_count += submatrix['data_read_count']
                        mem_read_count += submatrix['mem_read_count']
                        data_write_count += submatrix['data_write_count']
                        mem_write_count += submatrix['mem_write_count']
                        word_amount += submatrix['word_amount']
                        word_read_count += submatrix['word_read_count']
                        word_write_count += submatrix['word_write_count']
                        data_bit_packing += submatrix['data_bit_packing']
                        data_bitwidth = max(data_bitwidth, submatrix['data_bitwidth'])
                        fragmented_bits += submatrix['fragmented_bits']
                        insertion_time = min(insertion_time, submatrix['insertion_time'])
                        last_access_time = max(last_access_time, submatrix['last_access_time'])
                        cache_miss_count += submatrix['cache_miss_count']

                # Add the new concatenated matrix to the memory
                mem_contents[new_data_id] = {
                    'data_category': data_category,
                    'presence_matrix': presence_matrix,
                    'data_amount': data_amount,
                    'data_read_count': data_read_count,
                    'mem_read_count': mem_read_count,
                    'data_write_count': data_write_count,
                    'mem_write_count': mem_write_count,
                    'word_amount': word_amount,
                    'word_read_count': word_read_count,
                    'word_write_count': word_write_count,
                    'data_bit_packing': data_bit_packing,
                    'data_bitwidth': data_bitwidth,
                    'fragmented_bits': fragmented_bits,
                    'insertion_time': insertion_time,
                    'last_access_time': last_access_time,
                    'cache_miss_count': cache_miss_count
                }

                # If final operation is concatenation of final submatrices, we need to write them all the way back into offchip DRAM memory
                if self.operation._is_root and new_data_id == scheduler.execution_graph.root.output and mem != accelerator.dram:
                    for _, subtile in enumerate(subtiles_for_writeback.values()):
                        scheduler.schedule_event(self.event_time, MemoryWriteBackEvent("tile", new_data_id, data_category, self.operation.data_bitwidth, presence_matrix.shape, subtile['tile'], subtile['offset'], mem, accelerator.dram, self.operation._is_root, next_event=None, verbose=self.verbose))

                if self.verbose:
                    print(f"Replaced submatrices with {new_data_id} in memory {mem.name}.")

        # The original data IDs are now replaced with the new concatenated data ID
        for data_id in self.operation.input_data.keys():
            scheduler.execution_graph.tensors_needed.decrease_count(data_id)

        if self.operation._is_root:  # If final operation, the output is successfully in DRAM
            scheduler.execution_graph.tensors_needed.decrease_count(new_data_id)


class MatmulCompleteEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, cycles, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.matmul_array = matmul_array
        self.cycles = cycles

    def __str__(self):
        return f"MatmulCompleteEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name}, cycles={self.cycles})"

    def execute(self, scheduler):
        self.matmul_array.compute_done()
        self.operation.is_done = True
        # Remove the operation from the parents' children (dependencies) list (list of operations which produce input data for the parent)
        self.operation.remove_dependency()

        for p in self.operation.parents:
            # If this operation is the last child of the parent, and the parent's event is pending, finally reschedule it
            if not p.children and p.pending_event is not None:
                scheduler.schedule_event(self.event_time, p.pending_event)

        # Update the dictionary of required tensors
        input_vars = list(self.operation.input_data.keys())
        for i in input_vars:
            scheduler.execution_graph.tensors_needed.decrease_count(i)
        
        if self.operation._is_root:  # If final operation, the output is successfully in DRAM
            scheduler.execution_graph.tensors_needed.decrease_count(list(self.operation.output_data.keys())[0])
        if self.verbose:
            print(f"Completed matmul of '{self.operation.name}' on {self.matmul_array.name}.")
        if self.matmul_array.plan:
            next_operation = self.matmul_array.plan.pop(0)
            # TODO add and test row/col majorness
            next_start_event = MatmulStartEvent(next_operation, self.matmul_array, row_major=True, verbose=self.verbose)
            scheduler.schedule_event(self.event_time, next_start_event)


class SpatialTileComputeEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, r_c_tile_idxs, tile_dims, common_dim, tile_idx, subtiles, parent_event, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.row_tile_idx, self.col_tile_idx = r_c_tile_idxs
        self.row_tile_dim, self.col_tile_dim = tile_dims
        self.common_dim = common_dim
        self.matmul_array = matmul_array
        self.tile_idx = tile_idx
        self.subtiles = subtiles
        self.parent_event = parent_event
        self._cycles = 0

    def __str__(self):
        return f"SpatialTileComputeEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name}, tile_idx={self.tile_idx}, subtiles={self.subtiles})"

    def execute(self, scheduler):
        self.temp_subtiles = int(math.ceil(self.common_dim / self.matmul_array.buffer_length))
        self._tile_range = (range(self.temp_subtiles-1, -1, -1) if self.tile_idx != 0 else range(self.temp_subtiles))

        # Decide on order of temporal subtile computation based on data tensors reuse (in row/col buffers), either we start with temp tile 0 or the last
        inputs = self.operation.input_data
        in_a, in_b = inputs

        last_temp_tile_common_dim = min(self.common_dim-((self.temp_subtiles-1)*self.matmul_array.buffer_length), self.matmul_array.buffer_length)
        A_tile_shape = (self.row_tile_dim, last_temp_tile_common_dim)
        A_input_offset_x, _ = inputs[in_a]['offset']
        A_tile_offset = (A_input_offset_x + self.row_tile_idx * self.matmul_array.rows, (self.temp_subtiles-1)*self.matmul_array.buffer_length)            

        B_tile_shape = (self.col_tile_dim, last_temp_tile_common_dim) if (self.common_dim == inputs[in_a]['dimensions'][1]) and (self.common_dim == inputs[in_b]['dimensions'][1]) else (last_temp_tile_common_dim, self.col_tile_dim)  # Check if the matrix is transposed in memory or not..
        _, B_input_offset_y = inputs[in_b]['offset']
        B_tile_offset = (B_input_offset_y + self.col_tile_idx * self.matmul_array.columns, (self.temp_subtiles-1)*self.matmul_array.buffer_length) if (self.common_dim == inputs[in_a]['dimensions'][1]) and (self.common_dim == inputs[in_b]['dimensions'][1]) else ((self.temp_subtiles-1)*self.matmul_array.buffer_length, B_input_offset_y + self.col_tile_idx * self.matmul_array.columns)
        
        
        A_first_tile_shape = (self.row_tile_dim, min(self.common_dim, self.matmul_array.buffer_length))
        A_first_tile_offset = (A_input_offset_x + self.row_tile_idx * self.matmul_array.rows, 0)
        
        B_first_tile_shape = (self.col_tile_dim, min(self.common_dim, self.matmul_array.buffer_length)) if (self.common_dim == inputs[in_a]['dimensions'][1]) and (self.common_dim == inputs[in_b]['dimensions'][1]) else (min(self.common_dim, self.matmul_array.buffer_length), self.col_tile_dim)  # Check if the matrix is transposed in memory or not..
        B_first_tile_offset = (B_input_offset_y + self.col_tile_idx * self.matmul_array.columns, 0) if (self.common_dim == inputs[in_a]['dimensions'][1]) and (self.common_dim == inputs[in_b]['dimensions'][1]) else (0, B_input_offset_y + self.col_tile_idx * self.matmul_array.columns)
            
        if in_a in self.matmul_array.row_buffer.contents and self.matmul_array.row_buffer._check_tile_presence(in_a, A_first_tile_shape, A_first_tile_offset)[0]:
            self._forward_subtile_order = True
        elif in_b in self.matmul_array.col_buffer.contents and self.matmul_array.col_buffer._check_tile_presence(in_b, B_first_tile_shape, B_first_tile_offset)[0]:
            self._forward_subtile_order = True
        elif in_a in self.matmul_array.row_buffer.contents and self.matmul_array.row_buffer._check_tile_presence(in_a, A_tile_shape, A_tile_offset)[0]:
            self._forward_subtile_order = False
        elif in_b in self.matmul_array.col_buffer.contents and self.matmul_array.col_buffer._check_tile_presence(in_b, B_tile_shape, B_tile_offset)[0]:
            self._forward_subtile_order = False
        else:
            self._forward_subtile_order = True

        self.current_temp_tile = 0 if self._forward_subtile_order == True else self.temp_subtiles-1

        if self.verbose:
            print(f"Splitting spatial subtile {self.tile_idx+1}/{self.subtiles} (tile_id: {self.tile_idx}) computation for '{self.operation.name}' into {self.temp_subtiles} temporal subtiles.")
        self.schedule_next_tile(scheduler)
    
    def _create_compute_event(self, in_tile_a_info, in_tile_b_info, tile_common_dim, pending_reads):
        return TemporalTileComputeEvent(
            self.operation,
            self.matmul_array,
            (in_tile_a_info, in_tile_b_info),
            (self.row_tile_dim, self.col_tile_dim),
            tile_common_dim,
            (self.tile_idx, self.subtiles),
            (self.current_temp_tile, self.temp_subtiles),
            pending_reads=pending_reads,
            parent_event=self,
            verbose=self.verbose
        )

    def schedule_next_tile(self, scheduler):
        if (self._forward_subtile_order and self.current_temp_tile < self.temp_subtiles) or (not self._forward_subtile_order and self.current_temp_tile >= 0):
            in_tiles_offset = self.current_temp_tile * self.matmul_array.buffer_length
            tile_common_dim = min(self.common_dim-(self.current_temp_tile*self.matmul_array.buffer_length), self.matmul_array.buffer_length)

            input_vars = list(self.operation.input_data.keys())
            a, b = input_vars
        
            # Left matrix A
            A_tensor_shape = self.operation.input_data[a]['dimensions']
            A_data_category = self.operation.input_data[a]['data_category']
            A_tile_shape = (self.row_tile_dim, tile_common_dim)
            A_tile_memory = self.get_memory_for_var(a)
            A_input_offset_x, _ = self.operation.input_data[a]['offset']
            A_tile_offset = (A_input_offset_x + self.row_tile_idx * self.matmul_array.rows, in_tiles_offset)
            A_tile_key = f"{A_tile_shape[0]}x{A_tile_shape[1]}_{A_tile_offset[0]}x{A_tile_offset[1]}"
            in_tile_a_info = (a, A_tile_memory.name, A_tile_key)
            scheduler.execution_graph.tensors_needed.add_tile(*in_tile_a_info)

            # Right matrix B
            B_tensor_shape = self.operation.input_data[b]['dimensions']
            B_data_category = self.operation.input_data[b]['data_category']
            B_tile_shape = (self.col_tile_dim, tile_common_dim) if (self.common_dim == self.operation.input_data[a]['dimensions'][1]) and (self.common_dim == self.operation.input_data[b]['dimensions'][1]) else (tile_common_dim, self.col_tile_dim)  # Check if the matrix is transposed in memory or not..
            B_tile_memory = self.get_memory_for_var(b)
            _, B_input_offset_y = self.operation.input_data[b]['offset']
            B_tile_offset = (B_input_offset_y + self.col_tile_idx * self.matmul_array.columns, in_tiles_offset) if (self.common_dim == self.operation.input_data[a]['dimensions'][1]) and (self.common_dim == self.operation.input_data[b]['dimensions'][1]) else (in_tiles_offset, B_input_offset_y + self.col_tile_idx * self.matmul_array.columns)
            B_tile_key = f"{B_tile_shape[0]}x{B_tile_shape[1]}_{B_tile_offset[0]}x{B_tile_offset[1]}"
            in_tile_b_info = (b, B_tile_memory.name, B_tile_key)
            scheduler.execution_graph.tensors_needed.add_tile(*in_tile_b_info)

            # Event scheduling (computation after both reads succeed)            
            # TODO.. FOR FETCH.. ONLY READ OR READ AND ALSO WRITE COSTS??
            is_tile_a_found, missing_a_data = self.matmul_array.row_buffer._check_tile_presence(a, A_tile_shape, A_tile_offset)
            is_tile_b_found, missing_b_data = self.matmul_array.col_buffer._check_tile_presence(b, B_tile_shape, B_tile_offset)

            pending_reads = 2 if not is_tile_a_found and not is_tile_b_found else 1 if not is_tile_a_found or not is_tile_b_found else 0
            compute_event = TemporalTileComputeEvent(
                                self.operation,
                                self.matmul_array,
                                (in_tile_a_info, in_tile_b_info),
                                (self.row_tile_dim, self.col_tile_dim),
                                tile_common_dim,
                                (self.tile_idx, self.subtiles),
                                (self.current_temp_tile, self.temp_subtiles),
                                pending_reads=pending_reads,
                                parent_event=self,
                                verbose=self.verbose
                            )               

            if not is_tile_a_found:
                missing_a_data, missing_a_data_offset = missing_a_data
                A_read_event  = MemoryFetchEvent(a, A_data_category, self.matmul_array.data_bitwidth, A_tensor_shape, missing_a_data, missing_a_data_offset, self.matmul_array.row_buffer.upper_level_memory, self.matmul_array.row_buffer, self.operation._is_root, next_event=compute_event, verbose=self.verbose)
                scheduler.schedule_event(self.event_time, A_read_event)

            if not is_tile_b_found:
                missing_b_data, missing_b_data_offset = missing_b_data
                B_read_event  = MemoryFetchEvent(b, B_data_category, self.matmul_array.data_bitwidth, B_tensor_shape, missing_b_data, missing_b_data_offset, self.matmul_array.col_buffer.upper_level_memory, self.matmul_array.col_buffer, self.operation._is_root, next_event=compute_event, verbose=self.verbose)
                scheduler.schedule_event(self.event_time, B_read_event)

            if is_tile_a_found and is_tile_b_found:
                scheduler.schedule_event(self.event_time, compute_event)
            
            self.current_temp_tile = self.current_temp_tile + 1 if self._forward_subtile_order else self.current_temp_tile - 1
        else:
            dim_m, _, dim_n = get_matmul_dimensions(self.operation.input_data)
            out_var = list(self.operation.output_data.keys())[0]
            out_data_category = self.operation.output_data[out_var]['data_category']
            out_tensor_shape = (dim_m, dim_n)
            out_tile_shape = (self.row_tile_dim, self.col_tile_dim)
            out_offset = (self.matmul_array.rows * self.row_tile_idx, self.matmul_array.columns * self.col_tile_idx)

            write_event = MemoryWriteEvent(out_var, out_data_category, self.operation.data_bitwidth, out_tensor_shape, out_tile_shape, out_offset, self.matmul_array.dynamic_param_memory, self.operation._is_root, self.matmul_array.current_operation_event, verbose=self.verbose)
            if self.verbose:
                print(f"Computed spatial tile {self.tile_idx+1}/{self.subtiles} for '{self.operation.name}' on {self.matmul_array.name} in {self._cycles} cycles and in global time {self.event_time}.")
            tile_complete_event = SpatialTileCompleteEvent(self.operation, self.matmul_array, self.tile_idx, self.subtiles, self._cycles, next_event=write_event, verbose=self.verbose)
            self.parent_event._cycles += self._cycles  # Accumulate total cycles required for the whole matmul computation
            scheduler.schedule_event(self.event_time, tile_complete_event)

    def get_memory_for_var(self, var):
        if self.operation.input_data[var]['data_category'] == 'static':
            return self.matmul_array.static_param_memory
        else:
            return self.matmul_array.dynamic_param_memory


class SpatialTileCompleteEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, tile_idx, subtiles, cycles, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.matmul_array = matmul_array
        self.tile_idx = tile_idx
        self.subtiles = subtiles
        self.cycles = cycles
        self.next_event = next_event

    def __str__(self):
        return f"SpatialTileCompleteEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name}, tile_idx={self.tile_idx}, subtiles={self.subtiles}, cycles={self.cycles})"

    def execute(self, scheduler):
        if self.verbose:
            print(f"Completed computation of spatial tile {self.tile_idx+1}/{self.subtiles} (tile_id: {self.tile_idx}) for '{self.operation.name}' on {self.matmul_array.name}.")
        if self.next_event:
            scheduler.schedule_event(self.event_time, self.next_event)

class TemporalTileComputeEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, in_tiles_info, computed_tile_dims, common_dim, spat_subtile_info, temp_subtile_info, pending_reads, parent_event, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.matmul_array = matmul_array
        self._in_tile_a, self._in_tile_b = in_tiles_info
        self.row_tile_dim, self.col_tile_dim = computed_tile_dims
        self.common_dim = common_dim
        self.spat_tile_idx, self.spat_subtiles = spat_subtile_info
        self.temp_tile_idx, self.temp_subtiles = temp_subtile_info
        self.parent_event = parent_event
        self.pending_reads = pending_reads  # WE WAIT FOR READS OF TWO MATRICES

    def __str__(self):
        return f"TemporalTileComputeEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name}, spat_subtile_idx={self.spat_tile_idx}, spat_subtiles={self.spat_subtiles}, temp_subtile_idx={self.temp_tile_idx}, temp_subtiles={self.temp_subtiles})"

    def execute(self, scheduler):
        row_elem_reads = self.row_tile_dim * self.common_dim
        a_data_metadata = self.matmul_array.row_buffer.contents[self._in_tile_a[0]]
        a_data_metadata['data_read_count'] += row_elem_reads
        a_data_metadata['word_read_count'] += row_elem_reads
        a_data_metadata['mem_read_count'] += row_elem_reads
        self.matmul_array.row_buffer.data_read_count += row_elem_reads
        self.matmul_array.row_buffer.word_read_count += row_elem_reads
        self.matmul_array.row_buffer.mem_read_count += row_elem_reads

        b_data_metadata = self.matmul_array.col_buffer.contents[self._in_tile_b[0]]
        col_elem_reads = self.col_tile_dim * self.common_dim   
        b_data_metadata['data_read_count'] += col_elem_reads
        b_data_metadata['word_read_count'] += col_elem_reads
        b_data_metadata['mem_read_count'] += col_elem_reads
        self.matmul_array.col_buffer.data_read_count += col_elem_reads
        self.matmul_array.col_buffer.word_read_count += col_elem_reads
        self.matmul_array.col_buffer.mem_read_count += col_elem_reads

        cycles = self.matmul_array.compute(self.row_tile_dim, self.common_dim, self.col_tile_dim)
        self.parent_event._cycles += cycles  # Accumulate total cycles required for the whole spatial tile computation
        if self.verbose:
            print(f"Computed temporal tile {self.temp_tile_idx+1}/{self.temp_subtiles} of spatial tile {self.spat_tile_idx+1}/{self.spat_subtiles} for '{self.operation.name}' on {self.matmul_array.name} in {cycles} cycles and in global time {self.event_time + cycles}.")

        tile_complete_event = TemporalTileCompleteEvent(self.operation, self.matmul_array, self._in_tile_a, self._in_tile_b, (self.spat_tile_idx, self.spat_subtiles), (self.temp_tile_idx, self.temp_subtiles), cycles, next_event=self.parent_event, verbose=self.verbose)
        scheduler.schedule_event(self.event_time + cycles, tile_complete_event)


class TemporalTileCompleteEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, in_tile_a, in_tile_b, spat_subtile_info, temp_subtile_info, cycles, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.matmul_array = matmul_array
        self._in_tile_a = in_tile_a
        self._in_tile_b = in_tile_b
        self.spat_tile_idx, self.spat_subtiles = spat_subtile_info
        self.temp_tile_idx, self.temp_subtiles = temp_subtile_info
        self.cycles = cycles
        self.next_event = next_event

    def __str__(self):
        return f"TemporalTileCompleteEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name}, spat_subtile_idx={self.spat_tile_idx}, spat_subtiles={self.spat_subtiles}, temp_subtile_idx={self.temp_tile_idx}, temp_subtiles={self.temp_subtiles}, cycles={self.cycles})"

    def execute(self, scheduler):
        a, A_tile_memory, A_tile_key = self._in_tile_a
        b, B_tile_memory, B_tile_key = self._in_tile_b
        scheduler.execution_graph.tensors_needed.remove_tile(a, A_tile_memory, A_tile_key)
        scheduler.execution_graph.tensors_needed.remove_tile(b, B_tile_memory, B_tile_key)

        if self.verbose:
            print(f"Completed computation of temporal tile {self.temp_tile_idx+1}/{self.temp_subtiles} (tile_id: {self.temp_tile_idx}) of spatial tile {self.spat_tile_idx+1}/{self.spat_subtiles} (tile_id: {self.spat_tile_idx}) for '{self.operation.name}' on {self.matmul_array.name}.")
        if self.next_event:
            assert isinstance(self.next_event, SpatialTileComputeEvent)
            # We need to modify the event time since the next event is typically a MatmulStartEvent/SpatialTileComputeEvent from which we wish to continue with next tile
            self.next_event.event_time = self.event_time
            self.next_event.schedule_next_tile(scheduler)


def get_matmul_dimensions(input_data):
        input_vars = list(input_data.keys())
        if not input_vars or len(input_vars) != 2:
            raise ValueError("Invalid matmul inputs")
        x, y = input_vars
        dims_x = input_data[x]['tile_shape']
        dims_y = input_data[y]['tile_shape']

        if dims_x[1] == dims_y[0]:
            dim_m, dim_k, dim_n = dims_x[0], dims_x[1], dims_y[1]
        elif dims_x[1] == dims_y[1]:  # Transpose the second matrix
            dim_m, dim_k, dim_n = dims_x[0], dims_x[1], dims_y[0]
        else:
            raise ValueError(f"Incompatible dimensions for matmul: X:{dims_x}, Y:{dims_y}")

        return dim_m, dim_k, dim_n
