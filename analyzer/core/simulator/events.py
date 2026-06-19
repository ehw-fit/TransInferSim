from abc import ABC, abstractmethod
from analyzer.hardware_components.memories.offchip import OffChipMemory
import math


""" Abstract base class for Event """
class AbstractEvent(ABC):
    def __init__(self, verbose: bool = False, log_state_before: bool = False):
        self._event_time = 0
        self.state_before_changes = None
        self.verbose = verbose
        self.log_state_before = log_state_before

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

    def retract_changes(self):
        """
       Restores component state from `state_before_changes`.
        Supports:
            {"components":[ {"obj": ..., "params":[], "contents":[], "method_calls": []}, ... ]}
        """
        sbc = self.state_before_changes
        if not sbc:
            return

        for comp_entry in sbc.get("components", []):
            # actual component instance (the field that isn't params/contents/method_calls)
            comp_key = next(k for k in comp_entry.keys() if k not in ("params", "contents", "method_calls"))
            comp_obj = comp_entry[comp_key]

            # restore component params
            for p in comp_entry["params"]:
                if "index" not in p:
                    setattr(comp_obj, p["param"], p["value"])
                else:
                    self._restore_indexed_param(comp_obj, p)
            
            contents_log = comp_entry.get("contents", [])
            # Matmulstart restores the full snapshot
            if isinstance(self, MatmulStartEvent) and hasattr(comp_obj, "contents") and isinstance(comp_obj.contents, dict):
                # EXACT restore (MatmulStartEvent): snapshot is the full truth (even if empty)
                comp_obj.contents.clear()

                for c in contents_log:
                    data_id = c["id"]
                    # your snapshots store action="update" for present ids; ignore others if ever present
                    if c.get("action") == "remove":
                        continue

                    meta_snapshot = {mp["param"]: mp["value"] for mp in c.get("params", [])}
                    comp_obj.contents[data_id] = meta_snapshot

            else:
                for c in contents_log:
                    data_id = c["id"]
                    action = c["action"]

                    if action == "remove":
                        if data_id in comp_obj.contents:
                            del comp_obj.contents[data_id]
                        continue

                    if action == "readd":
                        # If the entry does not exist anymore (victim of free)
                        if data_id not in comp_obj.contents:
                            comp_obj.contents[data_id] = {}

                        meta = comp_obj.contents[data_id]
                        for mp in c.get("params", []):
                            meta[mp["param"]] = mp["value"]
                        continue

                    if action == "update":
                        if data_id in comp_obj.contents:
                            meta = comp_obj.contents[data_id]
                            for mp in c.get("params", []):
                                meta[mp["param"]] = mp["value"]
            
            # replay method calls in exact order
            for m in comp_entry.get("method_calls", []):
                method = getattr(comp_obj, m["method"])
                args = m.get("args", [])
                kwargs = m.get("kwargs", {})
                method(*args, **kwargs)
        
        for param_entry in sbc.get("params", []):
            setattr(self, param_entry["param"], param_entry["value"])
    
    def _restore_indexed_param(self, comp_obj, p):
        """
        Handles restoring values inside 2D arrays like cycles_per_ports.
        index is always "[port][0]" or "[port][1]".
        """

        param = p["param"]
        idx_string = p["index"]
        value = p["value"]

        # parse "[3][1]" → [3, 1]
        parts = idx_string.replace("[", " ").replace("]", " ").split()
        i, j = int(parts[0]), int(parts[1])

        arr = getattr(comp_obj, param)
        arr[i][j] = value






def _event_key(event):
    if isinstance(event, MemoryReadEvent):
        return (
            "MemoryReadEvent",
            event.data_id,
            event.data_category,
            event.is_data_broadcasted,
            event.data_bitwidth,
            event.tensor_shape,
            event.tile_shape,
            event.offset,
            id(event.memory),
            event.op_priority,
            event.operation,
        )

    if isinstance(event, MemoryWriteEvent):
        return (
            "MemoryWriteEvent",
            event.data_id,
            event.data_category,
            event.is_data_broadcasted,
            event.data_bitwidth,
            event.tensor_shape,
            event.tile_shape,
            event.offset,
            id(event.memory),
            event.op_priority,
            event.operation,
        )

    if isinstance(event, MemoryFetchEvent):
        return (
            "MemoryFetchEvent",
            event.data_id,
            event.data_category,
            event.is_data_broadcasted,
            event.data_bitwidth,
            event.tensor_shape,
            event.tile_shape,
            event.offset,
            id(event.src_memory),
            id(event.dest_memory),
            event.op_priority,
            event.operation,
        )

    if isinstance(event, MemoryWriteBackEvent):
        return (
            "MemoryWriteBackEvent",
            event.data_layout_info,
            event.data_id,
            event.data_category,
            event.is_data_broadcasted,
            event.data_bitwidth,
            event.tensor_shape,
            event.data_layout,
            event.offset,
            id(event.src_memory),
            id(event.dest_memory),
            event.op_priority,
            event.operation,
        )

    return None

def _remove_first_logical_match(lst, target):
    target_key = _event_key(target)
    if target_key is None:
        return

    for i, item in enumerate(lst):
        if type(item) is type(target) and _event_key(item) == target_key:
            del lst[i]
            return

def _remove_from_pending(mem_event):
    if isinstance(mem_event, (MemoryFetchEvent, MemoryWriteBackEvent)):
        src = mem_event.src_memory
        dst = mem_event.dest_memory
        _remove_first_logical_match(src.pending_actions, mem_event)
        if dst is not src:
            _remove_first_logical_match(dst.pending_actions, mem_event)

    elif isinstance(mem_event, (MemoryWriteEvent, MemoryReadEvent)):
        m = mem_event.memory
        _remove_first_logical_match(m.pending_actions, mem_event)

# HELPER FUNCTIONS FOR CACHING CHANGES MADE BY EVENT EXECUTIONS
def _param_key(p):
    return (p["param"], p.get("index"))

def upsert_params(dst_params: list, src_params: list):
    """Last-write-wins by (param, index). No duplicate keys."""
    idx = {_param_key(p): i for i, p in enumerate(dst_params)}
    for p in src_params:
        k = _param_key(p)
        if k in idx:
            dst_params[idx[k]] = p          # overwrite whole entry (incl. value)
        else:
            idx[k] = len(dst_params)
            dst_params.append(p)

def upsert_contents(dst_contents: list, src_contents: list):
    """Last-write-wins by data_id. No duplicate ids."""
    idx = {c["id"]: i for i, c in enumerate(dst_contents)}
    for c in src_contents:
        data_id = c["id"]
        if data_id in idx:
            dst_contents[idx[data_id]] = c  # overwrite whole record (action+params)
        else:
            idx[data_id] = len(dst_contents)
            dst_contents.append(c)

def snapshot_mem_contents(mem):
    out = []
    for data_id, entry in mem.contents.items():
        params = []
        for k, v in entry.items():
            if k == "presence_matrix":
                v_copy = v.copy() if v is not None else None
            else:
                v_copy = v
            params.append({"param": k, "value": v_copy})
        out.append({"id": data_id, "action": "update", "params": params})
    return out

def snapshot_mem_ports(mem):
    out = []
    for p in range(mem.ports):
        out.append({"param": "cycles_per_ports", "index": f"[{p}][0]", "value": mem.cycles_per_ports[p][0]})
        out.append({"param": "cycles_per_ports", "index": f"[{p}][1]", "value": mem.cycles_per_ports[p][1]})
    return out


class MemoryThrashingError(Exception):
    """Custom exception for memory thrashing issues."""
    pass


""" MEMORY EVENTS """
class MemoryReadEvent(AbstractEvent):
    def __init__(self, data_id, data_category, is_data_broadcasted, data_bitwidth, tensor_shape, tile_shape, offset, memory, op_priority, operation = None, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.data_id = data_id
        self.data_category = data_category
        self.is_data_broadcasted = is_data_broadcasted
        self.data_bitwidth = data_bitwidth
        self.tensor_shape = tensor_shape
        self.tile_shape = tile_shape
        self.offset = offset
        self.memory = memory
        self.memory_port = None
        self.op_priority = op_priority
        self.operation = operation
        self.next_event = next_event

    def __str__(self):
        return f"MemoryReadEvent(data_id={self.data_id}, is_data_broadcasted={self.is_data_broadcasted}, data_category={self.data_category}, data_bitwidth={self.data_bitwidth}, tensor_shape={self.tensor_shape}, tile_shape={self.tile_shape}, offset={self.offset}, memory={self.memory.name}, op_priority={self.op_priority}, operation={self.operation}, next_event={self.next_event})"

    def clone_for_retry(self):
        return MemoryReadEvent(
            data_id=self.data_id,
            data_category=self.data_category,
            is_data_broadcasted=self.is_data_broadcasted,
            data_bitwidth=self.data_bitwidth,
            tensor_shape=self.tensor_shape,
            tile_shape=self.tile_shape,
            offset=self.offset,
            memory=self.memory,
            op_priority=self.op_priority,
            operation=self.operation,
            next_event=self.next_event,
            verbose=self.verbose,
        )
    
    def execute(self, scheduler):
        # Check if the exact tile being requested is already being fetched by another event.
        # Key is (data_id, offset) so different tiles of the same tensor don't block each other.
        _lock_key = (self.data_id, self.offset)
        if _lock_key in self.memory.fetch_locks:
            self.memory.fetch_locks[_lock_key].append(self.clone_for_retry())
            if self.verbose:
                print(f"Fetch in progress for {self.data_id} (offset {self.offset}) from {self.memory.name}. Adding read to wait list.")
            return

        if not self.memory.available_ports:
            self.memory.pending_actions.append(self.clone_for_retry())
            if self.verbose:
                print(f"Memory {self.memory.name} ports are busy. Rescheduling read of {self.data_id} to pending actions.")
            return
        self.memory_port = self.memory.get_available_port()
        success, read_info = self.memory.read(read_mode="read_tile", data_id=self.data_id, data_category=self.data_category, is_data_broadcasted=self.is_data_broadcasted, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.memory_port, tile_shape=self.tile_shape, offset=self.offset)

        if success:
            cycles = read_info
            self.memory.update_per_ports_cycles(self.memory_port, cycles, update_idle=False)
            if self.verbose:
                print(f"Read {self.data_id} from {self.memory.name} in {cycles} cycles and in global time {self.event_time}.")
            scheduler.schedule_event(self.event_time + cycles, MemoryReadCompleteEvent(self.data_id, self.data_bitwidth, self.memory, self.memory_port, self.tensor_shape, self.tile_shape, self.offset, self.op_priority, self.operation, cycles, self.next_event, verbose=self.verbose), self.op_priority)
        else:
            missing_data, missing_data_offset = read_info

            # Free the port
            self.memory.free_port(self.memory_port)
            self.memory_port = None
            while self.memory.pending_actions:
                memory_op = self.memory.pending_actions.pop(0)
                _remove_from_pending(memory_op)
                scheduler.schedule_event(self.event_time, memory_op, memory_op.op_priority)

            mem_found, mem_miss_data = self.memory.query_lower_memories(read_mode="read_tile", data_id=self.data_id, tile_shape=missing_data, offset=missing_data_offset)
            missing_data, missing_data_offset = mem_miss_data

            if mem_found:
                scheduler.schedule_event(self.event_time, MemoryFetchEvent(self.data_id, self.data_category, self.is_data_broadcasted, self.data_bitwidth, self.tensor_shape, missing_data, missing_data_offset, mem_found, mem_found.upper_level_memory, self.op_priority, self.operation, self.clone_for_retry(), verbose=self.verbose), self.op_priority)
            # If not recursively present in any lower memory, try accessing the data from higher-level memory
            else:
                if self.memory.upper_level_memory:
                    if self.verbose:
                        print(f"Scheduled fetch of {self.data_id} from {self.memory.upper_level_memory.name} to {self.memory.name}.")
                    scheduler.schedule_event(self.event_time, MemoryFetchEvent(self.data_id, self.data_category, self.is_data_broadcasted, self.data_bitwidth, self.tensor_shape, missing_data, missing_data_offset, self.memory.upper_level_memory, self.memory, self.op_priority, self.operation, self.clone_for_retry(), verbose=self.verbose), self.op_priority)
                else:
                    if self.verbose:
                        print(f"Failed to fetch {self.data_id} from {self.memory.name}'s upper memory - no upper memory found containing this data.")


class MemoryReadCompleteEvent(AbstractEvent):
    def __init__(self, data_id, data_bitwidth, memory, memory_port, tensor_shape, tile_shape, offset, op_priority, operation, cycles, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.data_id = data_id
        self.data_bitwidth = data_bitwidth
        self.memory = memory
        self.memory_port = memory_port
        self.tensor_shape = tensor_shape
        self.tile_shape = tile_shape
        self.offset = offset
        self.op_priority = op_priority
        self.operation = operation
        self.cycles = cycles
        self.next_event = next_event

    def __str__(self):
        return f"MemoryReadCompleteEvent(data_id={self.data_id}, memory={self.memory.name}, memory_port={self.memory_port}, tensor_shape={self.tensor_shape}, tile_shape={self.tile_shape}, offset={self.offset}, op_priority={self.op_priority}, operation={self.operation}, cycles={self.cycles}, next_event={self.next_event})"

    def execute(self, scheduler):
        self.memory.free_port(self.memory_port)
        while self.memory.pending_actions:
            memory_op = self.memory.pending_actions.pop(0)
            _remove_from_pending(memory_op)
            scheduler.schedule_event(self.event_time, memory_op, memory_op.op_priority)

        if len(self.memory.available_ports) == self.memory.ports:
            self.memory.synchronize_per_ports_cycles()

        if self.operation is not None:
            self.operation._phase_log.append((self.event_time - self.cycles, self.event_time, "onchip"))
        if self.verbose:
            print(f"Completed read of {self.data_id} from {self.memory.name}.")
        if self.next_event:
            scheduler.schedule_event(self.event_time, self.next_event, self.next_event.op_priority)


class MemoryFetchEvent(AbstractEvent):
    def __init__(self, data_id, data_category, is_data_broadcasted, data_bitwidth, tensor_shape, tile_shape, offset, src_memory, dest_memory, op_priority, operation, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.data_id = data_id
        self.data_category = data_category
        self.is_data_broadcasted = is_data_broadcasted
        self.data_bitwidth = data_bitwidth
        self.tensor_shape = tensor_shape
        self.tile_shape = tile_shape
        self.offset = offset
        self.src_memory = src_memory
        self.src_port = None
        self.dest_memory = dest_memory
        self.dest_port = None
        self.op_priority = op_priority
        self.operation = operation
        self.next_event = next_event

    def __str__(self):
        return f"MemoryFetchEvent(data_id={self.data_id}, data_category={self.data_category}, is_data_broadcasted={self.is_data_broadcasted}, data_bitwidth={self.data_bitwidth}, tensor_shape={self.tensor_shape}, tile_shape={self.tile_shape}, offset={self.offset}, src_memory={self.src_memory.name}, dest_memory={self.dest_memory.name}, op_priority={self.op_priority}, operation={self.operation}, next_event={self.next_event})"

    def clone_for_retry(self):
        return MemoryFetchEvent(
            data_id=self.data_id,
            data_category=self.data_category,
            is_data_broadcasted=self.is_data_broadcasted,
            data_bitwidth=self.data_bitwidth,
            tensor_shape=self.tensor_shape,
            tile_shape=self.tile_shape,
            offset=self.offset,
            src_memory=self.src_memory,
            dest_memory=self.dest_memory,
            op_priority=self.op_priority,
            operation=self.operation,
            next_event=self.next_event,
            verbose=self.verbose,
        )

    def execute(self, scheduler):
        # If this src memory currently has an in-progress fetch for the EXACT same tile
        # (DRAM→src write not yet complete), block this read until the write completes.
        # Key is (data_id, offset) so different tiles of the same tensor are independent.
        _lock_key = (self.data_id, self.offset)
        if hasattr(self.src_memory, 'fetch_locks') and _lock_key in self.src_memory.fetch_locks:
            self.src_memory.fetch_locks[_lock_key].append(self.clone_for_retry())
            if self.verbose:
                print(f"Fetch in progress for {self.data_id} (offset {self.offset}) in {self.src_memory.name}. Adding read to wait list.")
            return

        if not self.src_memory.available_ports or not self.dest_memory.available_ports:
            self.src_memory.pending_actions.append(self.clone_for_retry())
            self.dest_memory.pending_actions.append(self.clone_for_retry())
            if self.verbose:
                print(f"Ports of memories (src) {self.src_memory.name} and (dest) {self.dest_memory.name} are busy. Rescheduling fetch of {self.data_id} to pending actions.")
            return
        self.src_port = self.src_memory.get_available_port()
        self.dest_port = self.dest_memory.get_available_port()

        read_success, read_info = self.src_memory.read(read_mode="read_tile", data_id=self.data_id, data_category=self.data_category, is_data_broadcasted=self.is_data_broadcasted, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.src_port, tile_shape=self.tile_shape, offset=self.offset)

        if self.src_memory.detect_memory_thrashing(self.data_id, self.tensor_shape, self.tile_shape, self.offset):
            tensor_info_list = [f"Data ID: {k}, Tensor Shape: {v['tensor_shape']}, Tile Shape: {v['tile_shape']}, Offset: {v['offset']}" for k, v in self.src_memory.last_reads.items()]
            print(f"Memory thrashing detected: All tensors ({'; '.join(tensor_info_list)}) to be fetched need to free up space, but that would cause an infinite loop of free and fetch.")
            print(f"The current memory size for {self.dest_memory.name} of {self.dest_memory.size} words may be too small.")
            raise MemoryThrashingError("Memory thrashing detected and simulation aborted.")

        if read_success:
            read_cycles = read_info
            write_success, write_info = self.dest_memory.write(write_mode="write_tile", data_id=self.data_id, data_category=self.data_category, is_data_broadcasted=self.is_data_broadcasted, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.dest_port, tensors_needed_info=scheduler.execution_graph.tensors_needed, tile_shape=self.tile_shape, offset=self.offset, verbose=self.verbose)

            if write_success:
                # Update action cycles during read/write
                self.src_memory.update_per_ports_cycles(self.src_port, read_cycles, update_idle=False)
                self.dest_memory.update_per_ports_cycles(self.dest_port, read_cycles, update_idle=False)
                if self.verbose:
                    print(f"Fetched {self.data_id} from {self.src_memory.name} to {self.dest_memory.name} in {read_cycles} cycles and in global time {self.event_time + read_cycles}.")
                scheduler.schedule_event(self.event_time + read_cycles, MemoryFetchCompleteEvent(self.data_id, self.data_bitwidth, self.src_memory, self.src_port, self.dest_memory, self.dest_port, self.tensor_shape, self.tile_shape, self.offset, self.op_priority, self.operation, read_cycles, self.next_event, verbose=self.verbose), self.op_priority)
            else:  # Read from upper memory was successful, but write to lower memory failed because of not enough space → we thus first need to free data in the lower memory (i.e. do a write-back to upper memory and then fetch)
                # Free the ports
                self.src_memory.free_port(self.src_port)
                self.src_port = None
                while self.src_memory.pending_actions:
                    src_memory_op = self.src_memory.pending_actions.pop(0)
                    _remove_from_pending(src_memory_op)
                    scheduler.schedule_event(self.event_time, src_memory_op, src_memory_op.op_priority)

                self.dest_memory.free_port(self.dest_port)
                self.dest_port = None
                while self.dest_memory.pending_actions:
                    dest_memory_op = self.dest_memory.pending_actions.pop(0)
                    _remove_from_pending(dest_memory_op)
                    scheduler.schedule_event(self.event_time, dest_memory_op, dest_memory_op.op_priority)

                # Check if the operation could not proceed because different write-back/fetch operation is currently in action for the chosen data id
                if write_info is None:
                    # Check for possible memory thrashing (loop where two data tensors need space.. but at the same time cannot free each other)
                    tensors_with_data = {k for k, v in self.dest_memory.contents.items() if v['word_amount'] > 0}  # All data ids present in memory
                    locked_data_ids = {k[0] for k in self.dest_memory.fetch_locks.keys()}  # data_id component of (data_id, offset) compound keys
                    if tensors_with_data == locked_data_ids or (self.data_id in locked_data_ids and len(tensors_with_data) == 1):
                        tensor_info_list = [f"Data ID: {t}, Data Bitwidth: {self.dest_memory.contents[t]['data_bitwidth']}, Tensor Shape: {self.dest_memory.contents[t]['presence_matrix'].shape}" for t in tensors_with_data]
                        # TODO investigate further if this still happens..
                        print(f"Memory thrashing detected: All tensors ({'; '.join(tensor_info_list)}) to be fetched need to free up space, but that would cause an infinite loop of free and fetch.")
                        print(f"The current memory size for {self.dest_memory.name} of {self.dest_memory.size} words may be too small.")
                        raise MemoryThrashingError("Memory thrashing detected and simulation aborted.")

                    # Reschedule this event
                    self.src_memory.pending_actions.append(self.clone_for_retry())
                    if self.verbose:
                        print(f"Write back or fetch in progress for {self.data_id}. Adding fetch to wait list.")
                    return
                
                # Else proceed to write-back
                victim_data_id, victim_data_category, victim_is_broadcasted, victim_tensor_shape, victim_layout_info, victim_layout, victim_offset, victim_data_bitwidth = write_info

                # First write back the chosen victim data to the upper memory
                if self.verbose:
                    print(f"Scheduled write-back of {victim_data_id}'s {victim_layout} elements from {self.dest_memory.name} to {self.src_memory.name} to free space.")
                scheduler.schedule_event(self.event_time, MemoryWriteBackEvent(victim_layout_info, victim_data_id, victim_data_category, victim_is_broadcasted, victim_data_bitwidth, victim_tensor_shape, victim_layout, victim_offset, self.dest_memory, self.src_memory, self.op_priority, self.operation, next_event=self.clone_for_retry(), verbose=self.verbose), self.op_priority)
        else:  # Read from upper memory failed.. schedule another fetch from the upper memory of this upper memory
            missing_data, missing_data_offset = read_info

            # Free the ports
            self.src_memory.free_port(self.src_port)
            self.src_port = None
            while self.src_memory.pending_actions:
                src_memory_op = self.src_memory.pending_actions.pop(0)
                _remove_from_pending(src_memory_op)
                scheduler.schedule_event(self.event_time, src_memory_op, src_memory_op.op_priority)

            self.dest_memory.free_port(self.dest_port)
            self.dest_port = None
            while self.dest_memory.pending_actions:
                dest_memory_op = self.dest_memory.pending_actions.pop(0)
                _remove_from_pending(dest_memory_op)
                scheduler.schedule_event(self.event_time, dest_memory_op, dest_memory_op.op_priority)

            mem_found, mem_miss_data = self.src_memory.query_lower_memories(read_mode="read_tile", data_id=self.data_id, tile_shape=missing_data, offset=missing_data_offset)
            missing_data, missing_data_offset = mem_miss_data

            if mem_found:
                scheduler.schedule_event(self.event_time, MemoryFetchEvent(self.data_id, self.data_category, self.is_data_broadcasted, self.data_bitwidth, self.tensor_shape, missing_data, missing_data_offset, mem_found, mem_found.upper_level_memory, self.op_priority, self.operation, self.clone_for_retry(), verbose=self.verbose), self.op_priority)
            # If not recursively present in any lower memory, try accessing the data from higher-level memory
            else:
                if self.src_memory.upper_level_memory:
                    if self.verbose:
                        print(f"Scheduled fetch of {self.data_id} from {self.src_memory.upper_level_memory.name} to {self.src_memory.name}.")
                    scheduler.schedule_event(self.event_time, MemoryFetchEvent(self.data_id, self.data_category, self.is_data_broadcasted, self.data_bitwidth, self.tensor_shape, missing_data, missing_data_offset, self.src_memory.upper_level_memory, self.src_memory, self.op_priority, self.operation, self.clone_for_retry(), verbose=self.verbose), self.op_priority)
                else:
                    if self.verbose:
                        print(f"Failed to fetch {self.data_id} from {self.src_memory.name}'s upper memory - no upper memory found containing this data.")


class MemoryFetchCompleteEvent(AbstractEvent):
    def __init__(self, data_id, data_bitwidth, src_memory, src_port, dest_memory, dest_port, tensor_shape, tile_shape, offset, op_priority, operation, cycles, next_event = None, verbose: bool = False):
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
        self.op_priority = op_priority
        self.operation = operation
        self.cycles = cycles
        self.next_event = next_event

    def __str__(self):
        return f"MemoryFetchCompleteEvent(data_id={self.data_id}, data_bitwidth={self.data_bitwidth}, src_memory={self.src_memory.name}, src_port={self.src_port}, dest_memory={self.dest_memory.name}, dest_port={self.dest_port}, tensor_shape={self.tensor_shape}, tile_shape={self.tile_shape}, offset={self.offset}, op_priority={self.op_priority}, operation={self.operation}, cycles={self.cycles}, next_event={self.next_event})"

    def execute(self, scheduler):
        self.src_memory.free_port(self.src_port)
        while self.src_memory.pending_actions:
            src_memory_op = self.src_memory.pending_actions.pop(0)
            _remove_from_pending(src_memory_op)
            scheduler.schedule_event(self.event_time, src_memory_op, src_memory_op.op_priority)

        self.dest_memory.free_port(self.dest_port)
        while self.dest_memory.pending_actions:
            dest_memory_op = self.dest_memory.pending_actions.pop(0)
            _remove_from_pending(dest_memory_op)
            scheduler.schedule_event(self.event_time, dest_memory_op, dest_memory_op.op_priority)
        
        if len(self.src_memory.available_ports) == self.src_memory.ports:
            self.src_memory.synchronize_per_ports_cycles()
        if len(self.dest_memory.available_ports) == self.dest_memory.ports:
            self.dest_memory.synchronize_per_ports_cycles()

        # Unlocking the fetch and processing queued reads (key matches (data_id, offset) lock set in generic_memory.read())
        _lock_key = (self.data_id, self.offset)
        waiting_reads = self.dest_memory.fetch_locks.pop(_lock_key, [])
        self.dest_memory.unlock_fetch_lock(_lock_key)
        for fetch_event in waiting_reads:
            if self.verbose:
                print(f"Rescheduled waiting fetch of {self.data_id} into {self.dest_memory.name}.")
            scheduler.schedule_event(self.event_time, fetch_event, fetch_event.op_priority)
        if self.operation is not None:
            kind = "dram" if isinstance(self.src_memory, OffChipMemory) else "onchip"
            self.operation._phase_log.append((self.event_time - self.cycles, self.event_time, kind))
        if self.verbose:
            print(f"Completed fetch of {self.data_id} from {self.src_memory.name} to {self.dest_memory.name}.")
        if self.next_event:
            # If the next event is TemporalTileComputeEvent, ensure all reads are complete before triggering it
            if isinstance(self.next_event, TemporalTileComputeEvent):
                self.next_event.pending_reads -= 1
                if self.next_event.pending_reads == 0:
                    scheduler.schedule_event(self.event_time, self.next_event, self.next_event.op_priority)
            else:
                scheduler.schedule_event(self.event_time, self.next_event, self.next_event.op_priority)


class MemoryWriteEvent(AbstractEvent):
    def __init__(self, data_id, data_category, is_data_broadcasted, data_bitwidth, tensor_shape, tile_shape, offset, memory, op_priority, operation, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.data_id = data_id
        self.data_category = data_category
        self.is_data_broadcasted = is_data_broadcasted
        self.data_bitwidth = data_bitwidth
        self.tensor_shape = tensor_shape
        self.tile_shape = tile_shape
        self.offset = offset
        self.memory = memory
        self.memory_port = None
        self.op_priority = op_priority
        self.operation = operation
        self.next_event = next_event

    def __str__(self):
        return f"MemoryWriteEvent(data_id={self.data_id}, is_data_broadcasted={self.is_data_broadcasted}, data_category={self.data_category}, data_bitwidth={self.data_bitwidth}, tensor_shape={self.tensor_shape}, tile_shape={self.tile_shape}, offset={self.offset}, memory={self.memory.name}, op_priority={self.op_priority}, operation={self.operation}, next_event={self.next_event})"

    def clone_for_retry(self):
        return MemoryWriteEvent(
            data_id=self.data_id,
            data_category=self.data_category,
            is_data_broadcasted=self.is_data_broadcasted,
            data_bitwidth=self.data_bitwidth,
            tensor_shape=self.tensor_shape,
            tile_shape=self.tile_shape,
            offset=self.offset,
            memory=self.memory,
            op_priority=self.op_priority,
            operation=self.operation,
            next_event=self.next_event,
            verbose=self.verbose,
        )

    def execute(self, scheduler):
        if not self.memory.available_ports:
            self.memory.pending_actions.append(self.clone_for_retry())
            if self.verbose:
                print(f"Memory {self.memory.name} ports are busy. Rescheduling write of {self.data_id} to pending actions.")
            return
        
        self.memory_port = self.memory.get_available_port()

        success, write_info = self.memory.write(write_mode="write_tile", data_id=self.data_id, data_category=self.data_category, is_data_broadcasted=self.is_data_broadcasted, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.memory_port, tensors_needed_info=scheduler.execution_graph.tensors_needed, tile_shape=self.tile_shape, offset=self.offset, verbose=self.verbose)

        if success:
            cycles = write_info
            self.memory.update_per_ports_cycles(self.memory_port, cycles, update_idle=False)
            if self.verbose:
                print(f"Wrote {self.data_id} to {self.memory.name} in {cycles} cycles and in global time {self.event_time + cycles}.")
            scheduler.schedule_event(self.event_time + cycles, MemoryWriteCompleteEvent(self.data_id, self.data_bitwidth, self.memory, self.memory_port, self.op_priority, self.operation, self.tensor_shape, self.tile_shape, self.offset, cycles, self.next_event, verbose=self.verbose), self.op_priority)

            # Additional write-back to higher-level memory for final output data
            if self.operation and self.operation._is_root and not (isinstance(self.memory, OffChipMemory)):
                assert self.memory.upper_level_memory, f"Final operation cannot be written to offchip since {self.memory.name} has no associated upper memory to write to!"
                # TODO layout info!!
                scheduler.schedule_event(self.event_time + cycles, MemoryWriteBackEvent("tile", self.data_id, self.data_category, self.is_data_broadcasted, self.data_bitwidth, self.tensor_shape, self.tile_shape, self.offset, self.memory, self.memory.upper_level_memory, self.op_priority, self.operation, next_event=self.next_event, verbose=self.verbose), self.op_priority)
        else:  # Write failed, we need to proceed with write-back of the chosen victim data to upper memory
            # Free the port
            self.memory.free_port(self.memory_port)
            self.memory_port = None
            while self.memory.pending_actions:
                memory_op = self.memory.pending_actions.pop(0)
                _remove_from_pending(memory_op)
                scheduler.schedule_event(self.event_time, memory_op, memory_op.op_priority)
            
            # Check if the operation could not proceed because different write-back or fetch operation is currently in action for the chosen data id
            if write_info is None:
                # Reschedule this event
                self.memory.pending_actions.append(self.clone_for_retry())
                if self.verbose:
                    print(f"Write back or fetch in progress for {self.data_id}. Adding write to wait list.")
                return

            # Else proceed to write-back
            victim_data_id, victim_data_category, victim_is_broadcasted, victim_tensor_shape, victim_layout_info, victim_layout, victim_offset, victim_data_bitwidth = write_info

            # Write back the chosen victim data to the upper memory first
            if self.memory.upper_level_memory:
                if self.verbose:
                    print(f"Scheduled write-back of {victim_data_id}'s {victim_layout} elements from {self.memory.name} to {self.memory.upper_level_memory.name} to free space.")
                scheduler.schedule_event(self.event_time, MemoryWriteBackEvent(victim_layout_info, victim_data_id, victim_data_category, victim_is_broadcasted, victim_data_bitwidth, victim_tensor_shape, victim_layout, victim_offset, self.memory, self.memory.upper_level_memory, self.op_priority, self.operation, next_event=self.clone_for_retry(), verbose=self.verbose), self.op_priority)
            else:
                if self.verbose:
                    print(f"Failed to write-back {victim_data_id} from {self.memory.name} to its upper memory - no attached upper memory found.")


class MemoryWriteCompleteEvent(AbstractEvent):
    def __init__(self, data_id, data_bitwidth, memory, memory_port, op_priority, operation, tensor_shape, tile_shape, offset, cycles, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.data_id = data_id
        self.data_bitwidth = data_bitwidth
        self.tensor_shape = tensor_shape
        self.tile_shape = tile_shape
        self.offset = offset
        self.op_priority = op_priority
        self.operation = operation
        self.memory = memory
        self.memory_port = memory_port
        self.cycles = cycles
        self.next_event = next_event

    def __str__(self):
        return f"MemoryWriteCompleteEvent(data_id={self.data_id}, data_bitwidth={self.data_bitwidth}, memory={self.memory.name}, memory_port={self.memory_port}, op_priority={self.op_priority}, operation={self.operation}, cycles={self.cycles}, next_event={self.next_event})"

    def execute(self, scheduler):
        self.memory.free_port(self.memory_port)
        while self.memory.pending_actions:
            memory_op = self.memory.pending_actions.pop(0)
            _remove_from_pending(memory_op)
            scheduler.schedule_event(self.event_time, memory_op, memory_op.op_priority)
        
        if len(self.memory.available_ports) == self.memory.ports:
            self.memory.synchronize_per_ports_cycles()
            
        if self.operation is not None:
            kind = "dram" if isinstance(self.memory, OffChipMemory) else "onchip"
            self.operation._phase_log.append((self.event_time - self.cycles, self.event_time, kind))
        if self.verbose:
            print(f"Completed write of {self.data_id} into {self.memory.name}.")
        if self.next_event and ((isinstance(self.memory, OffChipMemory) and self.operation and self.operation._is_root) or not (self.operation and self.operation._is_root)):
            # We need to modify the event time since the next event is typically a MatmulStartEvent/SpatialTileComputeEvent from which we wish to continue with next tile
            self.next_event.event_time = self.event_time
            self.next_event.schedule_next_tile(scheduler)


class MemoryWriteBackEvent(AbstractEvent):
    def __init__(self, data_layout_info, data_id, data_category, is_data_broadcasted, data_bitwidth, tensor_shape, data_layout, offset, src_memory, dest_memory, op_priority, operation, next_event=None, verbose: bool = False):
        super().__init__(verbose)
        self.data_layout_info = data_layout_info
        assert data_layout_info == "contiguous" or data_layout_info == "tile", "Invalid data layout type for MemoryWriteBackEvent. It should be either 'contiguous' or 'tile'."
        self.data_id = data_id
        self.data_category = data_category
        self.is_data_broadcasted = is_data_broadcasted
        self.data_bitwidth = data_bitwidth
        self.tensor_shape = tensor_shape
        self.data_layout = data_layout
        self.offset = offset
        self.src_memory = src_memory
        self.src_port = None
        self.dest_memory = dest_memory
        self.dest_port = None
        self.op_priority = op_priority
        self.operation = operation
        self.next_event = next_event

    def __str__(self):
        return f"MemoryWriteBackEvent(data_layout_info={self.data_layout_info}, data_id={self.data_id}, data_category={self.data_category}, is_data_broadcasted={self.is_data_broadcasted}, data_bitwidth={self.data_bitwidth}, tensor_shape={self.tensor_shape}, data_layout={self.data_layout}, offset={self.offset}, src_memory={self.src_memory.name}, dest_memory={self.dest_memory.name}, op_priority={self.op_priority}, operation={self.operation}, next_event={self.next_event})"

    def clone_for_retry(self):
        return MemoryWriteBackEvent(
            data_layout_info=self.data_layout_info,
            data_id=self.data_id,
            data_category=self.data_category,
            is_data_broadcasted=self.is_data_broadcasted,
            data_bitwidth=self.data_bitwidth,
            tensor_shape=self.tensor_shape,
            data_layout=self.data_layout,
            offset=self.offset,
            src_memory=self.src_memory,
            dest_memory=self.dest_memory,
            op_priority=self.op_priority,
            operation=self.operation,
            next_event=self.next_event,
            verbose=self.verbose,
        )

    def execute(self, scheduler):
        if not self.src_memory.available_ports or not self.dest_memory.available_ports:
            self.src_memory.pending_actions.append(self.clone_for_retry())
            if self.verbose:
                print(f"Ports of memories (src) {self.src_memory.name} and (dest) {self.dest_memory.name} are busy. Rescheduling write back of {self.data_id} to pending actions.")
            return
        self.src_port = self.src_memory.get_available_port()
        self.dest_port = self.dest_memory.get_available_port()

        if self.data_layout_info == "tile":
            read_success, read_cycles = self.src_memory.read(read_mode="read_tile", data_id=self.data_id, data_category=self.data_category, is_data_broadcasted=self.is_data_broadcasted, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.src_port, tile_shape=self.data_layout, offset=self.offset)
        else:
            read_success, read_cycles = self.src_memory.read(read_mode="read_elements", data_id=self.data_id, data_category=self.data_category, is_data_broadcasted=self.is_data_broadcasted, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.src_port, elems_to_read=self.data_layout, offset=self.offset)
        
        assert read_success, "Failed to read data for write-back from source memory. This should never happen. Probably data was not found in the source memory."

        if self.data_layout_info == "tile":
            write_success, write_info = self.dest_memory.write(write_mode="write_tile", data_id=self.data_id, data_category=self.data_category, is_data_broadcasted=self.is_data_broadcasted, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.dest_port, tensors_needed_info=scheduler.execution_graph.tensors_needed, tile_shape=self.data_layout, offset=self.offset, verbose=self.verbose)
        else: 
            write_success, write_info = self.dest_memory.write(write_mode="write_elements", data_id=self.data_id, data_category=self.data_category, is_data_broadcasted=self.is_data_broadcasted, tensor_shape=self.tensor_shape, data_bitwidth=self.data_bitwidth, port_id=self.dest_port, tensors_needed_info=scheduler.execution_graph.tensors_needed, elems_to_write=self.data_layout, offset=self.offset, verbose=self.verbose)

        if write_success:
            # Update action cycles during read/write            
            self.src_memory.update_per_ports_cycles(self.src_port, read_cycles, update_idle=False)
            self.dest_memory.update_per_ports_cycles(self.dest_port, read_cycles, update_idle=False)
            if self.verbose:
                print(f"Wrote back {self.data_id} from {self.src_memory.name} to {self.dest_memory.name} in {read_cycles} cycles and in global time {self.event_time + read_cycles}.")

            if self.operation and self.operation._is_root and self.data_id == scheduler.execution_graph.root.output and not (isinstance(self.dest_memory, OffChipMemory)):  # Final write back must complete all the way to the off-chip memory
                next_write_back_event = MemoryWriteBackEvent(self.data_layout_info, self.data_id, self.data_category, self.is_data_broadcasted, self.data_bitwidth, self.tensor_shape, self.data_layout, self.offset, self.dest_memory, self.dest_memory.upper_level_memory, self.op_priority, self.operation, next_event=self.next_event, verbose=self.verbose)
                scheduler.schedule_event(self.event_time + read_cycles, MemoryWriteBackCompleteEvent(self.data_id, self.data_bitwidth, self.src_memory, self.src_port, self.dest_memory, self.dest_port, self.tensor_shape, self.data_layout_info, self.offset, self.op_priority, self.operation, read_cycles, next_write_back_event, verbose=self.verbose), self.op_priority)
            else:
                scheduler.schedule_event(self.event_time + read_cycles, MemoryWriteBackCompleteEvent(self.data_id, self.data_bitwidth, self.src_memory, self.src_port, self.dest_memory, self.dest_port, self.tensor_shape, self.data_layout_info, self.offset, self.op_priority, self.operation, read_cycles, self.next_event, verbose=self.verbose), self.op_priority)
        else:  # Cannot write back the data to upper memory, because its memory is full.. we need to write-back the victim data chosen from it and write it back to its upper memory before retrying this write-back
            # Free the ports
            self.src_memory.free_port(self.src_port)
            self.src_port = None
            while self.src_memory.pending_actions:
                src_memory_op = self.src_memory.pending_actions.pop(0)
                _remove_from_pending(src_memory_op)
                scheduler.schedule_event(self.event_time, src_memory_op, src_memory_op.op_priority)

            self.dest_memory.free_port(self.dest_port)
            self.dest_port = None
            while self.dest_memory.pending_actions:
                dest_memory_op = self.dest_memory.pending_actions.pop(0)
                _remove_from_pending(dest_memory_op)
                scheduler.schedule_event(self.event_time, dest_memory_op, dest_memory_op.op_priority)
            
            # Check if the operation could not proceed because different write-back or fetch operation is currently in action for the chosen dat id
            if write_info is None:
                # Reschedule this event
                self.src_memory.pending_actions.append(self.clone_for_retry())
                if self.verbose:
                    print(f"Write back or fetch in progress for {self.data_id}. Adding free/write-back to wait list.")
                return

            # Else proceed to write-back
            victim_data_id, victim_data_category, victim_is_broadcasted, victim_tensor_shape, victim_layout_info, victim_layout, victim_offset, victim_data_bitwidth = write_info
            # TODO REFACTOR STATS INTO TUPLE...
            
            # Write back the victim data from this memory's upper memory to its upper memory first
            if self.dest_memory.upper_level_memory:
                if self.verbose:
                    print(f"Scheduled write-back of {victim_data_id}'s {victim_layout} elements from {self.dest_memory.name} to {self.dest_memory.upper_level_memory.name} to free space.")
                scheduler.schedule_event(self.event_time, MemoryWriteBackEvent(victim_layout_info, victim_data_id, victim_data_category, victim_is_broadcasted, victim_data_bitwidth, victim_tensor_shape, victim_layout, victim_offset, self.dest_memory, self.dest_memory.upper_level_memory, self.op_priority, self.operation, next_event=self.clone_for_retry(), verbose=self.verbose), self.op_priority)
            else:
                if self.verbose:
                    print(f"Failed to write-back {victim_data_id} from {self.dest_memory.name} to its upper memory - no attached upper memory found.")


class MemoryWriteBackCompleteEvent(AbstractEvent):
    def __init__(self, data_id, data_bitwidth, src_memory, src_port, dest_memory, dest_port, tensor_shape, data_layout_info, offset, op_priority, operation, cycles, next_event = None, verbose: bool = False):
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
        self.op_priority = op_priority
        self.operation = operation
        self.cycles = cycles
        self.next_event = next_event

    def __str__(self):
        return f"MemoryWriteBackCompleteEvent(data_id={self.data_id}, data_bitwidth={self.data_bitwidth}, src_memory={self.src_memory.name}, src_port={self.src_port}, dest_memory={self.dest_memory.name}, dest_port={self.dest_port}, tensor_shape={self.tensor_shape}, data_layout_info={self.data_layout_info}, offset={self.offset}, op_priority={self.op_priority}, operation={self.operation}, cycles={self.cycles}, next_event={self.next_event})"

    def execute(self, scheduler):
        self.src_memory.unlock_free_lock(self.data_id)

        self.src_memory.free_port(self.src_port)
        while self.src_memory.pending_actions:
            src_memory_op = self.src_memory.pending_actions.pop(0)
            _remove_from_pending(src_memory_op)
            scheduler.schedule_event(self.event_time, src_memory_op, src_memory_op.op_priority)

        self.dest_memory.free_port(self.dest_port)
        while self.dest_memory.pending_actions:
            dest_memory_op = self.dest_memory.pending_actions.pop(0)
            _remove_from_pending(dest_memory_op)
            scheduler.schedule_event(self.event_time, dest_memory_op, dest_memory_op.op_priority)
        
        if len(self.src_memory.available_ports) == self.src_memory.ports:
            self.src_memory.synchronize_per_ports_cycles()
        if len(self.dest_memory.available_ports) == self.dest_memory.ports:
            self.dest_memory.synchronize_per_ports_cycles()

        if self.operation is not None:
            kind = "dram" if isinstance(self.dest_memory, OffChipMemory) else "onchip"
            self.operation._phase_log.append((self.event_time - self.cycles, self.event_time, kind))
        if self.verbose:
            print(f"Completed write back of {self.data_id} into {self.dest_memory.name}.")
        if self.next_event:
            if isinstance(self.next_event, (MatmulStartEvent, SpatialTileComputeEvent)):
                params = [{"param": "event_time", "value": self.next_event.event_time}]
                if isinstance(self.next_event, SpatialTileComputeEvent):
                    params.append({"param": "current_temp_tile", "value": self.next_event.current_temp_tile})
                elif isinstance(self.next_event, MatmulStartEvent):
                    params.append({"param": "current_tile", "value": self.next_event.current_tile})
                # We need to modify the event time since the next event is typically a MatmulStartEvent/SpatialTileComputeEvent from which we wish to continue with next tile
                self.next_event.event_time = self.event_time
                self.next_event.schedule_next_tile(scheduler)
            else:                
                scheduler.schedule_event(self.event_time, self.next_event, self.next_event.op_priority)


class ConcatenationStartEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, verbose:bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.op_priority = operation.priority
        self.matmul_array = matmul_array
        # todo check/fix concat retraction?

    def __str__(self):
        return f"ConcatenationStartEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name})"

    def execute(self, scheduler):
        def compute_tile_offsets(input_data_list, final_shape):
            """
            input_data_list: The list of dicts [{'name': '...', 'data': {...}}, ...]
            Returns: A LIST of (row, col) tuples matching the input list order.
            """
            offsets = []
            row_cursor = 0
            col_cursor = 0
            max_rows, max_cols = final_shape
            current_row_height = 0
            
            for item in input_data_list:
                # Get dimensions from the new nested structure
                tile_rows, tile_cols = item['data']['dimensions']

                # Wrap to next row if needed
                if col_cursor + tile_cols > max_cols:
                    row_cursor += current_row_height
                    col_cursor = 0
                    current_row_height = 0

                if row_cursor + tile_rows > max_rows:
                    raise ValueError(f"Tile {item['name']} doesn't fit in output matrix.")

                # Append offset to a list so duplicates are preserved
                offsets.append((row_cursor, col_cursor))
                
                col_cursor += tile_cols
                current_row_height = max(current_row_height, tile_rows)
                
            return offsets

        # # Retrieve list of all memory blocks and update the concatenad blocks in all of them (some parts may have already been written to upper memory)
        accelerator = self.matmul_array.parent_component
        memory_blocks = accelerator.memory_blocks + [accelerator.dram]
        all_input_names = [item['name'] for item in self.operation.input_data]
        assert len(all_input_names) == len(set(all_input_names)), "Concatenation operation requires all input tensors to be unique."

        for mem in memory_blocks:
            mem_contents = mem.contents
            if any(name in mem_contents for name in all_input_names):
                subtiles_for_writeback = {}
                
                # Initialize variables for the new concatenated matrix
                new_data_id = self.operation.output_data[0]['name']
                output_item = self.operation.output_data[0]['data']
                tensor_shape = output_item['dimensions']
                presence_matrix = mem._initialize_presence_matrix(tensor_shape, 0)
                data_category = output_item['data_category']
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
                tile_offsets = compute_tile_offsets(self.operation.input_data, presence_matrix.shape)                

                # TODO refactor with offsets (should be more robust and better for irregular subtiles) info from operation splitting? should be faster and more elegant                
                for idx, input_item in enumerate(self.operation.input_data):
                    data_id = input_item['name']
                    if data_id in mem_contents:
                        row_off, col_off = tile_offsets[idx]
                        tile_rows, tile_cols = input_item['data']['dimensions']
                        
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
                    'broadcasted_view': False,
                    'tensor_shape': tensor_shape,
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
                        scheduler.schedule_event(self.event_time, MemoryWriteBackEvent("tile", new_data_id, data_category, False, self.operation.data_bitwidth, presence_matrix.shape, subtile['tile'], subtile['offset'], mem, accelerator.dram, self.op_priority, self.operation, next_event=None, verbose=self.verbose), self.op_priority)

                if self.verbose:
                    print(f"Replaced submatrices with {new_data_id} in memory {mem.name}.")

        # The original data IDs are now replaced with the new concatenated data ID
        for input_item in self.operation.input_data:
            data_id = input_item['name']
            scheduler.execution_graph.tensors_needed.decrease_count(data_id)

        if self.operation._is_root:  # If final operation, the output is successfully in DRAM
            scheduler.execution_graph.tensors_needed.decrease_count(new_data_id)
        
        # Schedule completion handler
        scheduler.schedule_event(self.event_time, ConcatenationCompleteEvent(self.operation, self.matmul_array, verbose=self.verbose), self.op_priority)

class ConcatenationCompleteEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.matmul_array = matmul_array
    
    def execute(self, scheduler):
        # Remove the operation from the parents children (dependencies) list (list of operations which produce input data for the parent)
        self.operation.remove_dependency()
        self.operation.is_done = True
        self.operation.finish_time = self.event_time
        for p in self.operation.parents:
            # If this operation is the last child of the parent, and the parent's event is pending, finally reschedule it
            if not p.children and p.pending_event is not None:
                scheduler.schedule_event(self.event_time, p.pending_event, p.pending_event.op_priority)

        if self.verbose:
            print(f"Concatenated tensor data of '{self.operation.output_data[0]['name']}' in global time {self.event_time}.")
        
        assert self.matmul_array.is_busy == False, f"Matmul array '{self.matmul_array.name}' is still busy, cannot schedule next operation! This shouldn't happen."
        while self.matmul_array.plan:
            next_operation = self.matmul_array.plan.pop(0)
            if next_operation.is_done:  # Skip operations that are already done (may occur during step-wise simulation)
                continue

            # TODO add and test row/col majorness
            next_start_event = MatmulStartEvent(next_operation, self.matmul_array, scheduler, row_major=True, verbose=self.verbose, log_state_before=scheduler.log_state_before)
            scheduler.schedule_event(self.event_time, next_start_event, next_start_event.op_priority)
            break













""" TODO TEST """

# goal.. similar as with concatenation.. only assert.. all subtensors same name!
# rename the one subtensor with the newly created "broadcasted view"
# physically it will have same stats and same info... but the logical view will be different
# i.e. same presence matrix as with original.. but when read/write... we should "extend"/"shrink" the view to map the physical storage!
# needed and tested on gqa..

class BroadcastStartEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.op_priority = operation.priority
        self.matmul_array = matmul_array

    def __str__(self):
        return f"BroadcastStartEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name})"

    def execute(self, scheduler):
        # ASSERT BROADCAST VALIDITY
        input_names = [item['name'] for item in self.operation.input_data]
        assert len(set(input_names)) == 1, "Broadcast operation requires all input tensors to have the same name."

        input_data_id = input_names[0]
        new_data_id = self.operation.output_data[0]['name']
        output_item = self.operation.output_data[0]['data']
        broadcasted_shape = output_item['dimensions']

        accelerator = self.matmul_array.parent_component
        memory_blocks = accelerator.memory_blocks + [accelerator.dram]

        for mem in memory_blocks:
            mem_contents = mem.contents

            if input_data_id not in mem_contents:
                continue

            source_tensor = mem_contents[input_data_id]
            # Create broadcasted VIEW (no physical duplication)
            mem_contents[new_data_id] = {
                'data_category': source_tensor['data_category'],
                'broadcasted_view': True,
                'tensor_shape': broadcasted_shape,                    # logical shape changes
                'presence_matrix': source_tensor['presence_matrix'],  # same physical presence
                'data_amount': source_tensor['data_amount'],
                'data_read_count': source_tensor['data_read_count'],
                'mem_read_count': source_tensor['mem_read_count'],
                'data_write_count': source_tensor['data_write_count'],
                'mem_write_count': source_tensor['mem_write_count'],
                'word_amount': source_tensor['word_amount'],
                'word_read_count': source_tensor['word_read_count'],
                'word_write_count': source_tensor['word_write_count'],
                'data_bit_packing': source_tensor['data_bit_packing'],
                'data_bitwidth': source_tensor['data_bitwidth'],
                'fragmented_bits': source_tensor['fragmented_bits'],
                'insertion_time': source_tensor['insertion_time'],
                'last_access_time': source_tensor['last_access_time'],
                'cache_miss_count': source_tensor['cache_miss_count']
            }
            mem_contents.pop(input_data_id)  # Remove the submatrix entry if it is not needed anymore

            if self.verbose:
                print(f"Created broadcasted view '{new_data_id}' from '{input_data_id}' in memory {mem.name}.")

        scheduler.execution_graph.tensors_needed.decrease_count(input_data_id)

        if self.operation._is_root:
            scheduler.execution_graph.tensors_needed.decrease_count(new_data_id)

        scheduler.schedule_event(self.event_time, BroadcastCompleteEvent(self.operation, self.matmul_array, verbose=self.verbose), self.op_priority)


class BroadcastCompleteEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.matmul_array = matmul_array

    def execute(self, scheduler):
        # Remove dependency links
        self.operation.remove_dependency()
        self.operation.is_done = True
        self.operation.finish_time = self.event_time

        for p in self.operation.parents:
            if not p.children and p.pending_event is not None:
                scheduler.schedule_event(self.event_time, p.pending_event, p.pending_event.op_priority)

        if self.verbose:
            print(f"Broadcasted tensor view '{self.operation.output_data[0]['name']}' in global time {self.event_time}.")

        assert self.matmul_array.is_busy == False, f"Matmul array '{self.matmul_array.name}' is still busy, cannot schedule next operation!"

        while self.matmul_array.plan:
            next_operation = self.matmul_array.plan.pop(0)
            if next_operation.is_done:  # Skip operations that are already done (may occur during step-wise simulation)
                continue

            next_start_event = MatmulStartEvent(next_operation, self.matmul_array, scheduler, row_major=True, verbose=self.verbose, log_state_before=scheduler.log_state_before)
            scheduler.schedule_event(self.event_time, next_start_event, next_start_event.op_priority)
            break


""" MATMUL EVENTS """
class MatmulStartEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, scheduler, is_rescheduled: bool = False, row_major: bool = True, verbose: bool = False, log_state_before: bool = False):
        super().__init__(verbose, log_state_before)
        self.operation = operation
        self.op_priority = operation.priority
        self.matmul_array = matmul_array
        self.num_row_tiles = 0
        self.num_col_tiles = 0
        self.matmul_array.is_busy = True
        self.row_major = row_major  # Specifies order of processing subtiles (row/col major)
        self._scheduler = scheduler
        self.is_rescheduled = is_rescheduled
        self.matmul_array.current_operation_event = self
        assert self.operation.is_done == False
        if self.log_state_before:
            self.state_before_changes = {"components": [{"obj": self.matmul_array, "params": [
                                            {"param": "is_busy", "value": False},  # i.e. compute_done()
                                            {"param": "current_operation_event", "value": None},  # i.e. compute_done()
                                            {"param": "global_cycles", "value": self.matmul_array.global_cycles},
                                            {"param": "active_cycles", "value": self.matmul_array.active_cycles},
                                            {"param": "total_flop_computes", "value": self.matmul_array.total_flop_computes},
                                            {"param": "accumulator_reads", "value": self.matmul_array.accumulator_reads},
                                            {"param": "accumulator_writes", "value": self.matmul_array.accumulator_writes},
                                            {"param": "pes_computational_cycles", "value": self.matmul_array.pes_computational_cycles},
                                            {"param": "pes_idle_cycles", "value": self.matmul_array.pes_idle_cycles},
                                            {"param": "energy", "value": self.matmul_array.energy},
                                            {"param": "area", "value": self.matmul_array.area},
                                            ], "contents": [], "method_calls": []},
                                                        {"obj": scheduler, "params": [
                                            {"param": "global_cycles", "value": scheduler.global_cycles},
                                            {"param": "_next_sequence_id", "value": scheduler._next_sequence_id},
                                            {"param": "events", "value": scheduler.events.copy()}
                                            ], "contents": [], "method_calls": []},
                                                        {"obj": self.operation, "params": [
                                            {"param": "is_done", "value": False},
                                            {"param": "pending_event", "value": self.operation.pending_event},
                                            {"param": "start_time", "value": self.operation.start_time},
                                            {"param": "finish_time", "value": self.operation.finish_time},
                                            {"param": "macs", "value": self.operation.macs},
                                            {"param": "compute_time", "value": self.operation.compute_time},
                                            {"param": "_phase_log", "value": self.operation._phase_log.copy()},
                                            ], "contents": [], "method_calls": [{"method": "readd_dependency", "args": [], "kwargs": {}}]},
                                                        ], "params": []}
            
            self.state_before_changes["components"].append({"obj": scheduler.execution_graph.tensors_needed, "params": [], "contents": [], "method_calls": []})
            for input_item in self.operation.input_data:
                tensor_name = input_item['name']
                self.state_before_changes["components"][3]["method_calls"].append({"method": "increase_count", "args": [tensor_name], "kwargs": {}})

            if self.operation._is_root and self.operation.output_data:
                self.state_before_changes["components"][3]["method_calls"].append({"method": "increase_count", "args": [self.operation.output_data[0]['name']], "kwargs": {}})

            accelerator = self.matmul_array.parent_component
            self.state_before_changes["components"].append({"obj": accelerator, "params": [{"param": "energy", "value": accelerator.energy}, {"param": "area", "value": accelerator.area}], "contents": [], "method_calls": []})

            # WE NEED TO BE MORE ROBUST HERE... WE CAN RETRACT TO BEFORE CURRENT MATMUL, WHILST A DIFFERENT MIGHT BE IN MIDDLE OF SPATIALTILE/TEMPTILE... AND THE STATES WILL BE LOST!
            for sa in accelerator.matmul_blocks:
                if sa is not self.matmul_array:
                    array_comp_state = {"obj": sa, "params": [], "contents": [], "method_calls": []}
                    array_comp_state["params"].extend([{"param": "is_busy", "value": sa.is_busy}])
                    array_comp_state["params"].extend([{"param": "current_operation_event", "value": sa.current_operation_event}])
                    array_comp_state["params"].extend([{"param": "global_cycles", "value": sa.global_cycles}])
                    array_comp_state["params"].extend([{"param": "active_cycles", "value": sa.active_cycles}])
                    array_comp_state["params"].extend([{"param": "total_flop_computes", "value": sa.total_flop_computes}])
                    array_comp_state["params"].extend([{"param": "accumulator_reads", "value": sa.accumulator_reads}])
                    array_comp_state["params"].extend([{"param": "accumulator_writes", "value": sa.accumulator_writes}])
                    array_comp_state["params"].extend([{"param": "pes_computational_cycles", "value": sa.pes_computational_cycles}])
                    array_comp_state["params"].extend([{"param": "pes_idle_cycles", "value": sa.pes_idle_cycles}])
                    array_comp_state["params"].extend([{"param": "energy", "value": sa.energy}])
                    array_comp_state["params"].extend([{"param": "area", "value": sa.area}])
                    
                    sa_event = sa.current_operation_event
                    if sa_event:
                        array_event_comp_state = {"obj": sa_event, "params": [], "contents": [], "method_calls": []}
                        if hasattr(sa_event, "current_tile"):
                            array_event_comp_state["params"].append({"param": "current_tile", "value": sa_event.current_tile})
                        if hasattr(sa_event, "_cycles"):
                            array_event_comp_state["params"].append({"param": "_cycles", "value": sa_event._cycles})

                        
                        
                        if hasattr(sa_event, "_current_spatial_tile_event"):
                            array_event_comp_state["params"].append({"param": "_current_spatial_tile_event", "value": sa_event._current_spatial_tile_event})
                            spat_tile_e = sa_event._current_spatial_tile_event
                        
                            if spat_tile_e:
                                array_spat_tile_comp_state = {"obj": spat_tile_e, "params": [], "contents": [], "method_calls": []}
                                array_spat_tile_comp_state["params"].extend([{"param": "_cycles", "value": spat_tile_e._cycles}])
                                if hasattr(spat_tile_e, "current_temp_tile"):
                                    array_spat_tile_comp_state["params"].extend([{"param": "current_temp_tile", "value": spat_tile_e.current_temp_tile}])

                                if hasattr(spat_tile_e, "_current_temporal_tile_event"):
                                    temp_tile_e = spat_tile_e._current_temporal_tile_event
                                    array_spat_tile_comp_state["params"].append({"param": "_current_temporal_tile_event", "value": temp_tile_e})
                                    if temp_tile_e:
                                        array_temp_tile_comp_state = {"obj": temp_tile_e, "params": [], "contents": [], "method_calls": []}
                                        array_temp_tile_comp_state["params"].extend([{"param": "pending_reads", "value": temp_tile_e.pending_reads}])
                                        self.state_before_changes["components"].append(array_temp_tile_comp_state)
                                self.state_before_changes["components"].append(array_spat_tile_comp_state)
                        self.state_before_changes["components"].append(array_event_comp_state)
                        
                        # from the view of this SA, if the other SAs are computing, restore their ops info and also the needed tensor!
                        # (because this op still needs to finish execution from this timeline, but is believed to have already finished – tensors otherwise already decreased, same for op dependency)
                        sa_op_comp_state = {"obj": sa_event.operation, "params": [
                            {"param": "is_done", "value": False},
                            {"param": "pending_event", "value": sa_event.operation.pending_event},
                            {"param": "start_time", "value": sa_event.operation.start_time},
                            {"param": "finish_time", "value": sa_event.operation.finish_time},
                            {"param": "compute_time", "value": sa_event.operation.compute_time},
                            {"param": "macs", "value": sa_event.operation.macs},
                            {"param": "_phase_log", "value": sa_event.operation._phase_log.copy()},
                        ], "contents": [], "method_calls": [{"method": "readd_dependency", "args": [], "kwargs": {}}]}
                        self.state_before_changes["components"].append(sa_op_comp_state)
                        
                        for i_data in sa_event.operation.input_data:
                            t_name = i_data['name']
                            self.state_before_changes["components"][3]["method_calls"].append({"method": "increase_count", "args": [t_name], "kwargs": {}})

                        if sa_event.operation._is_root and sa_event.operation.output_data:
                            self.state_before_changes["components"][3]["method_calls"].append({"method": "increase_count", "args": [sa_event.operation.output_data[0]['name']], "kwargs": {}})
                    self.state_before_changes["components"].append(array_comp_state)

            memory_blocks = accelerator.memory_blocks + [accelerator.dram]
            for m in memory_blocks:
                mem_comp_state = {"obj": m, "params": [], "contents": [], "method_calls": []}
                mem_comp_state["contents"] = snapshot_mem_contents(m)
                mem_comp_state["params"].extend([{"param": "area", "value": m.area}])
                mem_comp_state["params"].extend([{"param": "static_energy", "value": m.static_energy}])
                mem_comp_state["params"].extend([{"param": "dynamic_energy", "value": m.dynamic_energy}])
                mem_comp_state["params"].extend([{"param": "current_usage", "value": m.current_usage}])
                mem_comp_state["params"].extend([{"param": "global_cycles", "value": m.global_cycles}])

                mem_comp_state["params"].extend([{"param": "data_read_count", "value": m.data_read_count}])
                mem_comp_state["params"].extend([{"param": "word_read_count", "value": m.word_read_count}])
                mem_comp_state["params"].extend([{"param": "mem_read_count", "value": m.mem_read_count}])
                mem_comp_state["params"].extend([{"param": "data_write_count", "value": m.data_write_count}])
                mem_comp_state["params"].extend([{"param": "mem_write_count", "value": m.mem_write_count}])
                mem_comp_state["params"].extend([{"param": "word_write_count", "value": m.word_write_count}])
                mem_comp_state["params"].extend([{"param": "cache_miss_count", "value": m.cache_miss_count}])
                mem_comp_state["params"].extend([{"param": "fragmented_bits", "value": m.fragmented_bits}])
                mem_comp_state["params"].extend([{"param": "_cycles_per_ports", "value": m._cycles_per_ports.copy()}])
                mem_comp_state["params"].extend([{"param": "pending_actions", "value": m.pending_actions.copy()}])
                mem_comp_state["params"].extend([{"param": "available_ports", "value": m.available_ports.copy()}])
                mem_comp_state["params"].extend([{"param": "fetch_locks", "value": m.fetch_locks.copy()}])
                mem_comp_state["params"].extend(snapshot_mem_ports(m))
                self.state_before_changes["components"].append(mem_comp_state)

        # TODO remake buffer and move current usage etc params of buffer logic there and to read!

    def __str__(self):
        return f"MatmulStartEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name}, row_major={self.row_major})"

    def clone_for_retry(self):
        return MatmulStartEvent(
            operation=self.operation,
            matmul_array=self.matmul_array,
            scheduler=self._scheduler,
            is_rescheduled=True,
            row_major=self.row_major,
            verbose=self.verbose,
            log_state_before=self.log_state_before,
        )
    
    def execute(self, scheduler):
        self.current_tile = 0
        self._current_spatial_tile_event = None
        self._cycles = 0
        self.operation.start_time = self.event_time

        # Check if the operation has any children (operations that depend on the output of this operation),
        # if yes, postpone the start event until all children are done (the last child will reschedule this event)
        if self.operation.children:
            self.operation.pending_event = self.clone_for_retry()
            # This event returns early — no MatmulCompleteEvent will fire from it, so no
            # decrease_count will happen. So we remove any the increase_count retraction calls so that
            # retract_changes() on this original event so it doesn't incorrectly inflate the count.
            # The clone carries the correct increase_count for when it actually completes.
            if self.log_state_before and self.state_before_changes:
                for comp in self.state_before_changes["components"]:
                    if comp.get("obj") is self._scheduler.execution_graph.tensors_needed:
                        comp["method_calls"] = []
                        break
            return
        
        # TODO SPECIALLY
        if self.operation.computation.startswith('concat'):
            self.matmul_array.is_busy = False
            scheduler.schedule_event(self.event_time, ConcatenationStartEvent(self.operation, self.matmul_array, verbose=self.verbose), self.op_priority)
            return
        elif self.operation.computation.startswith('broadcast'):
            self.matmul_array.is_busy = False
            scheduler.schedule_event(self.event_time, BroadcastStartEvent(self.operation, self.matmul_array, verbose=self.verbose), self.op_priority)
            return

        dim_m, _, dim_n = get_matmul_dimensions(self.operation.input_data)

        self.num_row_tiles = math.ceil(dim_m / self.matmul_array.rows)
        self.num_col_tiles = math.ceil(dim_n / self.matmul_array.columns)

        if self.matmul_array.parent_component.auto_interconnect is True and self.matmul_array._auto_interconnect_set is False:
            self.matmul_array.find_and_assign_memories()

        if self.verbose:
            print(f"Splitting computation into {self.num_row_tiles * self.num_col_tiles} subtiles.")
        self.schedule_next_tile(scheduler)

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
            self._current_spatial_tile_event = compute_event
            scheduler.schedule_event(self.event_time, compute_event, self.op_priority)

            self.current_tile += 1
        else:
            scheduler.schedule_event(self.event_time, MatmulCompleteEvent(self.operation, self.matmul_array, self._cycles, scheduler, self, verbose=self.verbose), self.op_priority)


class MatmulCompleteEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, cycles, scheduler, start_event, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.matmul_array = matmul_array
        self.cycles = cycles
        self._scheduler = scheduler
        self.start_event = start_event

    def __str__(self):
        return f"MatmulCompleteEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name}, cycles={self.cycles})"

    def execute(self, scheduler):
        self.matmul_array.compute_done()
        self.matmul_array.global_cycles = self.event_time
        self.operation.is_done = True
        self.operation.finish_time = self.event_time
        self._current_spatial_tile_event = None
        self.operation.pending_event = None
        self.operation.remove_dependency()

        for p in self.operation.parents:
            # If this operation is the last child of the parent, and the parent's event is pending, finally reschedule it
            if not p.children and p.pending_event is not None:
                scheduler.schedule_event(self.event_time, p.pending_event, p.pending_event.op_priority)

        # Update the dictionary of required tensors
        for input_item in self.operation.input_data:
            tensor_name = input_item['name']
            scheduler.execution_graph.tensors_needed.decrease_count(tensor_name)

        if self.operation._is_root:  # If final operation, the output is successfully in DRAM
            scheduler.execution_graph.tensors_needed.decrease_count(self.operation.output_data[0]['name'])
        
        if self.verbose:
            print(f"Computed matmul operation '{self.operation.name}' on {self.matmul_array.name} in {self.cycles} cycles and in global time {self.event_time}.")
            print(f"Completed matmul of '{self.operation.name}' on {self.matmul_array.name}.")
        
        assert self.matmul_array.is_busy == False, f"Matmul array '{self.matmul_array.name}' is still busy, cannot schedule next operation! This shouldn't happen."
        while self.matmul_array.plan:
            next_operation = self.matmul_array.plan.pop(0)
            if next_operation.is_done:  # Skip operations that are already done (may occur during step-wise simulation)
                continue
            # self.matmul_array.is_busy = True
            # TODO add and test row/col majorness
            next_start_event = MatmulStartEvent(next_operation, self.matmul_array, scheduler, row_major=True, verbose=self.verbose, log_state_before=scheduler.log_state_before)
            scheduler.schedule_event(self.event_time, next_start_event, next_start_event.op_priority)
            break


class SpatialTileComputeEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, r_c_tile_idxs, tile_dims, common_dim, tile_idx, subtiles, parent_event, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.op_priority = operation.priority
        self.row_tile_idx, self.col_tile_idx = r_c_tile_idxs
        self.row_tile_dim, self.col_tile_dim = tile_dims
        self.common_dim = common_dim
        self.matmul_array = matmul_array
        self.tile_idx = tile_idx
        self.subtiles = subtiles
        self.parent_event = parent_event
        self._scheduler = self.parent_event._scheduler
        self._cycles = 0

    def __str__(self):
        return f"SpatialTileComputeEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name}, tile_idx={self.tile_idx}, subtiles={self.subtiles})"

    def execute(self, scheduler):
        self.temp_subtiles = int(math.ceil(self.common_dim / self.matmul_array.buffer_length))
        self._current_temporal_tile_event = None

        # Decide on order of temporal subtile computation based on data tensors reuse (in row/col buffers), either we start with temp tile 0 or the last
        a_item = self.operation.input_data[0]
        b_item = self.operation.input_data[1]
        a_name = a_item['name']
        b_name = b_item['name']
        A_data = a_item['data']
        B_data = b_item['data']

        # We first need to determine the tile shapes and offsets
        # of the inputs for this spatial subtile
        # Then determine if its the first/last one for reuse
        last_temp_tile_common_dim = min(self.common_dim-((self.temp_subtiles-1)*self.matmul_array.buffer_length), self.matmul_array.buffer_length)
        A_tile_shape = (self.row_tile_dim, last_temp_tile_common_dim)
        A_input_offset_x, _ = A_data['offset']
        A_tile_offset = (A_input_offset_x + self.row_tile_idx * self.matmul_array.rows, (self.temp_subtiles-1)*self.matmul_array.buffer_length)
        A_first_tile_shape = (self.row_tile_dim, min(self.common_dim, self.matmul_array.buffer_length))
        A_first_tile_offset = (A_input_offset_x + self.row_tile_idx * self.matmul_array.rows, 0)

        B_tile_shape = (self.col_tile_dim, last_temp_tile_common_dim) if (self.common_dim == A_data['dimensions'][1]) and (self.common_dim == B_data['dimensions'][1]) else (last_temp_tile_common_dim, self.col_tile_dim)  # Check if the matrix is transposed in memory or not..
        _, B_input_offset_y = B_data['offset']
        B_tile_offset = (B_input_offset_y + self.col_tile_idx * self.matmul_array.columns, (self.temp_subtiles-1)*self.matmul_array.buffer_length) if (self.common_dim == A_data['dimensions'][1]) and (self.common_dim == B_data['dimensions'][1]) else ((self.temp_subtiles-1)*self.matmul_array.buffer_length, B_input_offset_y + self.col_tile_idx * self.matmul_array.columns)
        B_first_tile_shape = (self.col_tile_dim, min(self.common_dim, self.matmul_array.buffer_length)) if (self.common_dim == A_data['dimensions'][1]) and (self.common_dim == B_data['dimensions'][1]) else (min(self.common_dim, self.matmul_array.buffer_length), self.col_tile_dim)  # Check if the matrix is transposed in memory or not..
        B_first_tile_offset = (B_input_offset_y + self.col_tile_idx * self.matmul_array.columns, 0) if (self.common_dim == A_data['dimensions'][1]) and (self.common_dim == B_data['dimensions'][1]) else (0, B_input_offset_y + self.col_tile_idx * self.matmul_array.columns)

        if a_name in self.matmul_array.row_buffer.contents and self.matmul_array.row_buffer._check_tile_presence(a_name, A_first_tile_shape, A_first_tile_offset)[0]:
            self._forward_subtile_order = True
        elif b_name in self.matmul_array.col_buffer.contents and self.matmul_array.col_buffer._check_tile_presence(b_name, B_first_tile_shape, B_first_tile_offset)[0]:
            self._forward_subtile_order = True
        elif a_name in self.matmul_array.row_buffer.contents and self.matmul_array.row_buffer._check_tile_presence(a_name, A_tile_shape, A_tile_offset)[0]:
            self._forward_subtile_order = False
        elif b_name in self.matmul_array.col_buffer.contents and self.matmul_array.col_buffer._check_tile_presence(b_name, B_tile_shape, B_tile_offset)[0]:
            self._forward_subtile_order = False
        else:
            self._forward_subtile_order = True

        self.current_temp_tile = 0 if self._forward_subtile_order == True else self.temp_subtiles-1

        if self.verbose:
            print(f"Splitting spatial subtile {self.tile_idx+1}/{self.subtiles} (tile_id: {self.tile_idx}) computation for '{self.operation.name}' into {self.temp_subtiles} temporal subtiles.")
        self.schedule_next_tile(scheduler)

    
    def schedule_next_tile(self, scheduler):
        if (self._forward_subtile_order and self.current_temp_tile < self.temp_subtiles) or (not self._forward_subtile_order and self.current_temp_tile >= 0):
            in_tiles_offset = self.current_temp_tile * self.matmul_array.buffer_length
            tile_common_dim = min(self.common_dim-(self.current_temp_tile*self.matmul_array.buffer_length), self.matmul_array.buffer_length)

            a_item = self.operation.input_data[0]
            b_item = self.operation.input_data[1]
            a_name = a_item['name']
            b_name = b_item['name']
        
            # Left matrix A
            A_data = a_item['data']
            A_tensor_shape = A_data['dimensions']
            A_is_broadcasted = A_data['broadcasted_view']
            A_data_category = A_data['data_category']
            A_tile_shape = (self.row_tile_dim, tile_common_dim)
            A_tile_memory = self.get_memory_for_var(a_item)
            A_input_offset_x, _ = A_data['offset']
            A_tile_offset = (A_input_offset_x + self.row_tile_idx * self.matmul_array.rows, in_tiles_offset)
            A_tile_key = f"{A_tile_shape[0]}x{A_tile_shape[1]}_{A_tile_offset[0]}x{A_tile_offset[1]}"
            in_tile_a_info = (a_name, A_tile_memory.name, A_tile_key)
            scheduler.execution_graph.tensors_needed.add_tile(*in_tile_a_info)

            # Right matrix B
            B_data = b_item['data']
            B_tensor_shape = B_data['dimensions']
            B_is_broadcasted = B_data['broadcasted_view']
            B_data_category = B_data['data_category']            
            is_transposed = (self.common_dim == A_tensor_shape[1]) and (self.common_dim == B_tensor_shape[1])
            if is_transposed:
                B_tile_shape = (self.col_tile_dim, tile_common_dim)
            else:
                B_tile_shape = (tile_common_dim, self.col_tile_dim)            
            B_tile_memory = self.get_memory_for_var(b_item)
            _, B_input_offset_y = B_data['offset']
            if is_transposed:
                B_tile_offset = (B_input_offset_y + self.col_tile_idx * self.matmul_array.columns, in_tiles_offset)
            else:
                B_tile_offset = (in_tiles_offset, B_input_offset_y + self.col_tile_idx * self.matmul_array.columns)
            B_tile_key = f"{B_tile_shape[0]}x{B_tile_shape[1]}_{B_tile_offset[0]}x{B_tile_offset[1]}"
            in_tile_b_info = (b_name, B_tile_memory.name, B_tile_key)
            scheduler.execution_graph.tensors_needed.add_tile(*in_tile_b_info)

            # Event scheduling (computation after both reads succeed)            
            # TODO.. FOR FETCH.. ONLY READ OR READ AND ALSO WRITE COSTS??
            is_tile_a_found, missing_a_data = self.matmul_array.row_buffer._check_tile_presence(a_name, A_tile_shape, A_tile_offset)
            is_tile_b_found, missing_b_data = self.matmul_array.col_buffer._check_tile_presence(b_name, B_tile_shape, B_tile_offset)

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
            self._current_temporal_tile_event = compute_event
            
            if not is_tile_a_found:
                missing_a_data, missing_a_data_offset = missing_a_data
                A_read_event  = MemoryFetchEvent(a_name, A_data_category, A_is_broadcasted, self.matmul_array.data_bitwidth, A_tensor_shape, missing_a_data, missing_a_data_offset, self.matmul_array.row_buffer.upper_level_memory, self.matmul_array.row_buffer, self.op_priority, self.operation, next_event=compute_event, verbose=self.verbose)
                scheduler.schedule_event(self.event_time, A_read_event, self.op_priority)

            if not is_tile_b_found:
                missing_b_data, missing_b_data_offset = missing_b_data
                B_read_event  = MemoryFetchEvent(b_name, B_data_category, B_is_broadcasted, self.matmul_array.data_bitwidth, B_tensor_shape, missing_b_data, missing_b_data_offset, self.matmul_array.col_buffer.upper_level_memory, self.matmul_array.col_buffer, self.op_priority, self.operation, next_event=compute_event, verbose=self.verbose)
                scheduler.schedule_event(self.event_time, B_read_event, self.op_priority)

            if is_tile_a_found and is_tile_b_found:
                scheduler.schedule_event(self.event_time, compute_event, self.op_priority)
            
            self.current_temp_tile = self.current_temp_tile + 1 if self._forward_subtile_order else self.current_temp_tile - 1
        else:
            dim_m, _, dim_n = get_matmul_dimensions(self.operation.input_data)
            out_item = self.operation.output_data[0]
            out_var = out_item['name']
            out_data = out_item['data']
            out_data_category = out_data['data_category']
            out_tensor_shape = (dim_m, dim_n)
            out_tile_shape = (self.row_tile_dim, self.col_tile_dim)
            out_offset = (self.matmul_array.rows * self.row_tile_idx, self.matmul_array.columns * self.col_tile_idx)

            write_event = MemoryWriteEvent(out_var, out_data_category, False, self.operation.data_bitwidth, out_tensor_shape, out_tile_shape, out_offset, self.matmul_array.dynamic_param_memory, self.op_priority, self.operation, self.parent_event, verbose=self.verbose)
            if self.verbose:
                print(f"Computed spatial tile {self.tile_idx+1}/{self.subtiles} for '{self.operation.name}' on {self.matmul_array.name} in {self._cycles} cycles and in global time {self.event_time}.")

            tile_complete_event = SpatialTileCompleteEvent(self.operation, self.matmul_array, self.tile_idx, self.subtiles, self._cycles, next_event=write_event, verbose=self.verbose)
            self.parent_event._cycles += self._cycles  # Accumulate total cycles required for the whole matmul computation
            scheduler.schedule_event(self.event_time, tile_complete_event, self.op_priority)

    def get_memory_for_var(self, input_item):
        """
        input_item: The dict from the input_data list, 
                e.g., {'name': 'gqa_key', 'data': {...}}
        """
        if input_item['data']['data_category'] == 'static':
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
        self._current_temporal_tile_event = None
        self.matmul_array.global_cycles = self.event_time
        if self.verbose:
            print(f"Completed computation of spatial tile {self.tile_idx+1}/{self.subtiles} (tile_id: {self.tile_idx}) for '{self.operation.name}' on {self.matmul_array.name}.")
        if self.next_event:
            scheduler.schedule_event(self.event_time, self.next_event, self.next_event.op_priority)


class TemporalTileComputeEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, in_tiles_info, computed_tile_dims, common_dim, spat_subtile_info, temp_subtile_info, pending_reads, parent_event, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.op_priority = operation.priority
        self.matmul_array = matmul_array
        self._in_tile_a, self._in_tile_b = in_tiles_info
        self.row_tile_dim, self.col_tile_dim = computed_tile_dims
        self.common_dim = common_dim
        self.spat_tile_idx, self.spat_subtiles = spat_subtile_info
        self.temp_tile_idx, self.temp_subtiles = temp_subtile_info
        self.parent_event = parent_event
        self._scheduler = parent_event._scheduler
        self.pending_reads = pending_reads  # WE WAIT FOR READS OF TWO MATRICES

    def __str__(self):
        return f"TemporalTileComputeEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name}, spat_subtile_idx={self.spat_tile_idx}, spat_subtiles={self.spat_subtiles}, temp_subtile_idx={self.temp_tile_idx}, temp_subtiles={self.temp_subtiles})"

    def execute(self, scheduler):
        a_data_metadata = self.matmul_array.row_buffer.contents[self._in_tile_a[0]]
        row_elem_reads = self.row_tile_dim * self.common_dim
        
        # TODO REFACTOR INTO READ/WRITE FOR BUFFER CLASS!
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

        is_first_temp_tile = self.temp_tile_idx == 0
        is_last_temp_tile  = self.temp_tile_idx == self.temp_subtiles - 1
        num_macs, cycles = self.matmul_array.compute(
            self.row_tile_dim, self.common_dim, self.col_tile_dim,
            include_input_skew=is_first_temp_tile,
            include_output_drain=is_last_temp_tile,
        )
        # Stats logging
        self.operation.macs += num_macs
        self.operation.compute_time += cycles
        self.operation._phase_log.append((self.event_time, self.event_time + cycles, "compute"))
        self.parent_event._cycles += cycles  # Accumulate total cycles required for the whole spatial tile computation
        if self.verbose:
            print(f"Computed temporal tile {self.temp_tile_idx+1}/{self.temp_subtiles} of spatial tile {self.spat_tile_idx+1}/{self.spat_subtiles} for '{self.operation.name}' on {self.matmul_array.name} in {cycles} cycles and in global time {self.event_time + cycles}.")

        tile_complete_event = TemporalTileCompleteEvent(self.operation, self.matmul_array, self._in_tile_a, self._in_tile_b, (self.spat_tile_idx, self.spat_subtiles), (self.temp_tile_idx, self.temp_subtiles), cycles, scheduler.execution_graph.tensors_needed, next_event=self.parent_event, verbose=self.verbose)
        scheduler.schedule_event(self.event_time + cycles, tile_complete_event, self.op_priority)


class TemporalTileCompleteEvent(AbstractEvent):
    def __init__(self, operation, matmul_array, in_tile_a, in_tile_b, spat_subtile_info, temp_subtile_info, cycles, tensors_needed, next_event = None, verbose: bool = False):
        super().__init__(verbose)
        self.operation = operation
        self.matmul_array = matmul_array
        self._in_tile_a = in_tile_a
        self._in_tile_b = in_tile_b
        self.spat_tile_idx, self.spat_subtiles = spat_subtile_info
        self.temp_tile_idx, self.temp_subtiles = temp_subtile_info
        self.cycles = cycles
        self._tensors_needed = tensors_needed
        self.next_event = next_event

    def __str__(self):
        return f"TemporalTileCompleteEvent(operation={self.operation.name}, matmul_array={self.matmul_array.name}, spat_subtile_idx={self.spat_tile_idx}, spat_subtiles={self.spat_subtiles}, temp_subtile_idx={self.temp_tile_idx}, temp_subtiles={self.temp_subtiles}, cycles={self.cycles})"

    def execute(self, scheduler):
        a, A_tile_memory, A_tile_key = self._in_tile_a
        b, B_tile_memory, B_tile_key = self._in_tile_b

        scheduler.execution_graph.tensors_needed.remove_tile(a, A_tile_memory, A_tile_key)
        scheduler.execution_graph.tensors_needed.remove_tile(b, B_tile_memory, B_tile_key)
        self.matmul_array.global_cycles = self.event_time

        if self.verbose:
            print(f"Completed computation of temporal tile {self.temp_tile_idx+1}/{self.temp_subtiles} (tile_id: {self.temp_tile_idx}) of spatial tile {self.spat_tile_idx+1}/{self.spat_subtiles} (tile_id: {self.spat_tile_idx}) for '{self.operation.name}' on {self.matmul_array.name}.")
        if self.next_event:
            assert isinstance(self.next_event, SpatialTileComputeEvent)
            # We need to modify the event time since the next event is typically a MatmulStartEvent/SpatialTileComputeEvent from which we wish to continue with next tile
            self.next_event.event_time = self.event_time
            self.next_event.schedule_next_tile(scheduler)


def get_matmul_dimensions(input_data):
    """
    input_data: list of dicts [{'name': '...', 'data': {...}}, ...]
    """
    if not input_data or len(input_data) != 2:
        raise ValueError("Invalid matmul inputs: expected 2 input tensors.")
    # Access by index to ensure Matrix A (x) and Matrix B (y) order
    dims_x = input_data[0]['data']['tile_shape']
    dims_y = input_data[1]['data']['tile_shape']

    if dims_x[1] == dims_y[0]:
        dim_m, dim_k, dim_n = dims_x[0], dims_x[1], dims_y[1]
    elif dims_x[1] == dims_y[1]:  # Transpose the second matrix
        dim_m, dim_k, dim_n = dims_x[0], dims_x[1], dims_y[0]
    else:
        raise ValueError(f"Incompatible dimensions for matmul: X:{dims_x}, Y:{dims_y}")

    return dim_m, dim_k, dim_n
