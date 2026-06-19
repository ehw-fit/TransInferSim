from abc import ABC
import math
import numpy as np
from collections import OrderedDict
import json
import numpy as np
from pathlib import Path
from datetime import datetime


class GenericMemory(ABC):
    """Generic abstract class for a generic memory block.
    
    Args:
        name (str): The name of the memory block.
        width (int): Width of each entry in the array, (in bits per entry).
        depth (int): Number of addressable entries (array depth, in entries).
        bus_bitwidth (int): Width of the memory bus per transfer, (in bits per beat).
        action_latency (float): Latency of a memory action, in (seconds per access, based on technology node).  # TODO should be from accelergy!
        cycle_time (float): Accelerator cycle time, (in seconds per cycle).
        ports (int): Number of independent r/w access ports supported by the memory (≥1). (Defaults to 1)
        word_size (int): Size of one memory word (in bits). (Defaults to 8)
        replacement_strategy (str): Policy for evicting/replacing memory contents. (Defaults to random)
        parent_component (Any): Optional reference to the parent HW component in the hierarchy. (Defaults to None)
        accelergy_class (str): Accelergy class name for energy/area modeling. (Defaults to "")    
    """
    
    def __init__(self, width: int, depth: int, bus_bitwidth: int, action_latency: float, cycle_time: float, ports: int = 1, word_size: int = 8, banks: int = 1, name: str = "generic_memory", replacement_strategy = "random", parent_component = None, accelergy_class = ""):
        assert bus_bitwidth >= word_size, f"Bus bitwidth must be greater than or equal to word size for {self.__class__.__name__}."
        assert word_size % 2 == 0, f"Word size must be a multiple of 2 for {self.__class__.__name__}."
        assert bus_bitwidth % 2 == 0, f"Bus bitwidth must be a multiple of 2 for {self.__class__.__name__}."
        assert ports >= 1, f"Number of ports must be greater than or equal to 1 for {self.__class__.__name__}."
        assert width > 0, f"Width must be a positive integer for {self.__class__.__name__}."
        assert depth > 0, f"Depth must be a positive integer for {self.__class__.__name__}."
        assert banks >= 1, f"Number of memory banks should be than or equal to 1 for {self.__class__.__name__}."
        assert action_latency > 0, f"Action latency must be a positive integer for {self.__class__.__name__}."
        assert cycle_time > 0, f"Cycle time must be a positive number for {self.__class__.__name__}."

        self.name = name
        self.width = width
        self.depth = depth
        self.size = width * depth
        self.banks = banks
        self.words_capacity = self.size // word_size
        self.cycle_time = cycle_time  # Accelerator cycle time
        self.parent_component = parent_component
        self._auto_interconnect_set = False  # Used for auto interconnection feature between other HW components to avoid repated interconnetct

        # Functional characteristics
        self.word_size = word_size
        self.row_words = math.ceil(width / word_size)  # Affects number of words accessed by one memory read/write operation
        self.ports = ports
        self.action_latency = action_latency
        self.mem_access_cycles = int(math.ceil(self.action_latency / self.cycle_time))
        self.bus_bitwidth = bus_bitwidth
        # Peak transfers per second on one port (TODO add ddr factor etc.) TODO REFACTOR..
        beats_per_second = 1 / self.cycle_time          # now same as core frequency
        self.bandwidth_per_port = self.bus_bitwidth * beats_per_second
        #self.bandwidth_per_port = self.bus_bitwidth / (self.mem_access_cycles * self.cycle_time)  # in bits per second TODO delete later
        self.total_bandwidth = self.bandwidth_per_port * self.ports

        # Accelergy attributes
        self.accelergy_class = accelergy_class
        self._global_cycles = 0
        self._area = 0
        self._static_energy = 0
        self._dynamic_energy = 0

        # Simulation attributes
        self._replacement_strategy = replacement_strategy
        self.current_usage = 0
        self.contents = {}
        self.fetch_locks = {}  # A lock dictionary to track ongoing fetches (to prevent multiple fetches of the same data at the same time)
        self.free_locks = {}   # A lock dictionary to track ongoing data frees/write-backs (to prevent free attempts of the exact same data segment multiple times)

        self.available_ports = list(range(self.ports))
        self.pending_actions = []
        self.last_reads = {}  # Info about last two reads from the current memory used for detecting possible loop in memory fetch
        
        # To keep track of each port utilization
        # Inner list specifies port's current cycle count (w.r.t. to the memory alone) and the port's idle cycles
        self._cycles_per_ports = [[0, 0] for _ in range(self.ports)]

        self.associated_matmuls = []
        self.upper_level_memory = None  # Assigned during simulation or manually during initialization
        self.lower_level_memories = []

        self._data_read_count = 0
        self._word_read_count = 0
        self._mem_read_count = 0
        self._data_write_count = 0
        self._mem_write_count = 0
        self._word_write_count = 0
        self._cache_miss_count = 0
        self._fragmented_bits = 0

    def __str__(self):
        return (f"<class={self.__class__.__name__} name={self.name}| "
                f"Size: {self.size}({self.depth}x{self.width}) "
                f"Words per Row: {self.row_words} "
                f"Wordsize: {self.word_size} bits "
                f"Bus bitwidth: {self.bus_bitwidth} "
                f"Action latency: {self.action_latency} s "
                f"Cycles per memory access: {self.mem_access_cycles} "
                f"Ports: {self.ports} "
                f"Bandwidth per port: {self.bandwidth_per_port / 1e9:.2f} Gbps "
                f"Total bandwidth: {self.total_bandwidth / 1e9:.2f} Gbps>")

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, value):
        self._area = value

    @property
    def energy(self):
        return self._static_energy + self._dynamic_energy

    @property
    def static_energy(self):
        return self._static_energy
    
    @static_energy.setter
    def static_energy(self, value):
        self._static_energy = value
    
    @property
    def dynamic_energy(self):
        return self._dynamic_energy
    
    @dynamic_energy.setter
    def dynamic_energy(self, value):
        self._dynamic_energy = value

    @property
    def data_read_count(self):
        return self._data_read_count

    @data_read_count.setter
    def data_read_count(self, value):
        self._data_read_count = value

    @property
    def mem_read_count(self):
        return self._mem_read_count

    @mem_read_count.setter
    def mem_read_count(self, value):
        self._mem_read_count = value
    
    @property
    def word_read_count(self):
        return self._word_read_count

    @word_read_count.setter
    def word_read_count(self, value):
        self._word_read_count = value

    @property
    def data_write_count(self):
        return self._data_write_count

    @data_write_count.setter
    def data_write_count(self, value):
        self._data_write_count = value

    @property
    def mem_write_count(self):
        return self._mem_write_count

    @mem_write_count.setter
    def mem_write_count(self, value):
        self._mem_write_count = value

    @property
    def word_write_count(self):
        return self._word_write_count

    @word_write_count.setter
    def word_write_count(self, value):
        self._word_write_count = value

    @property
    def cache_miss_count(self):
        return self._cache_miss_count

    @cache_miss_count.setter
    def cache_miss_count(self, value):
        self._cache_miss_count = value

    @property
    def fragmented_bits(self):
        return self._fragmented_bits

    @fragmented_bits.setter
    def fragmented_bits(self, value):
        self._fragmented_bits = value

    @property
    def global_cycles(self):
        return self._global_cycles

    @global_cycles.setter
    def global_cycles(self, value):
        self._global_cycles = value

    @property
    def cycles_per_ports(self):
        return self._cycles_per_ports

    @property
    def replacement_strategy(self):
        return self._replacement_strategy

    # Discrete simulation methods
    def add_associated_matmul(self, matmul):
        self.associated_matmuls.append(matmul)

    def set_upper_level_memory(self, memory):
        self.upper_level_memory = memory
        memory.lower_level_memories.append(self)

    def unlock_fetch_lock(self, data_id):
        """Unlocks the fetch lock for a specific data_id after the fetch is complete."""
        if data_id in self.fetch_locks:
            del self.fetch_locks[data_id]
    
    def unlock_free_lock(self, data_id):
        """Unlocks the free lock for a specific data_id after the free/write-back is complete."""
        if data_id in self.free_locks:
            del self.free_locks[data_id]

    def get_available_port(self):
        """
        Finds the port with the minimum global cycles, randomly chooses one if there are more and returns the port ID.
        """
        min_cycle_count = min(self.cycles_per_ports[port][0] for port in self.available_ports)
        cands = sorted(p for p in self.available_ports if self.cycles_per_ports[p][0] == min_cycle_count)
        port = cands[0]
        self.available_ports.remove(port)
        return port

    def free_port(self, port):
        if port not in self.available_ports:
            self.available_ports.append(port)
        self.available_ports.sort()

    def update_per_ports_cycles(self, port, cycles, update_idle: bool = False):
        """
        Updates the first element of the tuple (current cycle count) for the specified port.
        """
        self.cycles_per_ports[port][0] += cycles
        if update_idle:
            self.cycles_per_ports[port][1] += cycles

    def synchronize_per_ports_cycles(self, global_cycles: int=0):
        max_cycle_count = max(cycle_list[0] for cycle_list in self.cycles_per_ports) if global_cycles == 0 else global_cycles

        # Synchronizing each port's cycle count and adjusting idle cycles
        for cycle_list in self.cycles_per_ports:
            idle_cycles_diff = max_cycle_count - cycle_list[0]
            cycle_list[0] = max_cycle_count
            cycle_list[1] += idle_cycles_diff
    
    def query_lower_memories(self, read_mode, data_id, tile_shape=None, elems_to_read=None, offset=None):
        def get_lower_memory_path(mem):
            """Recursively get a dictionary with the lower memory objects as keys and their lower-level memories as values."""
            lower_dict = OrderedDict()
            for lower_mem in mem.lower_level_memories:
                lower_dict[lower_mem] = get_lower_memory_path(lower_mem)
            return lower_dict

        mem_paths = OrderedDict()        
        for lower_mem in self.lower_level_memories:
            mem_paths[lower_mem] = get_lower_memory_path(lower_mem)

        def traverse_and_check(data_id, tile_shape, elems_to_read, offset, mem, subpaths):
            """Recursively traverse memory paths to check for data presence."""
            if read_mode == 'read_tile':
                if tile_shape is None or offset is None:
                    raise ValueError(f"tile_shape and offset must be provided for 'read_tile' mode in data query operation for {self.name} memory.")
                check_presence = lambda mem: mem._check_tile_presence(data_id, tile_shape, offset)
            else:  # read continuous
                if elems_to_read is None or offset is None:
                    raise ValueError(f"elems_to_read and offset must be provided for 'read_elements' mode in data query operation for {self.name} memory.")
                check_presence = lambda mem: mem._check_data_presence(data_id, elems_to_read, offset)

            data_found, missing_data = check_presence(mem)
            
            # If all required data are found in the current memory    
            if data_found:
                return data_found, missing_data
            else:
                if read_mode == "read_tile":
                    tile_shape, offset = missing_data
                else:
                    elems_to_read, offset = missing_data

                for submem, subsubpaths in subpaths.items():
                    found, miss = traverse_and_check(data_id, tile_shape, elems_to_read, offset, submem, subsubpaths)
                    if found:
                        return True, miss
                    missing_data = miss 
                return False, missing_data

        # Traverse the memory paths to find where is the data ID with its specific tile/amount present
        for mem, subpaths in mem_paths.items():
            mem_found, missing_data = traverse_and_check(data_id, tile_shape, elems_to_read, offset, mem, subpaths)

            if mem_found:
                if read_mode == "read_tile":
                    return mem, (tile_shape, offset)
                else:
                    return mem, (elems_to_read, offset)
            else:
                if read_mode == "read_tile":
                    tile_shape, offset = missing_data
                else:
                    elems_to_read, offset = missing_data
        return None, (tile_shape, offset) if read_mode == "read_tile" else (elems_to_read, offset)

    def _initialize_presence_matrix(self, tensor_shape, values):
        """Initialize a presence matrix for the given tensor."""
        if values == 0:
            return np.zeros(tensor_shape, dtype=bool)
        elif values == 1:
            return np.ones(tensor_shape, dtype=bool)

    def _update_presence_matrix(self, mode, data_metadata, offset, tile_shape=None, elems_to_write=None, inplace=True):
        """
        Update the presence matrix either by marking a specific tile as present or by marking a number of elements as present in a row-major order.

        Args:
            mode (str): Update mode for the presence matrix. Either 'tile' or 'elements'.
            data_metadata (dict): Metadata for the data being updated.
            offset (tuple): The starting point (row, col) in the presence matrix.
            tile_shape (tuple, optional): Shape of the tile to mark as present. Used only in 'tile' mode.
            elems_to_write (int, optional): Number of elements to mark as present in row-major order. Used only in 'elements' mode.
            inplace (bool, optional): Whether to update the presence matrix in place. Default is True.

        Returns:
            int: Number of new elements marked as present.
        """
        presence_matrix = data_metadata['presence_matrix']
        start_row, start_col = offset

        # BROADCASTED VIEW (logical to physical mapping)
        if data_metadata.get('broadcasted_view', False):
            phys_rows, phys_cols = presence_matrix.shape
            logical_rows, logical_cols = data_metadata['tensor_shape']

            if mode == 'tile':
                if tile_shape is None:
                    raise ValueError("tile_shape must be provided in 'tile' mode.")

                tile_rows, tile_cols = tile_shape

                # Map logical tile rows/cols to physical indices (unique sets)
                row_idx = (start_row + np.arange(tile_rows)) % phys_rows
                col_idx = (start_col + np.arange(tile_cols)) % phys_cols
                row_idx = np.unique(row_idx)
                col_idx = np.unique(col_idx)

                view = presence_matrix[np.ix_(row_idx, col_idx)]
                num_new_elems = int((~view.astype(bool)).sum())

                if inplace:
                    presence_matrix[np.ix_(row_idx, col_idx)] = True

                return num_new_elems

            elif mode == 'elements':
                if elems_to_write is None:
                    raise ValueError("elems_to_write must be provided in 'elements' mode.")

                # Clamp logical stream within logical tensor bounds
                if start_row >= logical_rows:
                    return 0
                avail = (logical_rows - start_row) * logical_cols - start_col
                elems_to_write = min(elems_to_write, avail)
                if elems_to_write <= 0:
                    return 0

                # Helper: add a logical row segment to physical row intervals
                row_full = set()
                row_intervals = {}

                def _add_interval(pr, s, e):
                    if pr in row_full:
                        return
                    if pr not in row_intervals:
                        row_intervals[pr] = []
                    row_intervals[pr].append((s, e))

                def _add_segment(pr, logical_start_col, length):
                    if length <= 0 or pr in row_full:
                        return
                    eff = min(length, phys_cols)
                    if eff == phys_cols:
                        row_full.add(pr)
                        if pr in row_intervals:
                            row_intervals.pop(pr, None)
                        return
                    s = logical_start_col % phys_cols
                    e = s + eff
                    if e <= phys_cols:
                        _add_interval(pr, s, e)
                    else:
                        _add_interval(pr, s, phys_cols)
                        _add_interval(pr, 0, e - phys_cols)

                # Split row-major stream into: first partial row, full rows, last partial row
                remaining = elems_to_write
                first_row_take = min(remaining, logical_cols - start_col)
                remaining_after_first = remaining - first_row_take
                full_rows = remaining_after_first // logical_cols
                last_row_take = remaining_after_first % logical_cols

                # First row segment
                _add_segment(start_row % phys_rows, start_col, first_row_take)

                # Full middle rows (each starts at col 0, length logical_cols)
                if full_rows > 0:
                    if full_rows >= phys_rows:
                        phys_row_list = range(phys_rows)
                    else:
                        start_pr = (start_row + 1) % phys_rows
                        phys_row_list = [ (start_pr + i) % phys_rows for i in range(full_rows) ]

                    for pr in phys_row_list:
                        _add_segment(pr, 0, logical_cols)

                # Last row segment (if any)
                if last_row_take > 0:
                    last_logical_row = start_row + 1 + full_rows
                    _add_segment(last_logical_row % phys_rows, 0, last_row_take)

                # Count new elems without double-counting (merge intervals per row)
                num_new_elems = 0

                for pr in row_full:
                    row_view = presence_matrix[pr, :]
                    num_new_elems += int((~row_view.astype(bool)).sum())

                for pr, intervals in row_intervals.items():
                    intervals.sort()
                    merged = []
                    for s, e in intervals:
                        if not merged or s > merged[-1][1]:
                            merged.append([s, e])
                        else:
                            merged[-1][1] = max(merged[-1][1], e)

                    for s, e in merged:
                        seg = presence_matrix[pr, s:e]
                        num_new_elems += int((~seg.astype(bool)).sum())

                # Apply updates if inplace
                if inplace:
                    for pr in row_full:
                        presence_matrix[pr, :] = True
                    for pr, intervals in row_intervals.items():
                        intervals.sort()
                        merged = []
                        for s, e in intervals:
                            if not merged or s > merged[-1][1]:
                                merged.append([s, e])
                            else:
                                merged[-1][1] = max(merged[-1][1], e)
                        for s, e in merged:
                            presence_matrix[pr, s:e] = True

                return int(num_new_elems)

            else:
                raise ValueError(f"Unknown update mode: {mode} requested for presence matrix update in memory {self.name}. Supported modes are 'tile' and 'elements'.")
        
        # ORIGINAL LOGIC
        if mode == 'tile':
            if tile_shape is None:
                raise ValueError("tile_shape must be provided in 'tile' mode.")

            end_row = start_row + tile_shape[0]
            end_col = start_col + tile_shape[1]

            # Number of elements that are not already present and need to be fetched
            new_elems = ~presence_matrix[start_row:end_row, start_col:end_col].astype(bool)
            num_new_elems = int(new_elems.sum())
            # Update of the presence matrix to mark the to be fetched elements as present
            if inplace:
                presence_matrix[start_row:end_row, start_col:end_col] = True
            return num_new_elems
        elif mode == 'elements':
            if elems_to_write is None:
                raise ValueError("elems_to_write must be provided in 'elements' mode.")

            rows, cols = presence_matrix.shape
            # Update based on the number of elements to set in a row-major fashion
            num_new_elems = 0
            for row in range(start_row, rows):
                remaining_elems_in_row = cols - start_col

                if elems_to_write  <= remaining_elems_in_row:
                    # Set a specific part of the row
                    new_elems = ~presence_matrix[row, start_col:start_col + elems_to_write ].astype(bool)
                    num_new_elems += int(new_elems.sum())
                    if inplace:
                        presence_matrix[row, start_col:start_col + elems_to_write ] = True
                    break
                else:
                    # Set the entire remaining part of the row
                    new_elems = ~presence_matrix[row, start_col:].astype(bool)
                    num_new_elems += int(new_elems.sum())
                    if inplace:
                        presence_matrix[row, start_col:] = True
                    elems_to_write  -= remaining_elems_in_row
                start_col = 0  # Reset start_col for the next row
                
            return num_new_elems
        else:
            raise ValueError(f"Unknown update mode: {mode} requested for presence matrix update in memory {self.name}. Supported modes are 'tile' and 'elements'.")

    def _check_tile_presence(self, data_id, tile_shape, offset):
        """Check if the entire tile or parts of it are present in the memory.

        Returns:
            tuple:
                - (True, None) if the entire tile is present.
                - (False, (missing_tile_shape, offset)) if some parts of the tile are missing.
        """
        start_row, start_col = offset
        end_row = start_row + tile_shape[0]
        end_col = start_col + tile_shape[1]
        if data_id not in self.contents:
            return False, (tile_shape, offset)

        data_metadata = self.contents[data_id]
        presence_matrix = data_metadata['presence_matrix']

        # BROADCASTED VIEW HANDLING (logical to physical mapping for non repeated accesses)
        if data_metadata.get('broadcasted_view', False):
            phys_rows, phys_cols = presence_matrix.shape
            tile_rows, tile_cols = tile_shape

            # Scan the requested logical tile starting at *offset*, map to physical, find first miss
            for r in range(tile_rows):
                pr = (start_row + r) % phys_rows
                for c in range(tile_cols):
                    pc = (start_col + c) % phys_cols
                    if not presence_matrix[pr, pc]:
                        # Minimal + always correct: fetch the missing physical element once
                        return False, ((1, 1), (pr, pc))

            return True, None

        # non-broadcast logic
        end_row = min(end_row, presence_matrix.shape[0])  # Clamping to ensure we don't go out of bounds
        end_col = min(end_col, presence_matrix.shape[1])  # Clamping to ensure we don't go out of bounds

        # Check if the whole tile is marked as present in the presence_matrix
        if presence_matrix[start_row:end_row, start_col:end_col].all():
            return True, None
        else:
            total_missing = (presence_matrix[start_row:end_row, start_col:end_col] == False).sum()
            for row in range(start_row, end_row):
                row_sum = presence_matrix[row, start_col:end_col].sum()
                # We found first line containing missing values from the tile
                if row_sum != tile_shape[1]:
                    for col in range(start_col, end_col):
                        # The first missing element found
                        if not presence_matrix[row, col]:
                            # Check if exactly one rectangular subtile is missing
                            tile_missing_values = (presence_matrix[row:end_row, col:end_col] == False).sum()
                            if col == start_col and (tile_missing_values == total_missing and tile_missing_values == ((end_row-row)*(end_col-col))):
                                return False, ((end_row-row, end_col-col), (row, col))
                            # Else return the current row's missing elements vector
                            else:
                                contiguous_false_count = 0
                                for row_col in range(col, end_col):
                                    if presence_matrix[row, row_col]:  # The row always starts with missing elems, if we find an elem, we stop
                                        return False, ((1, contiguous_false_count), (row, col))
                                    else:
                                        contiguous_false_count += 1

                                # The whole remainder of the row contains missing values
                                return False, ((1, contiguous_false_count), (row, col))

    def _check_data_presence(self, data_id, elems_to_check, offset):
        """Checks if a specific portion of data (from offset for elems_to_check elements) is present in the memory.

        Returns: 
            tuple: 
                - (True, None) if all elements are present.
                - (False, (num_missing, offset)) if some elements are missing.
        """
        if data_id not in self.contents:
            return False, (elems_to_check, offset)
        data_metadata = self.contents[data_id]
        presence_matrix = data_metadata['presence_matrix']
        start_row, start_col = offset
        
        # BROADCASTED VIEW HANDLING (logical to physical mapping)
        if data_metadata.get('broadcasted_view', False):
            phys_rows, phys_cols = presence_matrix.shape
            logical_rows, logical_cols = data_metadata['tensor_shape']

            # Walk the logical stream row-major for elems_to_check, map to physical
            remaining = elems_to_check
            row = start_row
            col = start_col

            while remaining > 0:
                # Clamp logical traversal within logical tensor bounds
                if row >= logical_rows:
                    return False, (remaining, (row, col))

                pr = row % phys_rows
                pc = col % phys_cols

                if not presence_matrix[pr, pc]:
                    # Minimal + correct: report one missing physical element at the first miss
                    return False, (1, (pr, pc))

                remaining -= 1
                col += 1
                if col >= logical_cols:
                    col = 0
                    row += 1

            return True, None

        # non-broadcast logic
        rows, cols = presence_matrix.shape
        # Start checking from the specified row
        for row in range(start_row, rows):
            remaining_elems_in_row = cols - start_col

            if elems_to_check <= remaining_elems_in_row:  # Check just a specific part of the row
                if presence_matrix[row, start_col:start_col+elems_to_check].sum() == elems_to_check:
                    return True, None  # All elements found in memory
                else:
                    missing_count = elems_to_check
                    for col in range(start_col, start_col + elems_to_check):
                        if presence_matrix[row, col]:
                            missing_count -= 1
                        else:
                            return False, (missing_count, (row, col))

            else:  # Check the entire remaining part of the row
                if presence_matrix[row, start_col:].sum() == remaining_elems_in_row:
                    elems_to_check -= remaining_elems_in_row  # All elements in this row are present, subtract and continue
                else:
                    missing_count = elems_to_check
                    for col in range(start_col, cols):
                        if presence_matrix[row, col]:
                            missing_count -= 1
                        else:
                            return False, (missing_count, (row, col))
            start_col = 0  # Reset start_col to 0 for the next row (starting from start_row+1)
        return False, (elems_to_check, offset)

    
    def _json_safe(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "sum_true": int(obj.sum()) if obj.dtype == bool else None
            }
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(v) for v in obj]
        return obj
    
    def dump_contents(self, filename=None, dump_matrix=False):
        """
        Dumps self.contents to a JSON file for debugging.
        Presence matrix is summarized by default (shape + sum).
        """

        dump = {}

        for data_id, meta in self.contents.items():
            pm = meta.get("presence_matrix", None)

            entry = {
                "tensor_shape": tuple(meta.get("tensor_shape", ())),
                "tile_shape": tuple(meta.get("tile_shape", ())) if "tile_shape" in meta else None,
                "broadcasted_view": meta.get("broadcasted_view", False),
                "data_amount": meta.get("data_amount", None),
            }

            if pm is not None:
                entry["presence_matrix_shape"] = tuple(pm.shape)
                entry["presence_matrix_true_count"] = int(np.sum(pm))
            else:
                entry["presence_matrix_shape"] = None
                entry["presence_matrix_true_count"] = None

            if dump_matrix and pm is not None:
                # Convert bool matrix to 0/1 for JSON
                entry["presence_matrix"] = pm.astype(int).tolist()

            dump[data_id] = entry

        # if filename is None:
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     filename = f"memory_contents_dump_{self.name}_{timestamp}.json"

        # Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # with open(filename, "w") as f:
        #     safe_dump = self._json_safe(dump)
        #     json.dump(safe_dump, f, indent=2)

        # print(f"[DEBUG] Memory contents dumped to {filename}")




    def _determine_victim_shape_and_offset(self, victim_data_id, max_num_words, tensors_needed_info):
        """
        Determine the number of elements to be freed from the victim presence matrix 
        along with its data shape (cointiguous block or tile) and starting offset.

        Args:
            victim_data_id (str): Name of the victim data.
            max_num_words (int): Number of words required to be freed in total.
            tensors_needed_info (TensorsNeededTracker): Contains info about which data tensors (and their respective per-memory tiles) are still needed to be stored (written back) for future computation. Otherwise the data are just evicted.

        Returns:
            tuple: (layout_type, layout_info (tuple or int), offset (tuple))
                - layout_info: 'contiguous' if elements are in a contiguous block, 'tile' if in a tile.
                - data_layout: Shape of the tile as (rows, cols) or number of contiguous elements.
                - offset: Starting point (row, col) for freeing.
        """
        
        data_metadata = self.contents[victim_data_id]
        elements_per_word = self.word_size // data_metadata['data_bitwidth']
        max_elems_to_erase = max_num_words * elements_per_word
        presence_matrix = data_metadata['presence_matrix']
        is_broadcast = data_metadata.get('broadcasted_view', False)
        phys_rows, phys_cols = presence_matrix.shape

        # If the victim data has some tiles needed for a pending operation, we must not free them
        if tensors_needed_info.is_in_memory(victim_data_id, self.name):
            tiles = tensors_needed_info.get()[victim_data_id]["tiles"][self.name]
            mask = np.zeros_like(presence_matrix, dtype=bool)
            parse_tile_key = lambda k: tuple(map(lambda s: tuple(map(int, s.split('x'))), k.split('_')))

            for tile_key in tiles.keys():
                tile_shape, offset = parse_tile_key(tile_key)
                tile_h, tile_w = tile_shape
                start_row, start_col = offset
                
                if is_broadcast:
                    for r in range(tile_h):
                        pr = (start_row + r) % phys_rows
                        for c in range(tile_w):
                            pc = (start_col + c) % phys_cols
                            mask[pr, pc] = True
                else:
                    end_row = start_row + tile_h
                    end_col = start_col + tile_w
                    mask[start_row:end_row, start_col:end_col] = True
    
            subtiles_presence = np.logical_and(presence_matrix, mask)
            presence_matrix = np.logical_xor(presence_matrix, subtiles_presence)

        _, cols = presence_matrix.shape
        last_true_indices = np.argwhere(presence_matrix)

        assert last_true_indices.size > 0, f"Internal error! No True values found in the victim presence matrix of data '{victim_data_id}', this should not happen."
        
        last_row_with_data = last_true_indices[-1][0]
        last_col_with_data = last_true_indices[-1][1]

        # Try CONTIGUOUS: Traverse the matrix from the bottom up to find contiguous elements
        contiguous_elems_found = 0

        for row in range(last_row_with_data, -1, -1):
            row_sum = presence_matrix[row, :].sum()  # Sum of True values in the row (total present elements)

            # Situation when we have encountered the last row filled with data
            if contiguous_elems_found == 0 and row_sum > 0:
                if row_sum != last_col_with_data+1:  # If not all elements contiguous
                    break
                contiguous_elems_found += row_sum

                if row_sum >= max_elems_to_erase or row == 0:  # If the last row's data contains the required amount of words.. we return it for memory free
                    return ("contiguous", min(row_sum, max_elems_to_erase), (row, row_sum-min(row_sum, max_elems_to_erase)))
            else:
                if row_sum != cols:  # If any row above does not have whole row filled, the presence matrix is subtiled and not contiguous
                    break

                # When we have found the required amount or when we are at the end of the tensor occupancy
                if (row_sum >= (max_elems_to_erase-contiguous_elems_found)) or (row == 0):
                    return ("contiguous", contiguous_elems_found + min(row_sum, max_elems_to_erase-contiguous_elems_found), (row, row_sum-min(row_sum, max_elems_to_erase-contiguous_elems_found)))
                contiguous_elems_found += row_sum

        # Try TILE: Deduce the tile shape
        # Calculate the number of contiguous True elements from the last row containing data
        contiguous_elems_found = 0
        col_index = last_col_with_data
        while col_index >= 0 and presence_matrix[last_row_with_data, col_index]:
            contiguous_elems_found += 1
            col_index -= 1

        # Iterate through possible tile shapes (from `1` to `last_row_with_data+1` number of rows x present elems per last row)
        start_col = last_col_with_data - contiguous_elems_found + 1
        for num_rows, row_offset in enumerate(reversed(range(0, last_row_with_data+1)), start=1):
            current_tile_size = num_rows * contiguous_elems_found

            # We arrived at the first row, proceed to return the shape and offset
            if row_offset == 0:
                # The tile itself is a vector at the beginning of tensor (first row)
                if last_row_with_data == 0:
                    # The current tile is bigger than the required amount for freeing, we need to compute the shape and offset
                    if current_tile_size > max_elems_to_erase:
                        return ("tile", (1, max_elems_to_erase), (0, start_col+(current_tile_size-max_elems_to_erase)))
                    else:
                        return ("tile", (1, contiguous_elems_found), (0, start_col))

                # We iterated to this point with a larger found tile shape
                else:
                    if presence_matrix[row_offset:last_row_with_data+1, start_col:last_col_with_data+1].all():
                        # The current tile is bigger than the required amount for freeing, we need to compute the shape and offset
                        if current_tile_size > max_elems_to_erase:
                            # num_rows-1 because previous shape was the last valid
                            # row_offset+1 since we are iterating in a decreasing manner (finding the tile backwards from the view of memory occupation)
                            # and the +1 offset was the last valid
                            return ("tile", (num_rows-1, contiguous_elems_found), (row_offset+1, start_col))        
                        else:
                            return ("tile", (num_rows, contiguous_elems_found), (row_offset, start_col))

            # We found some missing values OR the amount of data is too much for the required amount, return the shape of previous valid shape and its offset
            if not presence_matrix[row_offset:last_row_with_data+1, start_col:last_col_with_data+1].all() or current_tile_size > max_elems_to_erase:
                if num_rows == 1:  # We are at the last row.. calculate the number of elements viable for removal in the vector
                    return ("tile", (1, max_elems_to_erase), (row_offset, start_col+(current_tile_size-max_elems_to_erase)))
                else:
                    # num_rows-1 because previous shape was the last valid
                    # row_offset+1 since we are iterating in a decreasing manner (finding the tile backwards from the view of memory occupation)
                    # and the +1 offset was the last valid
                    return ("tile", (num_rows-1, contiguous_elems_found), (row_offset+1, start_col))

    def _clear_presence_matrix(self, data_metadata):
        data_metadata['presence_matrix'][:,:] = False
    
    def _free_presence_matrix_data(self, data_metadata, elements_to_free, offset):
        """
        Mark a specific number of elements as absent (False) in the presence matrix starting from the given offset.

        Args:
            data_metadata (dict): Metadata for the data being updated.
            elements_to_free (int): Number of elements to mark as absent.
            offset (tuple): The starting point (row, col) in the presence matrix.
        """
        presence_matrix = data_metadata['presence_matrix']
        start_row, start_col = offset
        rows, cols = presence_matrix.shape

        for row in range(start_row, rows):
            remaining_elems_in_row = cols - start_col

            if elements_to_free <= remaining_elems_in_row:  # If the required elements fit within this row
                presence_matrix[row, start_col:start_col + elements_to_free] = False
                elements_to_free = 0  # All elements have been freed
                break
            else:  # If the required elements span across multiple rows
                presence_matrix[row, start_col:] = False
                elements_to_free -= remaining_elems_in_row  # Subtract the number of elements freed in this row
            start_col = 0  # Reset start_col to 0 for the next row

        if elements_to_free > 0:
            raise ValueError("Internal error occured. Not enough elements in the presence matrix to free the required number of elements.")

    def _free_presence_matrix_tile(self, data_metadata, tile_shape, offset):
        """
        Mark a specific tile of elements as absent (False) in the presence matrix starting from the given offset.

        Args:
            data_metadata (dict): Metadata for the data being updated.
            tile_shape (tuple): Shape of data tile to remove
            offset (tuple): The starting point (row, col) in the presence matrix.
        """
        presence_matrix = data_metadata['presence_matrix']
        start_row, start_col = offset
        end_row = start_row + tile_shape[0]
        end_col = start_col + tile_shape[1]

        assert presence_matrix[start_row:end_row, start_col:end_col].sum() == (tile_shape[0] * tile_shape[1]), "The sum of True values in the presence matrix tile does not match the tile size"
        presence_matrix[start_row:end_row, start_col:end_col] = False

    def read(self, read_mode, data_id, data_category, is_data_broadcasted, tensor_shape, data_bitwidth, port_id, tile_shape=None, elems_to_read=None, offset=None):
        """
        Read data from memory, either as a specific tile or a number of elements.

        Args:
            read_mode (str): Read mode for the memory block. Either 'read_elements' and 'read_tile' are supported.
            data_id (str): Identifier for the data.
            data_category (str): Identifier for 'static' or 'dynamic' data tensor.
            is_data_broadcasted (bool): Whether the data tensor is a broadcasted view.
            tensor_shape (tuple): Shape of the whole tensor to be read from memory, defined as (rows, cols).
            data_bitwidth (int): Bitwidth of the data elements to be read from memory.
            port_id (int): ID of port used for retrieval of the data.
            tile_shape (tuple): Shape of the tile to read, defined as (tile_rows, tile_cols). Used only in 'read_tile' mode.
            elems_to_read (int): Number of elements to read from memory. Used only in 'read_elements' mode.
            offset (tuple): Offset within the data, as (row_offset, col_offset).

        Returns:
            TODO  update tuple of three now...
            tuple:
                - success (bool): True if the read operation was successful, False if a fetch operation from upper memory is needed.
                - action_cycles (int) or fetch_info (tuple):
                If the read is successful, returns the number of cycles the read operation took. 
                If a fetch from upper memory is required, returns a tuple containing:
                    - tile_shape (tuple): The shape of the tile to be fetched from upper memory
                    - offset (tuple): The offset within the data where the fetch should start.
        """
        if data_id not in self.contents:
            self.contents[data_id] = {
                'data_category': data_category,
                'broadcasted_view': is_data_broadcasted,
                'tensor_shape': tensor_shape,
                'presence_matrix': self._initialize_presence_matrix(tensor_shape, 0),
                'data_amount': 0,
                'data_read_count': 0,
                'mem_read_count': 0,
                'data_write_count': 0,
                'mem_write_count': 0,
                'word_amount': 0,
                'word_read_count': 0,
                'word_write_count': 0,
                'data_bit_packing': self.word_size != data_bitwidth,
                'data_bitwidth': data_bitwidth,
                'fragmented_bits': 0,
                'insertion_time': 0,
                'last_access_time': 0,
                'cache_miss_count': tile_shape[0]*tile_shape[1] if read_mode == 'read_tile' else elems_to_read
            }
            # Lock the specific tile being fetched to prevent duplicate DRAM fetches for the
            # same tile. Key is (data_id, offset) so different tiles of the same tensor don't block each other.
            self.fetch_locks[(data_id, offset)] = []
            self.cache_miss_count += tile_shape[0]*tile_shape[1] if read_mode == 'read_tile' else elems_to_read

            data_layout = tile_shape if read_mode == 'read_tile' else elems_to_read
            return False, (data_layout, offset)

        # TODO update return in all cases, update function description, similarly update offchip memory
        # also check concatenation??!!

        data_metadata = self.contents[data_id]

        if read_mode == 'read_tile':
            if tile_shape is None or offset is None:
                raise ValueError(f"tile_shape and offset must be provided for 'read_tile' mode in read operation from {self.name} memory.")

            data_found, missing_data = self._check_tile_presence(data_id, tile_shape, offset)
            if data_found:
                tile_rows, tile_cols = tile_shape
                if data_metadata.get('broadcasted_view', False):
                    phys_rows, phys_cols = data_metadata['presence_matrix'].shape
                    tile_rows = min(tile_rows, phys_rows)
                    tile_cols = min(tile_cols, phys_cols)
                elements_per_word = self.word_size // data_bitwidth
                mem_words = math.ceil(tile_rows * tile_cols / elements_per_word)
                bits_requested = mem_words * self.word_size
                bits_per_array_access = self.banks * self.width
                array_accesses = int(math.ceil(bits_requested / bits_per_array_access))
                # TODO array_accesses = 1 as old code..
                bus_transfers  = int(math.ceil(bits_requested / self.bus_bitwidth))

                self.data_read_count += tile_rows * tile_cols
                self.word_read_count += mem_words
                self.mem_read_count += bus_transfers

                data_metadata['data_read_count'] += tile_rows * tile_cols
                data_metadata['word_read_count'] += mem_words
                data_metadata['mem_read_count'] += bus_transfers

                # TODO analyze srams, if there is overlap, then max should be here instead of addition?
                # cycles = array_accesses * self.mem_access_cycles + bus_transfers  # one-time access cost + burst bus transfers
                cycles = self.mem_access_cycles + bus_transfers  # one-time access cost + burst bus transfers
                data_metadata['last_access_time'] = self.cycles_per_ports[port_id][0] + cycles
                return True, cycles
            else:
                # If data not in memory, the data must be fetched
                missing_tile, missing_tile_offset = missing_data
                data_metadata['cache_miss_count'] += missing_tile[0]*missing_tile[1]
                self.cache_miss_count += missing_tile[0]*missing_tile[1]
                self.fetch_locks[(data_id, missing_tile_offset)] = []
                return False, (missing_tile, missing_tile_offset)

        elif read_mode == 'read_elements':
            if elems_to_read is None or offset is None:
                raise ValueError(f"elems_to_read and offset must be provided for 'read_elements' mode in read operation from {self.name} memory.")

            data_found, missing_data = self._check_data_presence(data_id, elems_to_read, offset)
            if data_found:
                if data_metadata.get('broadcasted_view', False):
                    phys_rows, phys_cols = data_metadata['presence_matrix'].shape
                    logical_rows, logical_cols = data_metadata['tensor_shape']

                    remaining = elems_to_read
                    row, col = offset

                    # Clamp to logical bounds
                    if row >= logical_rows:
                        elems_to_read = 0
                    else:
                        # Total logical elements available from offset to end of logical tensor
                        avail = (logical_rows - row) * logical_cols - col
                        remaining = min(remaining, avail)

                        if remaining <= 0:
                            elems_to_read = 0
                        else:
                            # Determine how many logical rows are touched by this stream
                            first_row_take = min(remaining, logical_cols - col)
                            remaining_after_first = remaining - first_row_take

                            full_rows = remaining_after_first // logical_cols
                            last_row_take = remaining_after_first % logical_cols

                            logical_rows_touched = 1 + full_rows + (1 if last_row_take > 0 else 0)

                            # Distinct physical rows touched
                            phys_rows_touched = min(phys_rows, logical_rows_touched)
                            phys_cols_touched = min(phys_cols, logical_cols)

                            elems_to_read = phys_rows_touched * phys_cols_touched

                elements_per_word = self.word_size // data_bitwidth
                mem_words = math.ceil(elems_to_read / elements_per_word)
                bits_requested = mem_words * self.word_size
                bits_per_array_access = self.banks * self.width
                array_accesses = int(math.ceil(bits_requested / bits_per_array_access))
                # TODO array_accesses = 1 as old code..
                bus_transfers  = int(math.ceil(bits_requested / self.bus_bitwidth))

                self.data_read_count += elems_to_read
                self.word_read_count += mem_words
                self.mem_read_count += bus_transfers

                data_metadata['data_read_count'] += elems_to_read
                data_metadata['word_read_count'] += mem_words
                data_metadata['mem_read_count'] += bus_transfers

                # TODO analyze srams, if there is overlap, then max should be here instead of addition?
                # cycles = array_accesses * self.mem_access_cycles + bus_transfers  # one-time access cost + burst bus transfers
                cycles = self.mem_access_cycles + bus_transfers  # one-time access cost + burst bus transfers
                data_metadata['last_access_time'] = self.cycles_per_ports[port_id][0] + cycles
                return True, cycles
            else:
                # If data not in memory, the data must be fetched
                missing_elems, missing_tile_offset = missing_data
                data_metadata['cache_miss_count'] += missing_elems
                self.cache_miss_count += missing_elems
                self.fetch_locks[(data_id, missing_tile_offset)] = []
                return False, (missing_elems , missing_tile_offset)
        else:
            raise ValueError(f"Unknown read mode: {read_mode} requested from memory {self.name}. Supported modes are 'read_elements' and 'read_tile'.")

    def write(self, write_mode, data_id, data_category, is_data_broadcasted, tensor_shape, data_bitwidth, port_id, tensors_needed_info, tile_shape=None, elems_to_write=None, offset=None, verbose=False):
        """
        Write data to memory, either as a specific tile or a number of elements.
        
        Args:
            write_mode (str): Write mode for the memory block. Either 'write_elements' and 'write_tile' are supported.
            data_id (str): Identifier for the data.
            data_category (str): Identifier for 'static' or 'dynamic' data tensor.
            is_data_broadcasted (bool): Whether the data tensor is a broadcasted view.
            tensor_shape (tuple): Shape of the whole tensor to be written to memory, defined as (rows, cols).
            data_bitwidth (int): Bitwidth of the data elements to be written to memory.
            port_id (int): ID of port used for storing the data.
            tensors_needed_info (TensorsNeededTracker): Contains info about which data tensors (and their respective per-memory tiles) are still needed to be stored (written back) for future computation. Otherwise the data are just evicted.
            tile_shape (tuple): Shape of the tile to write, defined as (tile_rows, tile_cols). Used only in 'write_tile' mode.
            elems_to_write (int): Number of elements to write to memory. Used only in 'write_elements' mode.
            offset (tuple): Offset within the data, as (row_offset, col_offset).
            verbose (bool): If True, prints detailed information about the write process.

        Returns:
            TODO  update tuple of three now...
            tuple:
                - success (bool): True if the write operation was successful, False if a write-back operation is needed.
                - action_cycles (int) or writeback_info (tuple):
                If the write is successful, returns the number of cycles the write operation took. 
                If a write-back is required, returns a tuple containing:
                    - victim_data_id (str): The data ID that needs to be written back to upper memory.
                    - victim_data_elems (int): The number of elements to be written back.
                    - victim_offset (tuple): The offset within the data where the write-back should start.          
        """
        if data_id not in self.contents:
            self.contents[data_id] = {
                'data_category': data_category,
                'broadcasted_view': is_data_broadcasted,
                'tensor_shape': tensor_shape,
                'presence_matrix': self._initialize_presence_matrix(tensor_shape, 0),
                'data_amount': 0,
                'data_read_count': 0,
                'mem_read_count': 0,
                'data_write_count': 0,
                'mem_write_count': 0,
                'word_amount': 0,
                'word_read_count': 0,
                'word_write_count': 0,
                'data_bit_packing': self.word_size != data_bitwidth,
                'data_bitwidth': data_bitwidth,
                'fragmented_bits': 0,
                'insertion_time': 0,
                'last_access_time': 0,
                'cache_miss_count': 0
            }
        data_metadata = self.contents[data_id]
        broadcasted_view = data_metadata.get('broadcasted_view', False)
        if broadcasted_view:
            phys_rows, phys_cols = data_metadata['presence_matrix'].shape
        new_elements = 0

        # NOTE: We do not allow for bit packing to split data across words (i.e. for 8bit words and 3bit data, 2 bits will be unused for the word)
        elements_per_word = self.word_size // data_bitwidth
        if write_mode == "write_tile":
            if tile_shape is None or offset is None:
                raise ValueError(f"tile_shape and offset must be provided for 'write_tile' mode in write operation to {self.name} memory.")

            # Check if the data can fit in memory at all
            tile_rows, tile_cols = tile_shape
            if broadcasted_view:
                total_words_required = math.ceil(min(tile_rows, phys_rows) * min(tile_cols, phys_cols) / elements_per_word)
            else:
                total_words_required = math.ceil(tile_rows * tile_cols / elements_per_word)
            assert total_words_required <= self.words_capacity, f"Data {data_id} too large ({total_words_required} words) to fit in memory of size {self.words_capacity} words ({self.size} b)."

            # Calculate the required amount of elements needed to be written into memory
            new_elements = self._update_presence_matrix(mode="tile", data_metadata=data_metadata, offset=offset, tile_shape=tile_shape, inplace=False)
        elif write_mode == "write_elements":
            if elems_to_write is None or offset is None:
                raise ValueError(f"elems_to_write and offset must be provided for 'write_elements' mode in write_elements operation to {self.name} memory.")

            # Check if the data can fit in memory at all
            if broadcasted_view:
                total_words_required = math.ceil(min(elems_to_write, phys_rows * phys_cols) / elements_per_word)
            else:
                total_words_required = math.ceil(elems_to_write / elements_per_word)
            assert total_words_required <= self.words_capacity, f"Data {data_id} too large ({total_words_required} words) to fit in memory of size {self.words_capacity} words ({self.size} b)."
            
            # Calculate the required amount of elements needed to be written into memory
            new_elements = self._update_presence_matrix(mode="elements", data_metadata=data_metadata, offset=offset, elems_to_write=elems_to_write, inplace=False)
        else:
            raise ValueError(f"Unknown write mode: {write_mode} requested from memory {self.name}. Supported modes are 'write_elements' and 'write_tile'.")

        # Determine the required space (number of words) to be written into memory
        write_mem_words = math.ceil(new_elements / elements_per_word)

        # Memory full, we need to free some space before writing the data
        if self.current_usage + write_mem_words > self.words_capacity:
            assert self.replacement_strategy is not None, f"Memory '{self.name}' is full and has no replacement strategy defined! Define it during instantiation and re-run."
            words_to_free = (self.current_usage + write_mem_words) - self.words_capacity
            if verbose:
                print(f"Not enough space for write of data '{data_id}' into memory '{self.name}'. Need space for {words_to_free} words. Finding a victim data to free space.")

            while words_to_free != 0:
                replacement_result = self._apply_replacement_strategy(words_to_free, tensors_needed_info, verbose)
                # No valid data found for replacement now, we need to reschedule this operation
                # (some other event may be writing back the only possible data id and no other event can at the same time)
                if replacement_result is None:
                    return False, None

                replacement_outcome, victim_data_id, victim_data_category, victim_is_broadcasted, victim_tensor_shape, victim_layout_info, victim_layout_type, victim_offset, victim_data_bitwidth, victim_words = replacement_result

                if replacement_outcome == "write_back":
                    self.free_locks[victim_data_id] = []
                    # Return to Discrete Simulation events and proceed with write-back (this write operation will be rescheduled)
                    return False, (victim_data_id, victim_data_category, victim_is_broadcasted, victim_tensor_shape, victim_layout_info, victim_layout_type, victim_offset, victim_data_bitwidth)
                else:  # Proceed with freeing the data.. no need for write_back (either the data are not important anymore or are already present in the upper memory)
                    self.free(victim_data_id, victim_layout_info, victim_layout_type, victim_words, victim_offset, tensors_needed_info)
                    # self.unlock_free_lock(victim_data_id)
                    words_to_free -= victim_words

        # Finally proceed with the write operation
        if write_mode == "write_tile":
            self._update_presence_matrix(mode="tile", data_metadata=data_metadata, offset=offset, tile_shape=tile_shape, inplace=True)
        else: 
            self._update_presence_matrix(mode="elements", data_metadata=data_metadata, offset=offset, elems_to_write=elems_to_write, inplace=True)
        
        data_bits = new_elements * data_bitwidth
        bits_to_write = write_mem_words * self.word_size
        fragmented_bits = bits_to_write - data_bits
        bus_transfers = int(math.ceil(bits_to_write / self.bus_bitwidth))
        bits_per_array_access = self.banks * self.width
        # TODO array_accesses = 1 as old code..
        array_accesses = int(math.ceil(bits_to_write / bits_per_array_access))

        data_metadata['data_amount'] += new_elements
        data_metadata['data_write_count'] += new_elements
        data_metadata['word_amount'] += write_mem_words
        data_metadata['word_write_count'] += write_mem_words
        data_metadata['mem_write_count'] += bus_transfers
        data_metadata['fragmented_bits'] += fragmented_bits

        self.current_usage += write_mem_words
        self.data_write_count += new_elements
        self.word_write_count += write_mem_words
        self.mem_write_count += bus_transfers
        self.fragmented_bits += fragmented_bits

        # TODO analyze srams, if there is overlap, then max should be here instead of addition?
        # cycles = array_accesses * self.mem_access_cycles + bus_transfers  # one-time access cost + burst bus transfers
        cycles = self.mem_access_cycles + bus_transfers  # one-time access cost + burst bus transfers
        data_metadata['insertion_time'] = self.cycles_per_ports[port_id][0] if data_metadata['insertion_time'] == 0 else data_metadata['insertion_time']
        data_metadata['last_access_time'] = self.cycles_per_ports[port_id][0] + cycles
        return True, cycles

    def detect_memory_thrashing(self, data_id: str, tensor_shape: tuple, tile_shape: tuple, offset: tuple) -> bool:
        """
        Track the last two read operations from memory and detect if memory thrashing is occurring.

        This method keeps track of the last two reads based on `data_id`, `tensor_shape`, `tile_shape`, and `offset`.
        If the same two read operations occur 20 times or more, it detects a potential memory thrashing situation.
        
        It is used during fetch operation.

        Args:
            data_id (str): The ID of the data tensor being fetched.
            tensor_shape (tuple): The shape of the tensor.
            tile_shape (tuple): The shape of the tile being fetched.
            offset (tuple): The offset within the data tensor.

        Returns:
            bool: True if memory thrashing is detected, False otherwise.
        """
        # If the data_id is already tracked, update the count and check for matching conditions
        if data_id in self.last_reads:
            # If the description matches previous read, increment the count for this operation
            if (self.last_reads[data_id]['tile_shape'] == tile_shape and self.last_reads[data_id]['offset'] == offset):
                self.last_reads[data_id]['count'] += 1
            else:
                # If the details do not match, replace the entry with the new details and reset the count
                self.last_reads[data_id] = {
                    'tensor_shape': tensor_shape,
                    'tile_shape': tile_shape,
                    'offset': offset,
                    #'max_tiles': math.ceil((tensor_shape[0] * tensor_shape[1]) / (tile_shape[0] * tile_shape[1])),
                    'count': 1
                }
        # If the data_id is not already tracked, manage the dictionary to keep only the last two reads
        else:
            if len(self.last_reads) >= 2:
                # Remove the oldest read entry to keep the list size to two
                oldest_read_id = list(self.last_reads.keys())[0]
                self.last_reads.pop(oldest_read_id)

            # Add the new read operation details
            self.last_reads[data_id] = {
                'tensor_shape': tensor_shape,
                'tile_shape': tile_shape,
                'offset': offset,
                #'max_tiles': math.ceil((tensor_shape[0] * tensor_shape[1]) / (tile_shape[0] * tile_shape[1])),
                'count': 1
            }

        # Check for potential memory thrashing
        if len(self.last_reads) == 2:
            read_counts = list(self.last_reads.values())
            # Check if both reads have occurred many times to CONFIDENTLY CAPTURE MEMORY THRASHING!
            # TODO HOW TO CONFIDENTLY DETERMINE THE AMOUNT FOR DETECTION? (because its not deducible only by tensor dims and mem size, but also by all the other mem events fighting over and potentially throwing one another out)
            if all(read['count'] > 5000 for read in read_counts):
                for i, read in enumerate(read_counts):
                    print(f"Memory thrashing detected: One of the last two memory fetches (for data_id: '{list(self.last_reads.keys())[i]}') have occurred {read['count']} times back to back.")
                return True
        return False

    def _apply_replacement_strategy(self, free_mem_words, tensors_needed_info, verbose):
        """
        Applies the configured replacement strategy to free up the necessary space in memory.

        Args:
            free_mem_words (int): The number of memory words that need to be freed.
            tensors_needed_info (TensorsNeededTracker): Contains info about which data tensors (and their respective per-memory tiles) are still needed to be stored (written back) for future computation. Otherwise the data are just evicted.
            verbose (bool): If True, prints details about the replacement process.
        
        Returns:
            tuple or None: If a victim is found, returns a tuple containing:
                - replacement_outcome (str): The outcome of the replacement process ('success' or 'failure').
                - victim_data_id (str): Identifier for the evicted data.
                - victim_data_elems (int): Number of elements evicted.
                - victim_offset (tuple): Offset within the evicted data, as (row_offset, col_offset).
                - data_bitwidth (int): Bitwidth of the evicted data.
            If no victim is found, returns None.
        """
        strategies = {
            'random': lambda: self._select_victim_by_random(tensors_needed_info),
            'lru': lambda: self._select_victim_by_metric('last_access_time', True, tensors_needed_info),
            'lfu': lambda: self._select_victim_by_metric('access_count', True, tensors_needed_info),
            'mru': lambda: self._select_victim_by_metric('last_access_time', False, tensors_needed_info),
            'fifo': lambda: self._select_victim_by_metric('insertion_time', True, tensors_needed_info)
        }
        if self.replacement_strategy not in strategies:
            raise ValueError(f"Unknown replacement strategy: {self.replacement_strategy}")

        victim_info = strategies[self.replacement_strategy]()
        if victim_info is None:
            return None
        else:
            return self._handle_victim(victim_info, free_mem_words, tensors_needed_info, verbose)
    
    def _get_data_for_victim_selection(self, tensors_needed_info):
        eligible_for_removal = {d: c for d, c in self.contents.items() if c['data_amount'] > 0 and d not in self.free_locks}
        # First select the candidates that are not in the tensors_needed_info for this memory
        candidates = {d: c for d, c in eligible_for_removal.items() if not tensors_needed_info.is_in_memory(d, self.name)}
        # Then check for the remaining data if any of the data present in memory is not needed, thus can  be a candidate for removal
        data_to_check = {d: c for d, c in eligible_for_removal.items() if d not in candidates}
        for data_id, metadata in data_to_check.items():
            tiles = tensors_needed_info.get()[data_id]["tiles"][self.name]
            parse_tile_key = lambda k: tuple(map(lambda s: tuple(map(int, s.split('x'))), k.split('_')))
            total_amount = 0
            
            for tile_key in tiles.keys():
                tile_shape, offset = parse_tile_key(tile_key)
                res, info = self._check_tile_presence(data_id, tile_shape, offset)
                
                if res:
                    present = tile_shape[0] * tile_shape[1]
                else:
                    present = tile_shape[0] * tile_shape[1] - info[0][0] * info[0][1]
                total_amount += present
                
            total_in_memory = metadata['data_amount']
            if total_in_memory > total_amount:
                candidates[data_id] = metadata
        return candidates if candidates else None

    def _select_victim_by_random(self, tensors_needed_info):
        """Select a victim data with the smallest data_id for removal."""
        candidates_for_removal = self._get_data_for_victim_selection(tensors_needed_info)
        return min(candidates_for_removal.items(), key=lambda kv: kv[0]) if candidates_for_removal else None

    def _select_victim_by_metric(self, metric, minimize, tensors_needed_info):
        """Select a victim based on a specific metric."""
        victim_choices = self._get_victim_data(tensors_needed_info, metric, minimize)
        return min(victim_choices, key=lambda kv: kv[0]) if victim_choices else None

    def _get_victim_data(self, tensors_needed_info, metric, minimize=True):
        """Get the data IDs with the least or most desirable value based on the given metric.

        Args:
            tensors_needed_info (TensorsNeededTracker): Contains info about which data tensors (and their respective per-memory tiles) are still needed to be stored (written back) for future computation. Otherwise the data are just evicted.
            metric (str): The metric to be used for selecting the victim data. Options include:
                        'last_access_time', 'insertion_time', 'access_count'.
            minimize (bool): Whether to minimize (True) or maximize (False) the metric.

        Returns:
            list: List of data items that are candidates for removal.
        """
        eligible_for_removal = self._get_data_for_victim_selection(tensors_needed_info)

        if not eligible_for_removal:
            return None

        if metric == 'access_count':  # Sum of read and write counts as access count, used for lfu
            metric_key_func = lambda x: x['data_read_count'] + x['data_write_count']
        else:
            metric_key_func = lambda x: x[metric]
            
        if minimize:
            target_metric_data = min(eligible_for_removal.values(), key=metric_key_func)
        else:
            target_metric_data = max(eligible_for_removal.values(), key=metric_key_func)
        
        target_value = metric_key_func(target_metric_data)
        victims = [item for item in eligible_for_removal.items() if metric_key_func(item[1]) == target_value]
        return victims

    def _handle_victim(self, victim_info, free_mem_words, tensors_needed_info, verbose):
        """
        This methods deduces the outcome of the replacement strategy and decides whether to proceed with freeing the memory or write-back the data.

        Args:
            victim_info (tuple): Identification of a selected victim tensor id and its info.
            free_mem_words (int): The number of memory words needed to free in total.
            tensors_needed_info (TensorsNeededTracker): Contains info about which data tensors (and their respective per-memory tiles) are still needed to be stored (written back) for future computation. Otherwise the data are just evicted.           
            verbose (bool): Whether to print verbose logs.

        Returns:
            tuple: A tuple containing:
                - replacement_outcome (str): 'write_back' or 'free', indicating the action taken.
                - victim_data_id (str): Identifier of the victim data.
                - victim_data_category (str): Identifier for 'static' or 'dynamic' victim data tensor.
                - victim_data_shape (tuple): Shape of the presence matrix of the victim data.
                - data_layout_info (str): Type of data layout ('contiguous' or 'tile').
                - data_layout (tuple or int): Layout of the victim data (tile shape or number of contiguous elements).
                - data_offset (tuple): Offset within the data, as (row_offset, col_offset).
                - data_bitwidth (int): Bitwidth of the victim data.
                - free_words (int): Number of memory words freed.
        """
        victim_data_id, victim_metadata = victim_info
        data_layout_info, data_layout, data_offset = self._determine_victim_shape_and_offset(victim_data_id, free_mem_words, tensors_needed_info)

        elements_to_free = 0
        if data_layout_info == "contiguous":
            elements_to_free = data_layout
        else:
            elements_to_free = data_layout[0] * data_layout[1]
        elements_per_word = self.word_size // victim_metadata['data_bitwidth']
        free_words = math.ceil(elements_to_free / elements_per_word)

        if verbose:
            print(f"{self.replacement_strategy.upper()} replacement strategy in {self.name}: freeing {free_words} words from data {victim_data_id} with {data_layout_info} layout, shape {data_layout} and starting offset {data_offset}. Remaining '{free_mem_words-free_words}' words to be freed.")

        assert self.upper_level_memory, f"Upper level memory is not assigned for the current memory {self.name}. Cannot write-back data during replacement strategy."
        # Check if the data_id is still needed (as input for some operation waiting to be computed)
        victim_data_required = True if tensors_needed_info.has_tensor(victim_data_id) else False

        # Data are still needed for future computation and they are not currently present in the upper memory, write-back
        if victim_data_required and victim_data_id not in self.upper_level_memory.contents:
            return 'write_back', victim_data_id, victim_metadata['data_category'], victim_metadata['broadcasted_view'] , victim_metadata['presence_matrix'].shape, data_layout_info, data_layout, data_offset, victim_metadata['data_bitwidth'], free_words
        # Upper level has the data but not the specific data layout and we still need the data for computation, write-back
        elif victim_data_required and (data_layout_info == "contiguous" and not self.upper_level_memory._check_data_presence(victim_data_id, data_layout, data_offset)[0]):
            return 'write_back', victim_data_id, victim_metadata['data_category'], victim_metadata['broadcasted_view'] , victim_metadata['presence_matrix'].shape, data_layout_info, data_layout, data_offset, victim_metadata['data_bitwidth'], free_words
        elif victim_data_required and (data_layout_info == "tile" and not self.upper_level_memory._check_tile_presence(victim_data_id, data_layout, data_offset)[0]):
            return 'write_back', victim_data_id, victim_metadata['data_category'], victim_metadata['broadcasted_view'] , victim_metadata['presence_matrix'].shape, data_layout_info, data_layout, data_offset, victim_metadata['data_bitwidth'], free_words
        # Upper level has the data segment OR the data are no longer required (thus no need for write-back), proceed with free
        else:
            return 'free', victim_data_id, victim_metadata['data_category'], victim_metadata['broadcasted_view'], victim_metadata['presence_matrix'].shape, data_layout_info, data_layout, data_offset, victim_metadata['data_bitwidth'], free_words





    def free(self, data_id, layout_info, data_layout, mem_words, data_offset, tensors_needed_info):
        """
        Free a specified number of memory words from the given data_id and update statistics.

        Args:
            data_id (str): Identifier for the data.
            layout_info (str): Type of data layout in memory to be freed. 'contiguous' or 'tile'
            data_layout (tuple or int): Shape of the tile as (rows, cols) or number of contiguous elements.
            mem_words (int): Number of data words to be freed from memory.
            data_offset (tuple): Offset within the data, as (row_offset, col_offset).
            tensors_needed_info (TensorsNeededTracker): Contains info about which data tensors (and their respective per-memory tiles) are still needed to be stored (written back) for future computation. Otherwise the data are just evicted.
        """
        if data_id in self.contents:
            data_metadata = self.contents[data_id]
            current_word_amount = data_metadata['word_amount']

            if mem_words == current_word_amount:  # Free the entire data block
                self.current_usage -= mem_words
                self.fragmented_bits -= data_metadata['fragmented_bits']

                if not tensors_needed_info.has_tensor(data_id):  # Presence matrix for this data tensor not needed anymore since it will never be required
                    data_metadata['presence_matrix'] = None     
                else:  # Otherwise clear the whole presence matrix of any elements
                    self._clear_presence_matrix(data_metadata)

                data_metadata['word_amount'] = 0
                data_metadata['data_amount'] = 0
                data_metadata['fragmented_bits'] = 0
                data_metadata['insertion_time'] = 0  # We fully removed the data from memory at this point..
            else:  # Free only a part of the data block
                if layout_info == "contiguous":
                    # Mark presence of required number of elements as False within the presence matrix
                    self._free_presence_matrix_data(data_metadata, data_layout, data_offset)
                    data_metadata['data_amount'] -= data_layout
                else:
                    # Mark presence of required number of elements as False within the presence matrix
                    self._free_presence_matrix_tile(data_metadata, data_layout, data_offset)                    
                    data_metadata['data_amount'] -= data_layout[0] * data_layout[1]

                self.current_usage -= mem_words
                data_metadata['word_amount'] -= mem_words
                # Update fragmentation info
                data_bits_remaining = data_metadata['data_amount'] * data_metadata['data_bitwidth']
                new_fragmented_bits = (data_metadata['word_amount'] * self.word_size) - data_bits_remaining
                old_fragmented_bits = data_metadata['fragmented_bits']
                data_metadata['fragmented_bits'] = new_fragmented_bits
                self.fragmented_bits -= (old_fragmented_bits-new_fragmented_bits)

    # Statistics report methods
    def get_ports_utilization(self):
        """
        Creates and returns a dictionary containing the utilization stats for each port.
        The dictionary includes global cycles, idle cycles, and utilization percentage for each port.
        """
        utilization_dict = {}
        for port_id, cycle_info in enumerate(self.cycles_per_ports):
            total_cycles = cycle_info[0]
            idle_cycles = cycle_info[1]
            active_cycles = total_cycles - idle_cycles
            utilization_percentage = (active_cycles / total_cycles) * 100 if total_cycles > 0 else 0
            
            utilization_dict[port_id] = {
                'global_cycles': total_cycles,
                'active_cycles': active_cycles,
                'idle_cycles': idle_cycles,
                'utilization': utilization_percentage
            }
        return utilization_dict
    
    def get_stats(self, log_mem_contents=False):
        """Updates and returns the current statistics."""
        # Create a copy of self.contents without the 'presence_matrix' key-value pair
        filtered_contents = {data_id: {k: v for k, v in data.items() if k != 'presence_matrix'} for data_id, data in self.contents.items()} if log_mem_contents else None
    
        stats = {
            'name': self.name,
            'width': self.width,
            'depth': self.depth,
            'size': self.size,
            'words_capacity': self.words_capacity,
            'word_size': self.word_size,
            'ports': self.ports,
            'bus_bitwidth': self.bus_bitwidth,
            'bandwidth_per_port': self.bandwidth_per_port,
            'total_bandwidth': self.total_bandwidth,
            'cycle_time': self.cycle_time,
            'action_latency': self.action_latency,
            'cycles_per_access': self.mem_access_cycles,
            'active_cycles': self.global_cycles,
            'latency': self.global_cycles * self.cycle_time,
            'current_usage': self.current_usage,
            'data_read_count': self.data_read_count,
            'word_read_count': self.word_read_count,
            'mem_read_count': self.mem_read_count,
            'data_write_count': self.data_write_count,
            'word_write_count': self.word_write_count,
            'mem_write_count': self.mem_write_count,
            'cache_miss_count': self.cache_miss_count,
            'cache_miss_rate': (self.cache_miss_count / self.data_read_count) if self.data_read_count != 0 else 0,
            'cache_hit_rate': (1 - (self.cache_miss_count / self.data_read_count)) if self.data_read_count != 0 else 0,
            'fragmented_bits': self.fragmented_bits,
            'replacement_strategy': self.replacement_strategy,
            'static_energy': self.static_energy * 1e-12,
            'dynamic_energy': self.dynamic_energy * 1e-12,
            'energy': self.energy * 1e-12,
            'edp_latency': (self.energy * 1e-12) * self.global_cycles * self.cycle_time,
            'edp_cycles': (self.energy * 1e-12) * self.global_cycles,
            'area': self.area,
            'contents': filtered_contents,
            'ports_utilization': self.get_ports_utilization()
        }
        return stats

    def reset_stats(self):
        self.global_cycles = 0
        self.area = 0
        self.static_energy = 0
        self.dynamic_energy = 0

        self.current_usage = 0
        self.contents = {}
        self.fetch_locks = {}
        self.free_locks = {}

        self.available_ports = list(range(self.ports))
        self.pending_actions = []
        self.last_reads = {}

        self._cycles_per_ports = [[0, 0] for _ in range(self.ports)]

        self.data_read_count = 0
        self.word_read_count = 0
        self.mem_read_count = 0
        self.data_write_count = 0
        self.mem_write_count = 0
        self.word_write_count = 0
        self.cache_miss_count = 0
        self.fragmented_bits = 0

    # Accelergy export methods
    def generate_action_counts(self):
        return {
            'name': self.name,
            'action_counts': [
                {
                    'name': 'read',
                    'counts': self.mem_read_count
                },
                {
                    'name': 'write',
                    'counts': self.mem_write_count
                },
                {
                    'name': 'leak',
                    'counts': self.global_cycles
                }
            ]
        }
