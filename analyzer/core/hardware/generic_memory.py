from abc import ABC
import math
import numpy as np
import random
from collections import OrderedDict


class GenericMemory(ABC):
    """Abstract class for a generic memory block."""
    def __init__(self, width: int, depth: int, bus_bitwidth: int, action_latency: float, cycle_time: float, ports: int = 1, word_size: int = 8, name: str = "generic_memory", replacement_strategy = "random", parent_component = None, accelergy_class = ""):
        assert bus_bitwidth >= word_size, f"Bus bitwidth must be greater than or equal to word size for {self.__class__.__name__}."
        assert word_size % 2 == 0, f"Word size must be a multiple of 2 for {self.__class__.__name__}."
        assert bus_bitwidth % 2 == 0, f"Bus bitwidth must be a multiple of 2 for {self.__class__.__name__}."
        assert ports >= 1, f"Number of ports must be greater than or equal to 1 for {self.__class__.__name__}."
        assert width > 0, f"Width must be a positive integer for {self.__class__.__name__}."
        assert depth > 0, f"Depth must be a positive integer for {self.__class__.__name__}."
        assert action_latency > 0, f"Action latency must be a positive integer for {self.__class__.__name__}."
        assert cycle_time > 0, f"Cycle time must be a positive number for {self.__class__.__name__}."

        self.name = name
        self.width = width
        self.depth = depth
        self.size = width * depth
        self.cycle_time = cycle_time  # Accelerator cycle time
        self.parent_component = parent_component
        self._auto_interconnect_set = False  # Used for auto interconnection feature between other HW components to avoid repated interconnetct

        # Functional characteristics
        self.word_size = word_size
        self.block_size = width // word_size  # Affects number of words accessed by one memory read/write operation
        self.ports = ports
        self.action_latency = action_latency
        self.mem_access_cycles = int(math.ceil(self.action_latency / self.cycle_time))
        self.bus_bitwidth = bus_bitwidth
        self.bandwidth_per_port = self.bus_bitwidth / (self.mem_access_cycles * self.cycle_time)  # in bits per second
        self.total_bandwidth = self.bandwidth_per_port * self.ports

        # Accelergy attributes
        self.accelergy_class = accelergy_class
        self._area = 0
        self._energy = 0

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
                f"Block Size: {self.block_size} "
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
        return self._energy

    @energy.setter
    def energy(self, value):
        self._energy = value

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
        """
        Returns the maximum current cycle count across all ports.
        """
        return max(cycle_list[0] for cycle_list in self.cycles_per_ports)

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
        local_random = random.Random()
        min_cycle_count = min(self.cycles_per_ports[port][0] for port in self.available_ports)
        port = local_random.choice([p for p in self.available_ports if self.cycles_per_ports[p][0] == min_cycle_count])
        self.available_ports.remove(port)
        return port

    def update_per_ports_cycles(self, port, cycles, update_idle: bool = False):
        """
        Updates the first element of the tuple (current cycle count) for the specified port.
        """
        self.cycles_per_ports[port][0] += cycles
        if update_idle:
            self.cycles_per_ports[port][1] += cycles

    def synchronize_per_ports_cycles(self):
        max_cycle_count = max(cycle_list[0] for cycle_list in self.cycles_per_ports)

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
        print(mem_paths)

        if read_mode == 'read_tile':
            if tile_shape is None or offset is None:
                raise ValueError(f"tile_shape and offset must be provided for 'read_tile' mode in data query operation for {self.name} memory.")
            check_presence = lambda mem: mem._check_tile_presence(data_id, tile_shape, offset)
        else:  # read continuous
            if elems_to_read is None or offset is None:
                raise ValueError(f"elems_to_read and offset must be provided for 'read_elements' mode in data query operation for {self.name} memory.")
            check_presence = lambda mem: mem._check_data_presence(data_id, elems_to_read, offset)

        def traverse_and_check(mem, subpaths):
            """Recursively traverse memory paths to check for data presence."""
            # Check if data is found in the current memory
            
            
            # TODO WHAT IF ONLY A PORTION IS THERE!!!! WE MUST READ PORTION...
            data_found, missing_data = check_presence(mem)
            if data_found:
                print(f"Data '{data_id}' found in memory {mem.name}")
                return mem
            else:
                # Recursively check in the lower-level submemories
                for lower_mem, lower_subpaths in subpaths.items():
                    print(f"Checking submemory {lower_mem.name}")
                    mem_found = traverse_and_check(lower_mem, lower_subpaths)
                    if mem_found is not None:
                        print(f"našlo se to yes... v {mem_found.name}")
                        return mem_found  # Data found in a lower memory
                return None  # Data not found in any submemory

        # Traverse the memory paths to find where is the data ID with its specific tile/amount present
        for mem, subpaths in mem_paths.items():
            #if data_id in mem.contents:
            #    print(f"Data '{data_id}' found in memory path starting from {mem.name}")
            #    # TODO... IF WE FIND THAT DATA IS IN PARTICULAR PATH.. WE NEED TO GET IT FROM THAT PATH!
            #
            
            mem_found = traverse_and_check(mem, subpaths)
            if mem_found is not None:
                print(f"ano, '{data_id}' tak vracím dál.. {mem_found.name}")
                return mem_found  # Data found in a lower memory
                    
                #print(f"Data '{data_id}' found in memory path starting from {mem.name}")
                
                # we should return the memory where the data reside... and then.. we basically must fetch data from the highest memory that has the data... into its upper memory... and as next event select this event still? the attempt to read?
                #return True
                
                #print(f"Data '{data_id}' not found in memory path starting from {mem.name}")
        
        # TODO... ONLY WHEN ALL FAIL..  (I.E. DATA IS NOT PRESENT IN ANY OF THE LOWER MEMS.. WE SHOULD PASS THROUG WITH FALSE AND ACCESS FROM UPPER)
        return None
            
        

            














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

    def _determine_victim_shape_and_offset(self, victim_data_id, max_num_words):
        """
        Determine the number of elements to be freed from the victim presence matrix 
        along with its data shape (cointiguous block or tile) and starting offset.

        Args:
            victim_data_id (str): Name of the victim data.
            max_num_words (int): Number of words required to be freed in total.

        Returns:
            tuple: (layout_type, layout_info (tuple or int), offset (tuple))
                - layout_info: 'contiguous' if elements are in a contiguous block, 'tile' if in a tile.
                - data_layout: Shape of the tile as (rows, cols) or number of contiguous elements.
                - offset: Starting point (row, col) for freeing.
        """
        data_metadata = self.contents[victim_data_id]
        presence_matrix = data_metadata['presence_matrix']
        _, cols = presence_matrix.shape
        elements_per_word = self.word_size // data_metadata['data_bitwidth']
        max_elems_to_erase = max_num_words * elements_per_word

        last_true_indices = np.argwhere(presence_matrix)
        assert last_true_indices.size > 0, "Internal error! No True values found in the victim presence matrix, this should not happen."

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

    def read(self, read_mode, data_id, tensor_shape, data_bitwidth, port_id, tile_shape=None, elems_to_read=None, offset=None):
        """
        Read data from memory, either as a specific tile or a number of elements.

        Args:
            read_mode (str): Read mode for the memory block. Either 'read_elements' and 'read_tile' are supported.
            data_id (str): Identifier for the data.
            tensor_shape (tuple): Shape of the whole tensor to be read from memory, defined as (rows, cols).
            data_bitwidth (int): Bitwidth of the data elements to be read from memory.
            port_id (int): ID of port used for retrieval of the data.
            tile_shape (tuple): Shape of the tile to read, defined as (tile_rows, tile_cols). Used only in 'read_tile' mode.
            elems_to_read (int): Number of elements to read from memory. Used only in 'read_elements' mode.
            offset (tuple): Offset within the data, as (row_offset, col_offset).

        Returns:
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
            # Lock the data_id for fetching
            self.fetch_locks[data_id] = []
            self.cache_miss_count += tile_shape[0]*tile_shape[1] if read_mode == 'read_tile' else elems_to_read
            data_layout = tile_shape if read_mode == 'read_tile' else elems_to_read
            return False, (data_layout, offset)

        data_metadata = self.contents[data_id]

        if read_mode == 'read_tile':
            if tile_shape is None or offset is None:
                raise ValueError(f"tile_shape and offset must be provided for 'read_tile' mode in read operation from {self.name} memory.")

            data_found, missing_data = self._check_tile_presence(data_id, tile_shape, offset)
            if data_found:
                tile_rows, tile_cols = tile_shape
                elements_per_word = self.word_size // data_bitwidth
                mem_words = math.ceil(tile_rows * tile_cols / elements_per_word)
                mem_reads = int(math.ceil(mem_words / self.block_size))

                self.data_read_count += tile_rows * tile_cols
                self.word_read_count += mem_words
                self.mem_read_count += mem_reads

                data_metadata['data_read_count'] += tile_rows * tile_cols
                data_metadata['word_read_count'] += mem_words
                data_metadata['mem_read_count'] += mem_reads

                cycles = math.ceil((mem_reads * self.block_size) / self.bus_bitwidth) * self.mem_access_cycles
                data_metadata['last_access_time'] = self.cycles_per_ports[port_id][0] + cycles
                return True, cycles
            else:
                # If data not in memory, the data must be fetched
                missing_tile, missing_tile_offset = missing_data
                data_metadata['cache_miss_count'] += missing_tile[0]*missing_tile[1]
                self.cache_miss_count += missing_tile[0]*missing_tile[1]
                self.fetch_locks[data_id] = []
                return False, (missing_tile, missing_tile_offset)

        elif read_mode == 'read_elements':
            if elems_to_read is None or offset is None:
                raise ValueError(f"elems_to_read and offset must be provided for 'read_elements' mode in read operation from {self.name} memory.")

            data_found, missing_data = self._check_data_presence(data_id, elems_to_read, offset)
            if data_found:
                elements_per_word = self.word_size // data_bitwidth
                mem_words = math.ceil(elems_to_read / elements_per_word)
                mem_reads = int(math.ceil(mem_words / self.block_size))

                self.data_read_count += elems_to_read
                self.word_read_count += mem_words
                self.mem_read_count += mem_reads

                data_metadata['data_read_count'] += elems_to_read
                data_metadata['word_read_count'] += mem_words
                data_metadata['mem_write_count'] += mem_reads

                cycles = math.ceil((mem_reads * self.block_size) / self.bus_bitwidth) * self.mem_access_cycles
                data_metadata['last_access_time'] = self.cycles_per_ports[port_id][0] + cycles
                return True, cycles
            else:
                # If data not in memory, the data must be fetched
                missing_elems, missing_tile_offset = missing_data
                data_metadata['cache_miss_count'] += missing_elems
                self.cache_miss_count += missing_elems
                self.fetch_locks[data_id] = []
                return False, (missing_elems , missing_tile_offset)
        else:
            raise ValueError(f"Unknown read mode: {read_mode} requested from memory {self.name}. Supported modes are 'read_elements' and 'read_tile'.")

    def write(self, write_mode, data_id, tensor_shape, data_bitwidth, port_id, tensors_needed_info, tile_shape=None, elems_to_write=None, offset=None, verbose=False):
        """
        Write data to memory, either as a specific tile or a number of elements.
        
        Args:
            write_mode (str): Write mode for the memory block. Either 'write_elements' and 'write_tile' are supported.
            data_id (str): Identifier for the data.
            tensor_shape (tuple): Shape of the whole tensor to be written to memory, defined as (rows, cols).
            data_bitwidth (int): Bitwidth of the data elements to be written to memory.
            port_id (int): ID of port used for storing the data.
            tensors_needed_info TODO
            tile_shape (tuple): Shape of the tile to write, defined as (tile_rows, tile_cols). Used only in 'write_tile' mode.
            elems_to_write (int): Number of elements to write to memory. Used only in 'write_elements' mode.
            offset (tuple): Offset within the data, as (row_offset, col_offset).
            verbose (bool): If True, prints detailed information about the write process.

        Returns:
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
        new_elements = 0

        # NOTE: We do not allow for bit packing to split data across words (i.e. for 8bit words and 3bit data, 2 bits will be unused for the word)
        elements_per_word = self.word_size // data_bitwidth
        if write_mode == "write_tile":
            if tile_shape is None or offset is None:
                raise ValueError(f"tile_shape and offset must be provided for 'write_tile' mode in write operation to {self.name} memory.")

            # Check if the data can fit in memory at all
            total_words_required = math.ceil(tile_shape[0] * tile_shape[1] / elements_per_word)
            assert total_words_required <= self.size, f"Data {data_id} too large ({total_words_required} words) to fit in memory of size {self.size} words."

            # Calculate the required amount of elements needed to be written into memory
            new_elements = self._update_presence_matrix(mode="tile", data_metadata=data_metadata, offset=offset, tile_shape=tile_shape, inplace=False)
        elif write_mode == "write_elements":
            if elems_to_write is None or offset is None:
                raise ValueError(f"elems_to_write and offset must be provided for 'write_elements' mode in write_elements operation to {self.name} memory.")

            # Check if the data can fit in memory at all
            total_words_required = math.ceil(elems_to_write / elements_per_word)
            assert total_words_required <= self.size, f"Data {data_id} too large ({total_words_required} words) to fit in memory of size {self.size} words."
            
            # Calculate the required amount of elements needed to be written into memory
            new_elements = self._update_presence_matrix(mode="elements", data_metadata=data_metadata, offset=offset, elems_to_write=elems_to_write, inplace=False)
        else:
            raise ValueError(f"Unknown write mode: {write_mode} requested from memory {self.name}. Supported modes are 'write_elements' and 'write_tile'.")

        # Determine the required space (number of words) to be written into memory
        write_mem_words = math.ceil(new_elements / elements_per_word)

        # Memory full, we need to free some space before writing the data
        if self.current_usage + write_mem_words > self.size:
            assert self.replacement_strategy is not None, f"Memory '{self.name}' is full and has no replacement strategy defined! Define it during instantiation and re-run."
            words_to_free = (self.current_usage + write_mem_words) - self.size
            if verbose:
                print(f"Not enough space for write of data '{data_id}' into memory '{self.name}'. Need space for {words_to_free} words. Finding a victim data to free space.")

            while words_to_free != 0:
                replacement_result = self._apply_replacement_strategy(words_to_free, tensors_needed_info, verbose)
                # No valid data found for replacement now, we need to reschedule this operation
                # (some other event may be writing back the only possible data id and no other event can at the same time)
                if replacement_result is None:
                    return False, None
    
                replacement_outcome, victim_data_id, victim_tensor_shape, victim_layout_info, victim_layout_type, victim_offset, victim_data_bitwidth, victim_words = replacement_result
            

                if replacement_outcome == "write_back":
                    self.free_locks[victim_data_id] = []
                    # Return to Discrete Simulation events and proceed with write-back (this write operation will be rescheduled)
                    return False, (victim_data_id, victim_tensor_shape, victim_layout_info, victim_layout_type, victim_offset, victim_data_bitwidth)
                else:  # Proceed with freeing the data.. no need for writ/back (either the data are not important anymore or are already present in the upper memory)
                    self.free(victim_data_id, victim_layout_info, victim_layout_type, victim_words, victim_offset, tensors_needed_info)
                    # self.unlock_free_lock(victim_data_id)
                    words_to_free -= victim_words

        # Finally proceed with the write operation
        if write_mode == "write_tile":
            self._update_presence_matrix(mode="tile", data_metadata=data_metadata, offset=offset, tile_shape=tile_shape, inplace=True)
        else: 
            self._update_presence_matrix(mode="elements", data_metadata=data_metadata, offset=offset, elems_to_write=elems_to_write, inplace=True)
        
        data_bits = new_elements * data_bitwidth
        word_bits = write_mem_words * self.word_size
        fragmented_bits = word_bits - data_bits
        mem_writes = int(math.ceil(write_mem_words / self.block_size))

        data_metadata['data_amount'] += new_elements
        data_metadata['data_write_count'] += new_elements
        data_metadata['word_amount'] += write_mem_words
        data_metadata['word_write_count'] += write_mem_words
        data_metadata['mem_write_count'] += mem_writes
        data_metadata['fragmented_bits'] += fragmented_bits

        self.current_usage += write_mem_words
        self.data_write_count += new_elements
        self.word_write_count += write_mem_words
        self.mem_write_count += mem_writes
        self.fragmented_bits += fragmented_bits

        cycles = math.ceil((mem_writes * self.block_size) / self.bus_bitwidth) * self.mem_access_cycles
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
                'count': 1
            }

        # Check for potential memory thrashing
        if len(self.last_reads) == 2:
            read_counts = list(self.last_reads.values())
            # Check if both reads have occurred 20 times or more
            if all(read['count'] >= 20 for read in read_counts):
                print("Memory thrashing detected: Both last memory fetch operations have occurred 20 times or more.")
                return True

        return False

    def _apply_replacement_strategy(self, free_mem_words, tensors_needed_info, verbose):
        """
        Applies the configured replacement strategy to free up the necessary space in memory.

        Args:
            free_mem_words (int): The number of memory words that need to be freed.
            tensors_needed_info TODO
            verbose (bool): If True, prints details about the replacement process.
        
        Returns:
            # TODO
            tuple: (replacement_outcome (str), victim_data_id (str), victim_data_elems (int), victim_offset (tuple), data_bitwidth (int))
        """
        strategies = {
            'random': lambda: self._select_victim_by_random(),
            'lru': lambda: self._select_victim_by_metric('last_access_time', True),
            'lfu': lambda: self._select_victim_by_metric('access_count', True),
            'mru': lambda: self._select_victim_by_metric('last_access_time', False),
            'fifo': lambda: self._select_victim_by_metric('insertion_time', True)
        }
        if self.replacement_strategy not in strategies:
            raise ValueError(f"Unknown replacement strategy: {self.replacement_strategy}")

        victim_search_result = strategies[self.replacement_strategy]()
        if victim_search_result is None:
            return None
        else:
            victim_data_id, victim_metadata = victim_search_result
            return self._handle_victim(victim_data_id, victim_metadata, free_mem_words, tensors_needed_info, verbose)
        
    def _select_victim_by_random(self):
        """Randomly select a victim data for removal."""
        eligible_for_removal = {d: c for d, c in self.contents.items() if c['word_amount'] > 0 and d not in self.free_locks and d not in self.fetch_locks}
        local_random = random.Random()
        return local_random.choice(list(eligible_for_removal.items())) if eligible_for_removal else None
    
    def _select_victim_by_metric(self, metric, minimize):
        """Select a victim based on a specific metric."""
        victim_choices = self._get_victim_data(metric, minimize)
        return random.choice(victim_choices) if victim_choices else None
    
    def _get_victim_data(self, metric, minimize=True):
        """Get the data IDs with the least or most desirable value based on the given metric.

        Args:
            metric (str): The metric to be used for selecting the victim data. Options include:
                        'last_access_time', 'insertion_time', 'access_count'.
            minimize (bool): Whether to minimize (True) or maximize (False) the metric.

        Returns:
            list: List of data items that are candidates for removal.
        """
        eligible_for_removal = {d: c for d, c in self.contents.items() if c['word_amount'] > 0 and d not in self.free_locks and d not in self.fetch_locks}
        
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

    def _handle_victim(self, victim_data_id, victim_metadata, free_mem_words, tensors_needed_info, verbose):
        """
        This methods deduces the outcome of the replacement strategy and decides whether to proceed with freeing the memory or write-back the data.

        Args:
            victim_data_id (str): The ID of the victim data.
            victim_metadata (dict): Metadata of the victim data.
            free_mem_words (int): The number of memory words needed to free in total.
            tensors_needed_info TODO
            verbose (bool): Whether to print verbose logs.

        Returns:
            # TODO CHANGE!!!
            tuple: (replacement_outcome (str), victim_data_id (str), victim_data_elems (int), victim_offset (tuple), data_bitwidth (int))
        """
        data_layout_info, data_layout, data_offset = self._determine_victim_shape_and_offset(victim_data_id, free_mem_words)

        elements_to_free = 0
        if data_layout_info == "contiguous":
            elements_to_free = data_layout
        else:
            elements_to_free = data_layout[0] * data_layout[1]
        elements_per_word = self.word_size // victim_metadata['data_bitwidth']
        free_words = elements_to_free // elements_per_word

        if verbose:
            print(f"{self.replacement_strategy.upper()} replacement strategy in {self.name}: freeing {free_words} words from data {victim_data_id} with {data_layout_info} layout, shape {data_layout} and starting offset {data_offset}. Remaining '{free_mem_words-free_words}' words to be freed.")

        assert self.upper_level_memory, f"Upper level memory is not assigned for the current memory {self.name}. Cannot write-back data during replacement strategy."
        # Check if the data_id is still needed (as input for some operation waiting to be computed)
        victim_data_required = True if victim_data_id in tensors_needed_info else False

        # Data are still needed for future computation and they are not currently present in the upper memory, write-back
        if victim_data_required and victim_data_id not in self.upper_level_memory.contents:
            return 'write_back', victim_data_id, victim_metadata['presence_matrix'].shape, data_layout_info, data_layout, data_offset, victim_metadata['data_bitwidth'], free_words
        # Upper level has the data but not the specific data layout and we still need the data for computation, write-back
        elif victim_data_required and (data_layout_info == "contiguous" and not self.upper_level_memory._check_data_presence(victim_data_id, data_layout, data_offset)[0]):
            return 'write_back', victim_data_id, victim_metadata['presence_matrix'].shape, data_layout_info, data_layout, data_offset, victim_metadata['data_bitwidth'], free_words
        elif victim_data_required and (data_layout_info == "tile" and not self.upper_level_memory._check_tile_presence(victim_data_id, data_layout, data_offset)[0]):
            return 'write_back', victim_data_id, victim_metadata['presence_matrix'].shape, data_layout_info, data_layout, data_offset, victim_metadata['data_bitwidth'], free_words
        # Upper level has the data segment OR the data are no longer required (thus no need for write-back), proceed with free
        else:
            return 'free', victim_data_id, victim_metadata['presence_matrix'].shape, data_layout_info, data_layout, data_offset, victim_metadata['data_bitwidth'], free_words

    def free(self, data_id, layout_info, data_layout, mem_words, data_offset, tensors_needed_info):
        """
        Free a specified number of memory words from the given data_id and update statistics.

        Args:
            data_id (str): Identifier for the data.
            layout_info (str): Type of data layout in memory to be freed. 'contiguous' or 'tile'
            # TODO layout int or tuple...
            # TODO  mem words
            data_offset (tuple): Offset within the data, as (row_offset, col_offset).
        """
        if data_id in self.contents:
            data_metadata = self.contents[data_id]
            current_word_amount = data_metadata['word_amount']

            if mem_words == current_word_amount:  # Free the entire data block
                self.current_usage -= mem_words
                self.fragmented_bits -= data_metadata['fragmented_bits']

                if data_id not in tensors_needed_info:  # Presence matrix for this data tensor not needed anymore since it will never be required
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
            active_cycles = cycle_info[0]
            idle_cycles = cycle_info[1]
            total_cycles = active_cycles + idle_cycles
            utilization_percentage = (active_cycles / total_cycles) * 100 if total_cycles > 0 else 0
            
            utilization_dict[port_id] = {
                'global_cycles': active_cycles,
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
            'word_size': self.word_size,
            'ports': self.ports,
            'bus_bitwidth': self.bus_bitwidth,
            'bandwidth_per_port': self.bandwidth_per_port,
            'total_bandwidth': self.total_bandwidth,
            'cycle_time': self.cycle_time,
            'action_latency': self.action_latency,
            'cycles_per_access': self.mem_access_cycles,
            'cycles': self.global_cycles,
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
            'energy': self.energy * 1e-12,
            'edp_latency': (self.energy * 1e-12) * self.global_cycles * self.cycle_time,
            'edp_cycles': (self.energy * 1e-12) * self.global_cycles,
            'area': self.area,
            'contents': filtered_contents,
            'ports_utilization': self.get_ports_utilization()
        }
        return stats

    # Accelergy export methods
    def generate_action_counts(self):
        return {
            'name': self.name,
            'action_counts': [
                {
                    'name': 'read',
                    'counts': self.word_read_count if self.mem_read_count == 0 else self.mem_read_count
                },
                {
                    'name': 'write',
                    'counts': self.word_write_count if self.mem_write_count == 0 else self.mem_write_count
                }
            ]
        }
