from analyzer.core.hardware.generic_memory import GenericMemory
import math
import re


class OffChipMemory(GenericMemory):
    def __init__(self, name: str, width: int, depth: int, action_latency: float, bus_bitwidth: int, cycle_time: float, bus_clock_hz: float, word_size: int = 8, ports: int = 2, banks: int = 1, prefetch_factor: int = 2, burst_length: int = 1, replacement_strategy = None, parent_component = None, accelergy_class = "offchip_DRAM"):
        super().__init__(width=width, depth=depth, bus_bitwidth=bus_bitwidth, action_latency=action_latency, word_size=word_size, ports=ports, name=name, cycle_time=cycle_time, replacement_strategy=replacement_strategy, parent_component=parent_component, accelergy_class=accelergy_class)
        assert banks > 0, f"Banks must be a positive integer for {self.__class__.__name__}."
        assert prefetch_factor in (1, 2, 4, 8, 16), "Prefetch factor must be one of 1 (SDR), 2 (DDR), 4 (DDR2), 8 (DDR3-4) , 16 (DDR5)"
        assert (burst_length in {1, 2, 4, 8, 16}) and (burst_length % prefetch_factor == 0 or (prefetch_factor >= 8 and burst_length * 2 == prefetch_factor)), \
        "Burst length must be in {1,2,4,8,16} and be a multiple of prefetch_factor (or half when prefetch>=8 for burst-chop)"

        self.prefetch_factor = prefetch_factor
        self.bus_clock_hz = bus_clock_hz
        self.banks = banks  # TODO Currently does nothing
        self.burst_length = burst_length
        self._min_burst_bits = burst_length * bus_bitwidth
        self._replacement_strategy = None  # Off-chip memory does not support replacement strategies
        # Peak transfers per second on one port
        beats_per_second = prefetch_factor * bus_clock_hz
        self.bandwidth_per_port = self.bus_bitwidth * beats_per_second
        self.total_bandwidth = self.bandwidth_per_port * self.ports
        # TODO add more fine-grained support later
        #self.pagesize = 8192
        #self.pages = self.size / self.pagesize

    def __str__(self):
        base_str = super().__str__()
        return (f"{base_str[:-1]} "
                f"Banks: {self.banks} "
                f"Prefetch Factor: {self.prefetch_factor} "
                f"Bus Clock Frequency: {round(self.bus_clock_hz / 1e9, 1)} GHz>")

    # Discrete simulation methods
    def read(self, read_mode, data_id, data_category, tensor_shape, data_bitwidth, port_id, tile_shape=None, elems_to_read=None, offset=None):
        """
        Read data from off-chip memory, either as a specific tile or a number of elements. It is assumed all data are there (besides those computed on-the-fly).

        Args:
            read_mode (str): Read mode for the memory block. Either 'read_elements' and 'read_tile' are supported.
            data_id (str): Identifier for the data.
            data_category (str): Identifier for 'static' or 'dynamic' data tensor.
            tensor_shape (tuple): Shape of the whole tensor to be read from memory, defined as (rows, cols).
            data_bitwidth (int): Bitwidth of the data elements to be read from memory.
            port_id (int): ID of port used for retrieval of the data. Not used here.
            tile_shape (tuple): Shape of the tile to read, defined as (tile_rows, tile_cols). Used only in 'read_tile' mode.
            elems_to_read (int): Number of elements to read from memory. Used only in 'read_elements' mode.
            offset (tuple): Offset within the data, as (row_offset, col_offset).

        Returns:
            tuple: (success (bool), action_cycles (int))
        """
        tensor_rows, tensor_cols = tensor_shape
        # NOTE: We do not allow for bit packing to split data across words (i.e. for 8bit words and 3bit data, 2 bits will be unused for the word)
        elements_per_word = self.word_size // data_bitwidth
        mem_words = math.ceil(tensor_rows * tensor_cols / elements_per_word)

        if data_id not in self.contents:
            data_bits = tensor_rows * tensor_cols * data_bitwidth
            word_bits = mem_words * self.word_size
            fragmented_bits = word_bits - data_bits
            self.fragmented_bits += fragmented_bits

            # Assuming the data is there even if it's not recorded in contents
            # TODO ADD CATEGORICAL DISTINGUISHING BETWEEN DATA TYPES (SIMILAR FOR OPS)
            # Inputs and weights are assumed to be always inside DRAM prior to starting the simulation
            if re.fullmatch(r'(?:.*_)?input(?:_.*)?', data_id) or data_category == "static":
                self.contents[data_id] = {
                    'data_category': data_category,
                    'presence_matrix': self._initialize_presence_matrix(tensor_shape, 1),
                    'data_amount': tensor_rows * tensor_cols,
                    'data_read_count': 0,
                    'mem_read_count': 0,
                    'data_write_count': 0,
                    'mem_write_count': 0,
                    'word_amount': mem_words,
                    'word_read_count': 0,
                    'word_write_count': 0,
                    'data_bit_packing': self.word_size != data_bitwidth,
                    'data_bitwidth': data_bitwidth,
                    'fragmented_bits': fragmented_bits,
                    'insertion_time': 0,  # Ignoring for off-chip memory
                    'last_access_time': 0,  # Ignoring for off-chip memory
                    'cache_miss_count': 0  # Ignoring for off-chip memory
                }
        
        read_mem_words = 0        
        data_metadata = self.contents[data_id]

        if read_mode == "read_tile":
            if tile_shape is None or offset is None:
                raise ValueError(f"tile_shape and offset must be provided for 'read_tile' mode in read operation from {self.name} memory.")

            tile_rows, tile_cols = tile_shape
            elems_to_read = tile_rows * tile_cols
            read_mem_words = math.ceil(tile_rows * tile_cols / elements_per_word)
            tile_found, _ = self._check_tile_presence(data_id, tile_shape, offset)
            assert tile_found, f"Tile of data id '{data_id}' not present in offchip memory, this should not happen!"

        elif read_mode == "read_elements":
            if elems_to_read is None or offset is None:
                raise ValueError(f"elems_to_read and offset must be provided for 'read_elements' mode in read operation from {self.name} memory.")

            read_mem_words = math.ceil(elems_to_read / elements_per_word)
            elems_found = self._check_data_presence(data_id, elems_to_read, offset)
            assert elems_found, f"Data of data id '{data_id}' not present in offchip memory, this should not happen!"
        else:
            raise ValueError(f"Unknown read mode: {read_mode} requested from memory {self.name}. Supported modes are 'read_elements' and 'read_tile'.")

        self.data_read_count += elems_to_read
        self.word_read_count += read_mem_words
        mem_reads = int(math.ceil(read_mem_words / self.row_words))
        self.mem_read_count += mem_reads
        
        data_metadata['data_read_count'] += elems_to_read
        data_metadata['word_read_count'] += read_mem_words
        data_metadata['mem_read_count'] += mem_reads

        total_bits = read_mem_words * self.word_size
        bursts = math.ceil(total_bits / self._min_burst_bits)
        transfers_per_second = self.prefetch_factor * self.bus_clock_hz
        burst_t = self.burst_length / transfers_per_second
        serialize_time_s = bursts * burst_t
        serialize_core_cycles = math.ceil(serialize_time_s / self.cycle_time)
        cycles = self.mem_access_cycles + serialize_core_cycles
        return True, cycles

    def write(self, write_mode, data_id, data_category, tensor_shape, data_bitwidth, port_id, tile_shape=None, elems_to_write=None, offset=None, verbose=False, **kwargs):
        """
        Write data to off-chip memory, either as a specific tile or a number of elements.

        Args:
            write_mode (str): Write mode for the memory block. Either 'write_elements' and 'write_tile' are supported.
            data_id (str): Identifier for the data.
            data_category (str): Identifier for 'static' or 'dynamic' data tensor.
            tensor_shape (tuple): Shape of the whole tensor to be written to memory, defined as (rows, cols).
            data_bitwidth (int): Bitwidth of the data elements to be written to memory.
            port_id (int): ID of port used for storing the data. Not used here.
            tile_shape (tuple): Shape of the tile to write, defined as (tile_rows, tile_cols). Used only in 'write_tile' mode.
            elems_to_write (int): Number of elements to write to memory. Used only in 'write_elements' mode.
            offset (tuple): Offset within the data, as (row_offset, col_offset).
            verbose (bool): If True, prints detailed information about the write process. Not used here. Defaults to False.

        Returns:
            tuple: (success (bool), action_cycles (int))
        """
        if data_id not in self.contents:
            self.contents[data_id] = {
                'data_category': data_category,
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
                'insertion_time': 0,  # Ignoring for off-chip memory
                'last_access_time': 0,  # Ignoring for off-chip memory
                'cache_miss_count': 0  # Ignoring for off-chip memory
            }
        data_metadata = self.contents[data_id]
        # NOTE: We do not allow for bit packing to split data across words (i.e. for 8bit words and 3bit data, 2 bits will be unused for the word)
        elements_per_word = self.word_size // data_bitwidth

        # NOTE: we assume DRAM can store all data
        if write_mode == "write_tile":
            if tile_shape is None or offset is None:
                raise ValueError(f"tile_shape and offset must be provided for 'write_tile' mode in write operation to {self.name} memory.")

            # Calculate the required space in memory
            new_elements = self._update_presence_matrix(mode="tile", data_metadata=data_metadata, offset=offset, tile_shape=tile_shape)
            write_mem_words = math.ceil(new_elements / elements_per_word)
        elif write_mode == "write_elements":
            if elems_to_write is None or offset is None:
                raise ValueError(f"elems_to_write and offset must be provided for 'write_elements' mode in write_elements operation to {self.name} memory.")

            # Calculate the required space in memory
            new_elements = self._update_presence_matrix(mode="elements", data_metadata=data_metadata, offset=offset, elems_to_write=elems_to_write)
            write_mem_words = math.ceil(new_elements / elements_per_word)
        else:
            raise ValueError(f"Unknown write mode: {write_mode} requested from memory {self.name}. Supported modes are 'write_elements' and 'write_tile'.")
  
        # Proceed with the write operation
        data_bits = new_elements * data_bitwidth
        word_bits = write_mem_words * self.word_size
        fragmented_bits = word_bits - data_bits
        mem_writes = int(math.ceil(write_mem_words / self.row_words))

        data_metadata['data_amount'] += new_elements
        data_metadata['data_write_count'] += new_elements
        data_metadata['word_amount'] += write_mem_words
        data_metadata['word_write_count'] += write_mem_words
        data_metadata['mem_write_count'] += mem_writes
        data_metadata['fragmented_bits'] += fragmented_bits

        self.data_write_count += new_elements
        self.word_write_count += write_mem_words
        self.mem_write_count += mem_writes
        self.fragmented_bits += fragmented_bits

        total_bits = write_mem_words * self.word_size
        bursts = math.ceil(total_bits / self._min_burst_bits)
        transfers_per_second = self.prefetch_factor * self.bus_clock_hz
        burst_t = self.burst_length / transfers_per_second
        serialize_time_s = bursts * burst_t
        serialize_core_cycles = math.ceil(serialize_time_s / self.cycle_time)
        cycles = self.mem_access_cycles + serialize_core_cycles
        return True, cycles

    def free(self, data_id, data_elems, data_offset):
        """Raise an exception when attempting to free memory on off-chip memory."""
        raise NotImplementedError("Free operation is not supported for off-chip memory.")

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
