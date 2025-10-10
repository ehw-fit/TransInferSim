from analyzer.hardware_components.memories.offchip import OffChipMemory
import math

""" COST FUNCTIONS (eval_xxx) """
# Updated analytical compute evaluation with buffers
def eval_operation_duration(operation, matmul_block, get_macs=False):
    """Evaluate operation duration for the given matmul block"""
    # If the operation is not matmul, return 0 tacts
    if operation.computation.split("(")[0] != "matmul":
        return (0, 0) if get_macs else 0

    sa_rows = matmul_block.rows
    sa_cols = matmul_block.columns
    dims_x = operation.input_data[list(operation.input_data.keys())[0]]["tile_shape"]
    dims_y = operation.input_data[list(operation.input_data.keys())[1]]["tile_shape"]

    if dims_x[1] == dims_y[0]:
        m, k, n = dims_x[0], dims_x[1], dims_y[1]
    elif dims_x[1] == dims_y[1]:  # Transpose the second matrix
        m, k, n = dims_x[0], dims_x[1], dims_y[0]
    else:
        raise ValueError(f"Incompatible dimensions for matmul: X:{dims_x}, Y:{dims_y}")

    # Temporal tiling over K due to buffer capacity
    buf_k = getattr(matmul_block, "buffer_length", None)
    P = 1 if (buf_k is None or buf_k <= 0 or buf_k >= k) else math.ceil(k / buf_k)

    def tile_cycles(rows, cols):
        # For one spatial tile of size rows×cols, with K split into P temporal tiles
        # per-temptile cycles (K_p) = (rows + cols - 2) + K_p; sum across all cycles(K_p) = cycles(K)
        return k + P * (rows + cols - 2)

    # Full spatial tiles
    spat_subtiles_cols = m // sa_rows
    spat_subtiles_rows = n // sa_cols
    spat_subtiles = spat_subtiles_cols * spat_subtiles_rows
    cycles = spat_subtiles * tile_cycles(sa_rows, sa_cols)

    # if not.. partial subtiles are needed to be also computed (the sa is not fully utilized in this case, but we dont care)
    partitioned_rows = m % sa_rows
    partitioned_cols = n % sa_cols

    total_macs = 0
    total_macs += spat_subtiles * sa_rows * sa_cols * k

    if partitioned_rows != 0:
        cycles += (partitioned_rows + sa_cols + k - 2) * spat_subtiles_rows
        total_macs += spat_subtiles_rows * partitioned_rows * sa_cols * k

    if partitioned_cols != 0:
        cycles += (sa_rows + partitioned_cols + k - 2) * spat_subtiles_cols
        total_macs += spat_subtiles_cols * sa_rows * partitioned_cols * k

    # if both.. then we need to compute the partial subtile in the corner
    if partitioned_rows != 0 and partitioned_cols != 0:
        cycles += (partitioned_rows + partitioned_cols + k - 2)
        total_macs += partitioned_rows * partitioned_cols * k

    if get_macs:
        return cycles, total_macs
    else:
        return cycles















# TODO add multiple port synchronization for reading more than one data simultaneously
def _cycles_per_xfer(mem, bits):
    """One blocking transfer on one port. No overlap, now (e.g. multiple ports for more data, more systolic arrays TODO)"""
    if mem.__class__ == OffChipMemory:
        bursts = math.ceil(bits / mem._min_burst_bits)
        serialize_time_s = (bursts * (mem.burst_length / mem.prefetch_factor)) / mem.bus_clock_hz
        serialize_core_cycles = math.ceil(serialize_time_s / mem.cycle_time)
        return mem.mem_access_cycles + serialize_core_cycles
    else:
        bus_transfers = int(math.ceil(bits / mem.bus_bitwidth))
        return mem.mem_access_cycles + bus_transfers

def _cycles_for_path(bits, path, write: bool = False, final_operation: bool = False):
    """
    Compute cycles along a memory path.
    
    Path is traversed either like:
        [upper ... lower], e.g. [dram, sram, row/col buf] for read
        [lower ... upper], e.g. [matmul registers, sram, (dram)] for read
            NOTE: only final operation's output is written all the way to DRAM
    """
    assert len(path) >= 2

    total = 0
    legs_detail = []
    
    if not write:
        # READ: upper -> lower (all legs)
        pairs = ((path[i], path[i-1]) for i in range(len(path) - 1, 0, -1))
        op_kind = "read"
    else:
        # WRITE: lower -> upper (first leg unless final op)
        if final_operation:
            pairs = ((path[i], path[i+1]) for i in range(0, len(path) - 1))
        else:
            pairs = ((path[0], path[1]),)  # only the first hop
        op_kind = "write"

    for src, dst in pairs:
        mem = src if op_kind == "read" else dst
        cycles = _cycles_per_xfer(mem, bits)
        total += cycles
        legs_detail.append({
            "src": getattr(src, "name", str(src)),
            "dst": getattr(dst, "name", str(dst)),
            "op": op_kind,
            "cycles": cycles,
        })
    return total, legs_detail, op_kind

def eval_mem_time_cycles(operation, data_bitwidth_bits, path_map):
    """
    Sum read-time cycles for each tensor along its full memory path.
    - path_map: dict[tensor_name -> list of mem objs or names], e.g.
        {'input': [row_buf, dram], 'weight': [col_buf, dram]}
      Order should be [lower, ..., upper]; we traverse top->down for read, down-> for write
    """
    # infer (m,k,n) like compute fn
    inps = list(operation.input_data.keys())
    out = list(operation.output_data.keys())
    dims_x = operation.input_data[inps[0]]["tile_shape"]
    dims_y = operation.input_data[inps[1]]["tile_shape"]
    if dims_x[1] == dims_y[0]:
        m, k, n = dims_x[0], dims_x[1], dims_y[1]
    elif dims_x[1] == dims_y[1]:
        m, k, n = dims_x[0], dims_x[1], dims_y[0]
    else:
        raise ValueError(f"Incompatible dimensions: X:{dims_x}, Y:{dims_y}")

    bits_A = m * k * data_bitwidth_bits
    bits_B = k * n * data_bitwidth_bits
    bits_out = m * n * data_bitwidth_bits
    sizes = {inps[0]: bits_A, inps[1]: bits_B, out[0]: bits_out}

    total_cycles = 0
    breakdown = {}


    for tensor, path in path_map.items():
        bits = sizes[tensor]
        
        if tensor == out[0]: # output needs to be written not read
            t_cycles, legs, op_kind = _cycles_for_path(bits, path, write=True, final_operation=operation._is_root)
        else:
            t_cycles, legs, op_kind = _cycles_for_path(bits, path)

        breakdown[tensor] = {
            "bits": bits,
            "legs": legs,
            "total_cycles": t_cycles,
            "memory_operation": op_kind
        }
        # print(f"data tensor {tensor} {t_cycles}  {breakdown[tensor]}\n")
        total_cycles += t_cycles

    return total_cycles, breakdown






# Utils
def _get_mem_path(mem):
    path = []
    while mem is not None:
        path.append(mem)
        mem = getattr(mem, "upper_level_memory", None)
    return path

def get_memory_paths(inputs, output, matmul_block):
    in_a, in_b = inputs
    (_, out_val), = output.items()
    category_map = {"static": "static_param_memory", "dynamic": "dynamic_param_memory"}
    mem_category = lambda k: category_map.get(k, k)

    in_a_path = [matmul_block.row_buffer] + _get_mem_path(getattr(matmul_block, mem_category(inputs[in_a]['data_category']), None))
    in_b_path = [matmul_block.col_buffer] + _get_mem_path(getattr(matmul_block, mem_category(inputs[in_b]['data_category']), None))
    out_path  = [matmul_block] + _get_mem_path(getattr(matmul_block, mem_category(out_val['data_category']), None))
    return in_a_path, in_b_path, out_path

