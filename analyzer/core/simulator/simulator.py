import heapq
import random
import os
import subprocess
import yaml
import time
import tempfile
from abc import ABC, abstractmethod
from collections import OrderedDict
from .events import MatmulStartEvent, MemoryReadCompleteEvent, MemoryWriteBackCompleteEvent, MemoryFetchCompleteEvent, SpatialTileComputeEvent, TemporalTileComputeEvent, MemoryWriteCompleteEvent, SpatialTileCompleteEvent, TemporalTileCompleteEvent, ConcatenationCompleteEvent, MatmulCompleteEvent

from .events import MemoryWriteBackEvent, MemoryWriteEvent, MemoryFetchEvent

import csv
import os
        
        
class MemoryUsageCSVLogger:
    EXCLUDE_KEYS = {
        "contents",           # explicitly excluded
        "usage_breakdown",    # already flattened
        "ports_utilization",  # nested, not CSV-safe
    }

    def __init__(self, csv_path, hw_arch_name):
        self.csv_path = csv_path
        self.hw_arch_name = hw_arch_name

        # ALWAYS start fresh
        with open(self.csv_path, "w", newline="") as f:
            pass

        self.header_written = False

    def _flatten_mem_stat(self, mem_stat):
        """
        Flatten mem_stat into scalar key-value pairs,
        excluding nested or unwanted fields.
        """
        flat = {}

        for k, v in mem_stat.items():
            if k in self.EXCLUDE_KEYS:
                continue

            # only allow scalar values
            if isinstance(v, (int, float, str, bool)):
                flat[f"mem_{k}"] = v

        return flat

    def log(self, op_index, op_compute_time, req_macs, op_start_time, op_end_time, op_name, matmul, mem_stat):
        ub = mem_stat["usage_breakdown"]

        # Base row (semantic identifiers)
        row = {
            "hw_arch": self.hw_arch_name,
            "matmul": matmul,
            "op_index": op_index,
            "macs": req_macs,
            "op_compute_time": op_compute_time,
            "op_start_time": op_start_time,
            "op_end_time": op_end_time,
            "op_name": op_name,
            "memory_name": mem_stat["name"],
        }

        # Usage breakdown (explicit)
        row.update({
            "used_pct": ub["used_pct"],
            "needed_pct": ub["needed_pct"],
            "obsolete_pct": ub["obsolete_pct"],
            "free_pct": ub["free_pct"],
            "used_words": mem_stat["current_usage"],
            "capacity_words": mem_stat["words_capacity"],
        })

        # Flattened memory statistics
        row.update(self._flatten_mem_stat(mem_stat))

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)

def compute_memory_usage_breakdown(mem_stats, tensors_needed):
    capacity_words = mem_stats["words_capacity"]
    used_words = mem_stats["current_usage"]

    contents = mem_stats.get("contents", {})
    if not contents:
        return {
            "used_pct": 0.0,
            "needed_pct": 0.0,
            "obsolete_pct": 0.0,
            "free_pct": 1.0
        }

    needed_words_total = 0

    for data_id, data_info in contents.items():
        # If tensor is still needed in the future,
        # then everything it occupies in this memory is needed
        if data_id in tensors_needed:
            needed_words_total += data_info["word_amount"]

    # Sanity clamp (important)
    needed_words_total = min(needed_words_total, used_words)

    obsolete_words = used_words - needed_words_total
    free_words = capacity_words - used_words

    return {
        "used_pct": used_words / capacity_words,
        "needed_pct": needed_words_total / capacity_words,
        "obsolete_pct": obsolete_words / capacity_words,
        "free_pct": free_words / capacity_words
    }


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

class EventScheduler:
    def __init__(self, execution_graph, log_state_before: bool = False, verbose: bool = False):
        self.events = []
        self._events_log = []
        self.global_cycles = 0
        self._next_sequence_id = 0
        self.verbose = verbose
        self.log_state_before = log_state_before
        self.execution_graph = execution_graph

    def schedule_event(self, event_time, event, priority):
        # TODO DELETE/PLAY WITH tests with analytical functions
        # if event.__class__ is MatmulStartEvent:
        #     # print("here matmul")
        #     # print(event.operation)
        #     # print(event.matmul_array)
        #     op = event.operation

        #     if not op.computation.startswith('concat'):
        #         inputs = op.input_data
        #         output = op.output_data
        #         # for each input tensor, get memory block path
        #         data_mem_paths = get_memory_paths(inputs, output, event.matmul_array)
        #         input_paths = dict(zip(list(inputs.keys()) + list(output.keys()), data_mem_paths))

        #         # here we have all three data paths for inputs and outputs... for regular data (not final... just write to one upper level.. if final.. all the way to dram (all way))
        #         movement_cycles, detail = eval_mem_time_cycles(op, data_bitwidth_bits=8, path_map=input_paths)
        #         compute_cycles = eval_operation_duration(op, event.matmul_array)
        #         total_no_overlap = movement_cycles + compute_cycles

        #         # total_movement_cycles += movement_cycles
        #         # total_compute_cycles += compute_cycles
        #         # total_cycles += total_no_overlap
        #         print(movement_cycles, compute_cycles, total_no_overlap)

        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1
        heapq.heappush(self.events, (event_time, priority, sequence_id, event))
        if event.__class__ in [MatmulStartEvent]:
            self._events_log.append(event)
        if self.verbose:
            print(f"Scheduled event {event} with priority {priority} at time {event_time}")
            print(f"current events:  {self.events}")

    def next_event(self):
        if self.events:
            event_time, _, _, event = heapq.heappop(self.events)
            return (event_time, event)
        return None

    def has_events(self):
        return len(self.events) > 0

class AbstractSimulationEngine(ABC):
    def __init__(self, model, execution_graph, hw_arch, permutation_seed: int = None, scheduling_seed: int = None, verbose: bool = False, log_state_before: bool = False, store_to_tmp: bool = False, **kwargs):
        self.model = model
        self.execution_graph = execution_graph
        self.hw_arch = hw_arch
        self._stimulus_plan = []  # TODO, stores the ops from the simulation run, which serve as a plan for validation through RTL behavioral simulation (think which only are truly essential to store, complete ones?)
        self._events_log = []  # all events as they executed in order; used for step/wise simulation and retraction of past events
        self._scheduled = []  # list of (op, matmul_array) tuples in execution-completion order; populated by both static and dynamic engines
        self.store_to_tmp = store_to_tmp
        if permutation_seed is None:
            permutation_seed = random.randint(0, 2**32 - 1)
        self.permutation_seed = permutation_seed
        self.scheduling_seed = scheduling_seed
        self.scheduler = EventScheduler(execution_graph, log_state_before=log_state_before, verbose=verbose)
        self._perm_rng = random.Random(self.permutation_seed)
        self._sched_rng = random.Random(self.scheduling_seed) if self.scheduling_seed is not None else None
        self.verbose = verbose
        self.log_state_before = log_state_before

    def reset(self, execution_graph):
        self._stimulus_plan = []
        self._events_log = []
        self._scheduled = []
        self.execution_graph = execution_graph
        self.scheduler = EventScheduler(self.execution_graph, log_state_before=self.log_state_before, verbose=self.verbose)
        self._perm_rng = random.Random(self.permutation_seed)
        self._sched_rng = random.Random(self.scheduling_seed) if self.scheduling_seed is not None else None

    @abstractmethod
    def initialize_simulation(self):
        pass

    @abstractmethod
    def run_simulation(self):
        pass

    def retrieve_stats(self):
        for m in self.hw_arch.memory_blocks + [self.hw_arch.dram]:
            m.global_cycles = self.hw_arch.global_cycles  # TODO change for powergating support!
        start_time = time.time()
        # Once analysis is done, the HW arch is populated with runtime stats, which are used to generate action counts for accelergy along with the underlying hw arch description
        # TODO maybe rethink?.. basically to eliminate requerying Accelergy for ART/ERT of same HW across many experiments, we reuse them
        if self.store_to_tmp:
            tmp_dir = tempfile.gettempdir()
            pid = os.getpid()

            arch_yaml = os.path.join(tmp_dir, f"{self.hw_arch.name}_{pid}_accelergy_arch.yaml")
            action_yaml = os.path.join(tmp_dir, f"{self.hw_arch.name}_{pid}_action_counts.yaml")
            accelergy_out_dir = os.path.join(tmp_dir, f"{self.hw_arch.name}_{pid}_ACCELERGY_OUTPUT")
            log_path = os.path.join(tmp_dir, f"{self.hw_arch.name}_{pid}_accelergy.log")
        else:
            arch_yaml = f"{self.hw_arch.name}_accelergy_arch.yaml"
            action_yaml = f"{self.hw_arch.name}_action_counts.yaml"
            accelergy_out_dir = f"{self.hw_arch.name}_ACCELERGY_OUTPUT"
            log_path = f"{self.hw_arch.name}_accelergy.log"

        self.hw_arch.get_accelergy_description(out_fname=arch_yaml)
        self.hw_arch.generate_action_counts(out_fname=action_yaml)

        if os.path.exists(accelergy_out_dir):
            command = [
                "accelergy",
                arch_yaml,
                "compound_components.yaml",
                os.path.join(accelergy_out_dir, "ART.yaml"),
                os.path.join(accelergy_out_dir, "ERT.yaml"),
                action_yaml,
                "-o", accelergy_out_dir
            ]
        else:
            command = [
                "accelergy",
                arch_yaml,
                action_yaml,
                "compound_components.yaml",
                "-o", accelergy_out_dir
            ]

        with open(log_path, "w") as f:
            result = subprocess.run(
                command,
                stdout=f,
                stderr=subprocess.STDOUT,  # put stderr into same log
                text=True
            )
        if result.returncode != 0:
            print(f"Accelergy failed. See log: {log_path}")

        # Fill component area stats with the generated stats
        art = load_yaml(f'{accelergy_out_dir}/ART.yaml')
        for item in art["ART"]["tables"]:
            for key, value in self.hw_arch.comp_names_map.items():
                if key in item['name'] and key != self.hw_arch.dram.name:  # We dont want the area of DRAM since it is offchip
                    value.area += item['area']
                    self.hw_arch.area += item['area']

        # Fill component energy stats with the generated stats
        ert = load_yaml(f'{accelergy_out_dir}/ERT_summary.yaml')
        en_sum = load_yaml(f'{accelergy_out_dir}/energy_estimation.yaml')
        for item in en_sum["energy_estimation"]["components"]:
            for key, value in self.hw_arch.comp_names_map.items():
                if key in item['name']:
                    if value in self.hw_arch.memory_blocks + [self.hw_arch.dram]:  # Memory units
                        leak_energy_ref_pJ  = 0
                        for ert_item in ert["ERT_summary"]["table_summary"]:
                            if key in ert_item['name']:
                                # Iterate through the actions to find the 'leak' energy
                                for action in ert_item['actions']:
                                    if action['name'] == 'leak':
                                        leak_energy_ref_pJ  = action['energy']
                                        break
                                if leak_energy_ref_pJ  > 0:
                                    break
                        value.static_energy += value.global_cycles * leak_energy_ref_pJ
                        value.dynamic_energy += item['energy'] - (value.global_cycles * leak_energy_ref_pJ)
                        self.hw_arch.energy += item['energy']
                    elif value in self.hw_arch.matmul_blocks:  # Compute units
                        value.energy += item['energy']
                        self.hw_arch.energy += item['energy']
        self.hw_arch._accelergy_time = time.time() - start_time

class StaticSimulationEngine(AbstractSimulationEngine):
    def __init__(self, model, execution_graph, hw_arch, verbose: bool = False, permutation_seed: int = None, scheduling_seed: int = None, store_to_tmp: bool = False, log_mem_trace: bool = False, mem_trace_path: str = "mem_trace.csv", **kwargs):
        super().__init__(model=model, execution_graph=execution_graph, hw_arch=hw_arch, permutation_seed=permutation_seed, scheduling_seed=scheduling_seed, store_to_tmp=store_to_tmp, verbose=verbose, **kwargs)
        if self.verbose:
            print(f"Permutation seed for Static Simulation Engine set to {self.permutation_seed}")
            mode = "uniform across units" if self.scheduling_seed is None else "random across units"
            print(f"Scheduling seed for Static Simulation Engine set to {self.scheduling_seed} ({mode})\n")

        self.mem_csv_logger = None
        if log_mem_trace:
            self.mem_csv_logger = MemoryUsageCSVLogger(csv_path=mem_trace_path, hw_arch_name=self.hw_arch.name)

    def verbose_permutations(self, permutation):
        print(f"Number of operations: {len(permutation)}")
        print(f"Permutation:" + ", ".join([f"{(f'OP: {n.name}', f'ENCODING_VAL: {n.encoding_value}')}" for n in permutation]) + "\n")

    def generate_permutation(self):
        for node in self.execution_graph.all_nodes:
            lower, upper = node.encoding_range
            node.encoding_value = self._perm_rng.uniform(lower, upper)
        permutation = sorted(self.execution_graph.all_nodes, key=lambda x: x.encoding_value)

        # Assign priority based on sorted position
        for i, node in enumerate(permutation):
            node.priority = i

        if self.verbose:
            self.verbose_permutations(permutation)
        return permutation

    def schedule_operations(self, permutation):
        matmul_arrays = self.hw_arch.matmul_blocks
        assert matmul_arrays, "No compute blocks available in provided HW description"

        if self._sched_rng is None:  # Uniform assignment of operations across computational units
            for i, operation in enumerate(permutation):
                assigned_array = matmul_arrays[i % len(matmul_arrays)]
                assigned_array.plan.append(operation)
        else:  # Random assignment of operations across computational units
            for operation in permutation:
                assigned_array = self._sched_rng.choice(matmul_arrays)
                assigned_array.plan.append(operation)

    def initialize_simulation(self, **kwargs):
        assert self.hw_arch.global_cycles == 0, "The number of cycles should be zero at the beginning of a Static simulation! The Static Simulation Engine is expected to be run over a full exectution graph from the beginning! If you wish to continue ongoing simulation at the current state, try dynamic engine instead."

        if self.verbose:
            print("<ANALYSIS PHASE 0: SCHEDULING OPERATIONS ON HW>")
        permutation = self.generate_permutation()
        self.schedule_operations(permutation)

        if self.verbose:
            print("<ANALYSIS PHASE 1: OPERATIONS SUCCESSFULLY SCHEDULED ACROSS SPATIAL ARRAYS>")
            print("\n".join(f"Plan for {a.name}, Operations: {len(a.plan)}: {[op.name for op in a.plan]}" for a in self.hw_arch.matmul_blocks) + "\n")

        if self.verbose:
            print("<ANALYSIS PHASE 2: STARTING SIMULATION>")
        self.run_simulation()

        if self.verbose:
            print("<ANALYSIS PHASE 3: SIMULATION COMPLETED>")

        self.retrieve_stats()
        self.hw_arch._runtime = self.hw_arch._simulation_time + self.hw_arch._accelergy_time

        if self.verbose:
            print("<ANALYSIS PHASE 4: METRICS RETRIEVED>")

    # TODO DELETE maybe later
    def assert_memory_usage_consistency(self, stats: dict):
        """
        Asserts that for every memory:
        current_usage == sum of word_amount of all data entries in contents
        """

        for mem in stats["memory_stats"]:
            mem_name = mem["name"]
            current_usage = mem["current_usage"]

            contents = mem.get("contents")
            if contents is None:
                # logging disabled, nothing to assert
                continue

            summed_words = 0
            for data_id, data_stats in contents.items():
                # Be defensive about key names
                if "word_amount" in data_stats:
                    summed_words += data_stats["word_amount"]
                elif "Word Amount" in data_stats:
                    summed_words += data_stats["Word Amount"]
                else:
                    raise KeyError(
                        f"[{mem_name}] Missing word_amount for data_id '{data_id}'"
                    )

            # print(f"summed: {summed_words}")
            # print(f"usage: {current_usage}")
            
            assert summed_words == current_usage, (
                f"[Memory usage mismatch] {mem_name}: "
                f"current_usage={current_usage}, "
                f"summed_contents={summed_words}"
            )

    
    
    def run_simulation(self):
        assert self.hw_arch.analysis_done is False, "Simulation has already been run. Please reset the simulation engine before running again."
        start_time = time.time()

        # First we preprocess the arrays to filter out already completed ops TODO should this be here? or rather assert?
        for matmul_array in self.hw_arch.matmul_blocks:
            # We access the first element of the list here to check its status
            while matmul_array.plan and matmul_array.plan[0].is_done:
                matmul_array.plan.pop(0)

        available_arrays = [arr for arr in self.hw_arch.matmul_blocks if not arr.is_busy and arr.plan]
        available_arrays.sort(key=lambda arr: arr.plan[0].priority)
        # Plan first operations of each spatial array into the calendar
        # The event scheduler then handles planning additional events
        for matmul_array in available_arrays:
            operation = matmul_array.plan.pop(0)
            start_event = MatmulStartEvent(operation, matmul_array, self.scheduler, verbose=self.verbose)
            start_event.event_time = self.scheduler.global_cycles
            self.scheduler.schedule_event(start_event.event_time, start_event, operation.priority)

        # After all events have been scheduled, start analyzing them
        while self.scheduler.has_events():
            event_time, event = self.scheduler.next_event()
            event.event_time = event_time
            self.scheduler.global_cycles = self.hw_arch.global_cycles = max(event_time, self.hw_arch.global_cycles)
            event.execute(self.scheduler)

            # To export data for RTL validation/deployment 
            if event.__class__ in [MemoryFetchCompleteEvent, MemoryWriteBackCompleteEvent, MemoryReadCompleteEvent, MemoryWriteCompleteEvent, SpatialTileCompleteEvent, TemporalTileCompleteEvent, MatmulCompleteEvent]:
                self._stimulus_plan.append(event)

            if event.__class__ is MatmulCompleteEvent:
                self._scheduled.append((event.operation, event.matmul_array))
                if self.mem_csv_logger:
                    stats = self.hw_arch.get_statistics(log_mem_contents=True)
                    tensors_needed = self.scheduler.execution_graph.tensors_needed.get()
                    for mem_stats in stats["memory_stats"]:
                        mem_stats["usage_breakdown"] = compute_memory_usage_breakdown(mem_stats, tensors_needed)
                        self.mem_csv_logger.log(
                            op_index=event.operation.priority + 1,
                            op_compute_time=event.operation.compute_time,
                            req_macs=event.operation.macs,
                            op_start_time=event.operation.start_time,
                            op_end_time=event.operation.finish_time,
                            op_name=event.operation.name,
                            matmul=event.matmul_array.name,
                            mem_stat=mem_stats,
                        )

            if self.scheduler.events == []:  # If at the end there are still some pending memory read/write actions, reschedule them
                for mem in self.hw_arch.memory_blocks:
                    while mem.pending_actions:
                        memory_op = mem.pending_actions.pop(0)
                        self.scheduler.schedule_event(self.scheduler.global_cycles, memory_op, memory_op.op_priority)

        self.hw_arch.analysis_done = True
        assert self.scheduler.global_cycles == self.hw_arch.global_cycles

        for m in self.hw_arch.memory_blocks + [self.hw_arch.dram]:
            m.synchronize_per_ports_cycles(self.hw_arch.global_cycles)

        # Check that all computations have been calculated
        assert sum((comp_block.total_flop_computes // 2) for comp_block in self.hw_arch.matmul_blocks) == self.model.num_macs, f"Simulation has not completed all computations! Check possible cause with verbose debug. (expected: {self.model.num_macs}; got: {sum((comp_block.total_flop_computes // 2) for comp_block in self.hw_arch.matmul_blocks)}"
        self.hw_arch._simulation_time = time.time() - start_time


class DynamicSimulationEngine(AbstractSimulationEngine):
    """
    Step-wise simulation:
      - One operation is scheduled and simulated to completion per call to run_simulation_step().
      - Accelergy metrics are generated only when requested (generate_metrics=True).
      - Deterministic by default; randomness governed only by RNG seed (again can be deterministic)
    """
    def __init__(self, model, execution_graph, hw_arch, verbose: bool = False, permutation_seed: int = None, scheduling_seed: int = None, store_to_tmp: bool = False, external_scheduling: bool = False, log_mem_trace: bool = False, mem_trace_path: str = "mem_trace.csv", **kwargs):
        super().__init__(model=model, execution_graph=execution_graph, hw_arch=hw_arch, permutation_seed=permutation_seed, scheduling_seed=scheduling_seed, log_state_before=True, store_to_tmp=store_to_tmp, verbose=verbose, **kwargs)
        self._rr_idx = 0 # Used ONLY IF no scheduling seed is provided (see _schedule_next_op()), corresponds with uniform distribution of OPs to compute units
        self.simulation_step = 0
        self._next_priority = 0
        self._unscheduled = list(self.execution_graph.all_nodes)  # All not yet processed operations
        self._scheduled = []  # List of tuples containing operation and the assigned compute unit
        self._external_scheduling = external_scheduling
        self.mem_csv_logger = None
        if log_mem_trace:
            self.mem_csv_logger = MemoryUsageCSVLogger(csv_path=mem_trace_path, hw_arch_name=self.hw_arch.name)

        # This is used only in case no external scheduler is used, because this adds metadata to operations, potentially rewriting the scheduler's values
        if not external_scheduling:
            self._initialize_encodings()

        if self.verbose:
            mode = "uniform across units" if self.scheduling_seed is None else "random across units"
            print(f"Scheduling seed for Dynamic Simulation Engine set to {self.scheduling_seed} ({mode})\n")

    def _initialize_encodings(self):
        for node in self._unscheduled:
            lower, upper = node.encoding_range
            node.encoding_value = self._perm_rng.uniform(lower, upper)

    def reset(self, execution_graph):
        super().reset(execution_graph)
        self._rr_idx = 0
        self.simulation_step = 0
        self._next_priority = 0
        self._unscheduled = list(execution_graph.all_nodes)
        self._scheduled = []
        if not self._external_scheduling:
            self._initialize_encodings()

    def _schedule_next_op(self):
        """
        Pick the next operation by encoding_value (deterministic) and assign it
        to a matmul array (either round-robin or governed by scheduling seed).

        Returns:
            (op, array): tuple of the chosen operation and the assigned matmul array.
                        Returns (None, None) if no operations remain.
        """
        if not self._unscheduled:
            return None, None

        # Pick operation deterministically
        next_op = min(self._unscheduled, key=lambda x: x.encoding_value)
        next_op.priority = self._next_priority
        self._next_priority += 1

        # Select compute array in round-robin order / randomly by scheduling seed
        matmul_arrays = self.hw_arch.matmul_blocks
        assert matmul_arrays, "No compute blocks available in provided HW description"
        if self._sched_rng is None:
            idx = self._rr_idx % len(matmul_arrays)
            self._rr_idx += 1
            assigned_array = matmul_arrays[idx]
        else:
            assigned_array = self._sched_rng.choice(matmul_arrays)

        if self.verbose:
            print(f"Scheduled {next_op.name} with priority {next_op.priority} → {assigned_array.name}")

        return next_op, assigned_array

    def initialize_simulation(self, exec_op_assign = None, generate_metrics: bool = False, **kwargs):
        self.simulation_step += 1
        if self.verbose:
            print(f"<STEP-WISE SIMULATION (STEP: {self.simulation_step}) ANALYSIS PHASE 0: SCHEDULING OPERATION ON HW>")

        op, arr = self._schedule_next_op() if exec_op_assign is None else exec_op_assign
        if op is None:
            assert not self._unscheduled
            assert sum((comp_block.total_flop_computes // 2) for comp_block in self.hw_arch.matmul_blocks) == self.model.num_macs, f"Simulation has not completed all computations! Check possible cause with verbose debug. (expected: {self.model.num_macs}; got: {sum((comp_block.total_flop_computes // 2) for comp_block in self.hw_arch.matmul_blocks)}"
            self.hw_arch.analysis_done = True
            assert self.scheduler.global_cycles == self.hw_arch.global_cycles

            for m in self.hw_arch.memory_blocks + [self.hw_arch.dram]:
                m.synchronize_per_ports_cycles(self.hw_arch.global_cycles)
            # TODO simulation time?
            self._generate_metrics()
            if self.verbose:
                print(f"<STEP-WISE SIMULATION (STEP: {self.simulation_step}) METRICS RETRIEVED; SIMULATION IS DONE>")
            return

        # Decide on parallel or sequential processing strategy
        furthest_op = None
        if self._scheduled and len(self.hw_arch.matmul_blocks) > 1:            
            # Case 1: Sequential Execution
            # If the current array 'arr' was already the bottleneck (its cycles are the global max), we remove just this one newly evaluated operation
            is_same_array_as_last_op = self._scheduled[-1][1] == arr
            is_array_bottleneck = arr.global_cycles == self.scheduler.global_cycles
            if is_same_array_as_last_op and is_array_bottleneck:
                pass
            # Case 2: Possible parallelism (we must also retract some previous operations)
            else:
                if arr.global_cycles == 0:
                    furthest_op = self._scheduled[0][0]
                else:
                    # Iterate backwards (most recent first) through executed operations
                    for exec_op, exec_arr in reversed(self._scheduled):
                        if exec_arr == arr:
                            if exec_op.start_time == 0:
                                furthest_op = self._scheduled[0][0]
                            else:
                                furthest_op = exec_op
                            break
                        elif exec_op in op._children:
                            furthest_op = exec_op
                            break



        if furthest_op:
            self.retract_simulation_steps(wall_op=furthest_op, remove_last=False, generate_metrics=False)
        arr.plan.append(op)

        self._unscheduled.remove(op)
        self._scheduled.append((op, arr))

        if self.verbose:
            print(f"<STEP-WISE SIMULATION (STEP: {self.simulation_step}) ANALYSIS PHASE 1: OPERATION '{op.name}' SUCCESSFULLY SCHEDULED TO SPATIAL ARRAY '{arr.name}'>")

        if self.verbose:
            print(f"<STEP-WISE SIMULATION (STEP: {self.simulation_step}) ANALYSIS PHASE 2: EXECUTION SIMULATION STEP>")

        

                
                
                
        self.run_simulation(generate_metrics)
        


        if self.verbose:
            print(f"<STEP-WISE SIMULATION (STEP: {self.simulation_step}) ANALYSIS PHASE 3: SIMULATION STEP COMPLETED>")

        if generate_metrics:
            self._generate_metrics()
            if self.verbose:
                print(f"<STEP-WISE SIMULATION (STEP: {self.simulation_step}) ANALYSIS PHASE 4: METRICS RETRIEVED>")












    def find_speculative_retraction_wall(self, op, arr, original_block_global_cycles: dict, original_global_cycles: int):
        """
        After a speculative initialize_simulation(op, arr) call, determine the wall_op
        to pass to retract_simulation_steps() so the trial's effect is fully undone.
        Defaults to op itself (retract only the single speculative step).
        """
        furthest_op = op
        if self._scheduled and len(self.hw_arch.matmul_blocks) > 1:
            is_same_array_as_last_op = self._scheduled[-1][1] == arr
            is_array_bottleneck = original_block_global_cycles[arr] == original_global_cycles
            if not (is_same_array_as_last_op and is_array_bottleneck):
                if original_block_global_cycles[arr] == 0:
                    furthest_op = self._scheduled[0][0]
                else:
                    for exec_op, exec_arr in reversed(self._scheduled[:-1]):
                        if exec_arr == arr:
                            furthest_op = self._scheduled[0][0] if exec_op.start_time == 0 else exec_op
                            break
                        elif exec_op in op._children:
                            furthest_op = exec_op
                            break
        return furthest_op

    def retract_simulation_steps(self, wall_op=None, remove_last: bool = False, drop_op=None, generate_metrics: bool=True):
        assert wall_op is not None, "Missing info about the last operation to retract"
        # ---------- ALIGN WITH SOME OP-START BOUNDARY ----------
        # Find the N-th MatmulStartEvent from the end = restore point
        target_start_event = None
        for e in reversed(self.scheduler._events_log):
            if e.operation is wall_op and e.is_rescheduled == False:
                target_start_event = e
                break

        assert target_start_event is not None, "Error during dynamic simulation, target MatmulStartEvent not found in stimulus plan."

        # Events that will be restored to scheduler, do not retract them
        components = target_start_event.state_before_changes.get("components", [])
        scheduler_component_state = next((comp for comp in components if comp.get("obj") == target_start_event._scheduler), None)
        saved_sched_events = []
        saved_global_cycles = None
        if scheduler_component_state:
            saved_sched_events = next((p["value"] for p in scheduler_component_state.get("params", []) if p.get("param") == "events"), [])
            saved_next_sequence_id = next((p["value"] for p in scheduler_component_state.get("params", []) if p.get("param") == "_next_sequence_id"), None)
            saved_global_cycles = next((p["value"] for p in scheduler_component_state.get("params", []) if p.get("param") == "global_cycles"), None)
        
        dropped_op = None

        while self.scheduler._events_log:
            e = self.scheduler._events_log.pop()
            e.retract_changes()

            # If we are also retracting the last executed "speculative" operation,
            # remember what operation it was so as not to reappend it for running 
            # the remainder of the simulation after retracting
            if remove_last and dropped_op is None:
                dropped_op = e.operation                

            if e.is_rescheduled == False and e.operation is not dropped_op and e.operation is not drop_op:
                e.operation.pending_event = None
                e.matmul_array.plan.insert(0, e.operation)

                if e.operation._is_root:
                    self.hw_arch.analysis_done = False

            if e is target_start_event:
                break

        # Clear any stale heap events left by intermediate retract_changes() calls.
        # Then mark all arrays as idle so they're re-scheduled fresh from their plans.
        # Arrays that were mid-computation had their ops re-inserted into plans above,
        # so they will get fresh MSEs when run_simulation's for-loop runs next.
        self.scheduler.events = saved_sched_events

        # Non-retracted ops in _events_log completed during a previous restore. Two cases:
        # - Active in restore (tile events in saved_sched_events): will re-complete naturally.
        #   Their dependency links must be live so remove_dependency() works when they fire.
        # - Not active (completed before the snapshot): truly done, must be cleared from
        #   parents' children so they don't cause ghost deadlocks.
        active_ops_in_restore = {
            e[-1].operation
            for e in self.scheduler.events  # already set to saved_sched_events above
            if hasattr(e[-1], 'operation') and e[-1].operation is not None
        }
        # Memory events (MemoryReadEvent, MemoryFetchEvent, etc.) have no .operation, so
        # ops mid-memory-transfer are missed by the set comprehension above. Include them
        # via the busy array's current_operation_event, which retract_changes already restored.
        for sa in self.hw_arch.matmul_blocks:
            if sa.is_busy and sa.current_operation_event is not None:
                active_ops_in_restore.add(sa.current_operation_event.operation)
        for e in self.scheduler._events_log:
            if not e.is_rescheduled:
                node = e.operation
                if node in active_ops_in_restore:
                    # Will re-complete naturally: restore dependency link so remove_dependency works
                    node.is_done = False
                    node.readd_dependency()
                else:
                    # Completed before snapshot: mark done, clear stale dependency links
                    node.is_done = True
                    node.pending_event = None
                    for parent in node.parents:
                        if node in parent.children:
                            parent.children.remove(node)

        self.scheduler._next_sequence_id = saved_next_sequence_id
        self.scheduler.global_cycles = saved_global_cycles
        self.hw_arch.global_cycles = saved_global_cycles
        if generate_metrics:
            self._generate_metrics()

    def run_simulation(self, generate_metrics: bool = False):
        """
        Execute exactly ONE operation to completion:
          1) Choose op (if not provided) using deterministic encoding order.
          2) Assign it to an array (RR or seeded RNG).
          3) Enqueue MatmulStartEvent at current cycle.
          4) Process the event loop until the queue is empty AND memory pending actions are drained.

        If generate_metrics=True, Accelergy is invoked at the end of the step.
        """
        assert self.hw_arch.analysis_done is False, "Simulation has already been run. Please reset the simulation engine before running again."
        start_time = time.time()

        for matmul_array in self.hw_arch.matmul_blocks:
            # We access the first element of the list here to check its status
            while matmul_array.plan and matmul_array.plan[0].is_done:
                matmul_array.plan.pop(0)

        available_arrays = [arr for arr in self.hw_arch.matmul_blocks if not arr.is_busy and arr.plan]
        available_arrays.sort(key=lambda arr: arr.plan[0].priority)
        # Remove stale MSEs from the heap before creating fresh ones.
        # retract_simulation_steps() restores the heap from a snapshot that may
        # contain MSEs for arrays that are now re-available. Leaving them in causes
        # both the stale and fresh MSE to fire, executing tile computation twice and
        # inflating operation.macs / compute_time.
        stale_arrays = {arr for arr in available_arrays}
        if stale_arrays:
            before = len(self.scheduler.events)
            self.scheduler.events = [
                e for e in self.scheduler.events
                if not (isinstance(e[-1], MatmulStartEvent) and e[-1].matmul_array in stale_arrays)
            ]
            if len(self.scheduler.events) != before:
                heapq.heapify(self.scheduler.events)

        # Plan first operations of each spatial array into the calendar
        # The event scheduler then handles planning additional events
        for matmul_array in available_arrays:
            operation = matmul_array.plan.pop(0)
            start_event = MatmulStartEvent(operation, matmul_array, self.scheduler, verbose=self.verbose, log_state_before=self.log_state_before)
            start_event.event_time = matmul_array.global_cycles
            self.scheduler.schedule_event(start_event.event_time, start_event, operation.priority)

        while self.scheduler.has_events():
            event_time, event = self.scheduler.next_event()
            # print((event_time, event))
            event.event_time = event_time
            self.scheduler.global_cycles = self.hw_arch.global_cycles = max(event_time, self.hw_arch.global_cycles)

            event.execute(self.scheduler)

            if generate_metrics and self.mem_csv_logger and event.__class__ is MatmulCompleteEvent:
                stats = self.hw_arch.get_statistics(log_mem_contents=True)
                tensors_needed = self.scheduler.execution_graph.tensors_needed.get()
                for mem_stats in stats["memory_stats"]:
                    mem_stats["usage_breakdown"] = compute_memory_usage_breakdown(mem_stats, tensors_needed)
                    self.mem_csv_logger.log(
                        op_index=event.operation.priority + 1,
                        op_compute_time=event.operation.compute_time,
                        req_macs=event.operation.macs,
                        op_start_time=event.operation.start_time,
                        op_end_time=event.operation.finish_time,
                        op_name=event.operation.name,
                        matmul=event.matmul_array.name,
                        mem_stat=mem_stats,
                    )

            if self.scheduler.events == []:  # If at the end there are still some pending memory read/write actions, reschedule them
                for mem in self.hw_arch.memory_blocks:
                    while mem.pending_actions:
                        memory_op = mem.pending_actions.pop(0)
                        self.scheduler.schedule_event(self.scheduler.global_cycles, memory_op, memory_op.op_priority)

        if not self._unscheduled:
            assert sum((comp_block.total_flop_computes // 2) for comp_block in self.hw_arch.matmul_blocks) == self.model.num_macs, f"Simulation has not completed all computations! Check possible cause with verbose debug. (expected: {self.model.num_macs}; got: {sum((comp_block.total_flop_computes // 2) for comp_block in self.hw_arch.matmul_blocks)}"
            self.hw_arch.analysis_done = True
            assert self.scheduler.global_cycles == self.hw_arch.global_cycles

            for m in self.hw_arch.memory_blocks + [self.hw_arch.dram]:
                m.synchronize_per_ports_cycles(self.hw_arch.global_cycles)

            if generate_metrics:
                self._generate_metrics()
            if self.verbose:
                print(f"<STEP-WISE SIMULATION (STEP: {self.simulation_step}) METRICS RETRIEVED; SIMULATION IS DONE>")
        self.hw_arch._simulation_time = time.time() - start_time

    # ---------- Helper ----------
    def _generate_metrics(self):
        """Run Accelergy."""
        # Reset energy/area accumulators to avoid double counting across runs
        self.hw_arch.area = 0
        self.hw_arch.energy = 0
        for comp in self.hw_arch.comp_names_map.values():
            comp.area = 0
            if comp in self.hw_arch.matmul_blocks:
                comp.energy = 0
            else:
                comp.static_energy = 0
                comp.dynamic_energy = 0

        self.retrieve_stats()
        self.hw_arch._runtime = self.hw_arch._simulation_time + self.hw_arch._accelergy_time
