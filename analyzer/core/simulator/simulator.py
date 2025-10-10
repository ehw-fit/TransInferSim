import heapq
import random
import os
import subprocess
import yaml
import time
import tempfile
from abc import ABC, abstractmethod
from .events import MatmulStartEvent, MemoryWriteBackEvent, MemoryFetchEvent, MemoryReadEvent, TemporalTileCompleteEvent, MemoryReadCompleteEvent, MemoryWriteBackCompleteEvent, MemoryFetchCompleteEvent, MemoryWriteCompleteEvent, SpatialTileCompleteEvent


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

class EventScheduler:
    def __init__(self, execution_graph, verbose: bool = False):
        self.events = []
        self.global_cycles = 0
        self.verbose = verbose
        self.execution_graph = execution_graph
        self._deterministic_key = None

    @property
    def deterministic_key(self):
        return self._deterministic_key

    @deterministic_key.setter
    def deterministic_key(self, value: int):
        self._deterministic_key = value

    def schedule_event(self, event_time, event):
        heapq.heappush(self.events, (event_time, event))
        if self.verbose:
            print(f"Scheduled event {event} at time {event_time}")

    def next_event(self):
        if self.events:
            return heapq.heappop(self.events)
        return None

    def has_events(self):
        return len(self.events) > 0

class AbstractSimulationEngine(ABC):
    def __init__(self, model, execution_graph, hw_arch, deterministic_seed: int = None, deterministic: bool = False, verbose: bool = False, store_to_tmp: bool = False, **kwargs):
        self.model = model
        self.execution_graph = execution_graph
        self.hw_arch = hw_arch
        self.scheduler = EventScheduler(execution_graph, verbose=verbose)
        self._stimulus_plan = []  # TODO, stores the ops from the simulation run, which serve as a plan for validation through RTL behavioral simulation
        self.deterministic = deterministic
        self.store_to_tmp = store_to_tmp
        if deterministic_seed is None:
            deterministic_seed = random.randint(0, 2**32 - 1)
        self.deterministic_seed = deterministic_seed
        if self.deterministic:
            self.scheduler.deterministic_key = self.deterministic_seed
        self.verbose = verbose

    def reset(self, execution_graph):
        self._stimulus_plan = []
        self.execution_graph = execution_graph
        self.scheduler = EventScheduler(self.execution_graph, verbose=self.verbose)
        if self.deterministic:
            self.scheduler.deterministic_key = self.deterministic_seed

    def verbose_permutations(self, permutation):
        print(f"Number of operations: {len(permutation)}")
        print(f"Permutation:" + ", ".join([f"{(f'OP: {n.name}', f'ENCODING_VAL: {n.encoding_value}')}" for n in permutation]) + "\n")

    @abstractmethod
    def generate_permutation(self):
        pass

    @abstractmethod
    def schedule_operations(self, permutation):
        pass
    
    @abstractmethod
    def initialize_simulation(self):
        pass

    def run_simulation(self):
        assert self.hw_arch.analysis_done is False, "Simulation has already been run. Please reset the simulation engine before running again."
        start_time = time.time()
        # Plan first operations of each spatial array into the calendar
        # The event scheduler then handles planning additional events
        for matmul_array in self.hw_arch.matmul_blocks:
            if matmul_array.plan:
                operation = matmul_array.plan.pop(0)
                start_event = MatmulStartEvent(operation, matmul_array, verbose=self.verbose)
                start_event.event_time = self.scheduler.global_cycles
                self.scheduler.schedule_event(self.scheduler.global_cycles, start_event)

        # After all events have been scheduled, start analyzing them
        while self.scheduler.has_events():
            event_time, event = self.scheduler.next_event()
            event.event_time = event_time
            self.scheduler.global_cycles = self.hw_arch.global_cycles = max(event_time, self.hw_arch.global_cycles)

            # Used for export of final execution plan
            if event.__class__ in [MemoryFetchCompleteEvent, MemoryWriteBackCompleteEvent, MemoryReadCompleteEvent, MemoryWriteCompleteEvent, SpatialTileCompleteEvent, TemporalTileCompleteEvent]:
                self._stimulus_plan.append(event)
            
            event.execute(self.scheduler)
            if self.scheduler.events == []:  # If at the end there are still some pending memory read/write actions, reschedule them
                for mem in self.hw_arch.memory_blocks:
                    while mem.pending_actions:
                        memory_op = mem.pending_actions.pop(0)
                        self.scheduler.schedule_event(self.scheduler.global_cycles, memory_op)

        self.hw_arch.analysis_done = True
        # Check that all computations have been calculated
        assert sum((comp_block.total_flop_computes // 2) for comp_block in self.hw_arch.matmul_blocks) == self.model.num_macs, "Simulation has not completed all computations! Check possible cause with verbose debug."
        self.hw_arch._simulation_time = time.time() - start_time

    def retrieve_stats(self):
        start_time = time.time()
        # Once analysis is done, the HW arch is populated with runtime stats, which are used to generate action counts for accelergy along with the underlying hw arch description
        # TODO maybe rethink?.. basically to eliminate requerying Accelergy for ART/ERT of same HW across many experiments, we reuse them
        if self.store_to_tmp:
            tmp_dir = tempfile.gettempdir()

            arch_yaml = os.path.join(tmp_dir, f"{self.hw_arch.name}_accelergy_arch.yaml")
            action_yaml = os.path.join(tmp_dir, f"{self.hw_arch.name}_action_counts.yaml")
            accelergy_out_dir = os.path.join(tmp_dir, f"{self.hw_arch.name}_ACCELERGY_OUTPUT")
        else:
            arch_yaml = f"{self.hw_arch.name}_accelergy_arch.yaml"
            action_yaml = f"{self.hw_arch.name}_action_counts.yaml"
            accelergy_out_dir = f"{self.hw_arch.name}_ACCELERGY_OUTPUT"

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

        with open(os.devnull, 'w') as devnull:
            subprocess.run(command, stdout=devnull, stderr=devnull)

        # Fill component area stats with the generated stats
        art = load_yaml(f'{accelergy_out_dir}/ART.yaml')
        for item in art["ART"]["tables"]:
            for key, value in self.hw_arch.comp_names_map.items():
                if key in item['name'] and key != self.hw_arch.dram.name:  # We dont want the area of DRAM since it is offchip
                    value.area += item['area']
                    self.hw_arch.area += item['area']

        # Fill component energy stats with the generated stats
        ert = load_yaml(f'{accelergy_out_dir}/energy_estimation.yaml')            
        for item in ert["energy_estimation"]["components"]:
            for key, value in self.hw_arch.comp_names_map.items():
                if key in item['name']:
                    value.energy += item['energy']
                    self.hw_arch.energy += item['energy']
        self.hw_arch._accelergy_time = time.time() - start_time
    
class StaticSimulationEngine(AbstractSimulationEngine):
    def __init__(self, model, execution_graph, hw_arch, verbose: bool = False, permutation_seed: int = None, scheduling_seed: int = None, deterministic_seed: int = None, deterministic: bool = True, store_to_tmp: bool = False, **kwargs):
        super().__init__(model=model, execution_graph=execution_graph, hw_arch=hw_arch, deterministic_seed=deterministic_seed, deterministic=deterministic, store_to_tmp=store_to_tmp, verbose=verbose, **kwargs)
        if permutation_seed is None:
            permutation_seed = random.randint(0, 2**32 - 1)
        self.permutation_seed = permutation_seed
        self.scheduling_seed = scheduling_seed

        if self.verbose:
            print(f"Permutation seed for Static Simulation Engine set to {self.permutation_seed}")
            mode = "uniform across units" if self.scheduling_seed is None else "random across units"
            print(f"Scheduling seed for Static Simulation Engine set to {self.scheduling_seed} ({mode})\n")
            print(f"Deterministic mode: {self.deterministic}\n")
   
    def generate_permutation(self):
        perm_random = random.Random(self.permutation_seed)
        
        random.seed(self.permutation_seed)
        for node in self.execution_graph.all_nodes:
            lower, upper = node.encoding_range
            node.encoding_value = perm_random.uniform(lower, upper)

        operations = self.execution_graph.all_nodes
        permutation = sorted(operations, key=lambda x: x.encoding_value)
        if self.verbose:
            self.verbose_permutations(permutation)
        return permutation

    def schedule_operations(self, permutation):
        sched_random = random.Random(self.scheduling_seed)

        matmul_arrays = self.hw_arch.matmul_blocks
        if self.scheduling_seed is None:  # Uniform assignment of operations across computational units
            for i, operation in enumerate(permutation):
                assigned_array = matmul_arrays[i % len(matmul_arrays)]
                assigned_array.plan.append(operation)
        else:  # Random assignment of operations across computational units
            for operation in permutation:
                assigned_array = sched_random.choice(matmul_arrays)
                assigned_array.plan.append(operation)

    def initialize_simulation(self):
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


# TODO DYNAMIC SIMULATION ENGINE (WITH RESCHEDULES!.. ALSO THE KEYS SHOULD SOMEHOW ALWAYS PROBABLY GUIDE ALL THE RANDOM SELECTIONS/CHOICES! TO MAKE IT DETERMINISTIC.. OR CHOOSE DETERMINISTIC TRUE??)
class DynamicSimulationEngine(AbstractSimulationEngine):
    pass


# USE PYMOO HERE ? TO IMPLEMENT NSGA-II
class GeneticAlgorithmSimulationEngine(AbstractSimulationEngine):
    pass