import heapq
import random
from abc import ABC, abstractmethod
from .events import MatmulStartEvent

class EventScheduler:
    def __init__(self, execution_graph, verbose: bool = False):
        self.events = []
        self.global_cycles = 0
        self.verbose = verbose
        self.execution_graph = execution_graph

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
    def __init__(self, model, execution_graph, hw_arch, verbose: bool = False, **kwargs):
        self.model = model
        self.execution_graph = execution_graph
        self.hw_arch = hw_arch
        self.scheduler = EventScheduler(execution_graph, verbose=verbose)
        self.verbose = verbose

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
            event.execute(self.scheduler)
            if self.scheduler.events == []:  # If at the end there are still some pending memory read/write actions, reschedule them
                for mem in self.hw_arch.memory_blocks:
                    while mem.pending_actions:
                        memory_op = mem.pending_actions.pop(0)
                        self.scheduler.schedule_event(self.scheduler.global_cycles, memory_op)

        self.hw_arch.analysis_done = True
        # Check that all computations have been calculated
        assert sum((comp_block.total_flop_computes // 2) for comp_block in self.hw_arch.matmul_blocks) == self.model.num_macs, "Simulation has not completed all computations! Check possible cause with verbose debug."


class StaticSimulationEngine(AbstractSimulationEngine):
    def __init__(self, model, execution_graph, hw_arch, verbose: bool = False, permutation_seed: int = None, scheduling_seed: int = None,  **kwargs):
        super().__init__(model=model, execution_graph=execution_graph, hw_arch=hw_arch, verbose=verbose, **kwargs)
        if permutation_seed is None:
            permutation_seed = random.randint(0, 2**32 - 1)
        self.permutation_seed = permutation_seed
        self.scheduling_seed = scheduling_seed

        if self.verbose:
            print(f"Permutation seed for Static Simulation Engine set to {self.permutation_seed}")
            mode = "uniform across units" if self.scheduling_seed is None else "random across units"
            print(f"Scheduling seed for Static Simulation Engine set to {self.scheduling_seed} ({mode})\n")
   
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


# USE PYMOO HERE ? TO IMPLEMENT NSGA-II
class GeneticAlgorithmSimulationEngine(AbstractSimulationEngine):
    pass