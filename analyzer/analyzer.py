from graphviz import Digraph
import yaml
import re
from analyzer.core.simulator.simulator import StaticSimulationEngine
import subprocess 
import os


""" Classes for building up the execution tree from model under test's execution plan """
class ExecutionGraph:
    def __init__(self):
        self.root = None
        self.leaves = []
        self.all_nodes = []
        self.max_depth = 0
        self.tensors_needed = {}


class ExecutionNode:
    def __init__(self, name, computation=None, batch_size=None, data_bitwidth=None, output=None, input_data=None, output_data=None, depth=0):
        self.name = name
        self.computation = computation
        self.batch_size = batch_size
        self.data_bitwidth = data_bitwidth
        self.output = output
        self.children = []
        self.depth = depth
        self.encoding_range = (None, None)  # Used for encoding the range for valid scheduling of the operation in time
        self.encoding_value = None          # Encoding value for valid scheduling is added during analysis itself
        self.input_data = input_data if input_data is not None else {}
        self.output_data = output_data if output_data is not None else {}
        self._is_root = False
        self._is_done = False
        self._pending_event = None  # Used in simulation analysis to reschedule the node operation if it was waiting for its dependent (children) nodes to finish
        self.parents = []

    def __str__(self):
        return (f"<ExecutionNode name={self.name}, encoding_value={self.encoding_value}, "
                f"encoding_range={self.encoding_range}, next_operations={self.parents}, "
                f"dependencies={[child.name for child in self.children]}, computation={self.computation}, batches={self.batch_size}, data_bitwidth={self.data_bitwidth}, "
                f"output={self.output}, input_data={self.input_data}, "
                f"input_data={self.input_data}, output_data={self.output_data}, depth={self.depth}>")

    def add_child(self, node):
        self.children.append(node)
        node.parents.append(self)
        if node.depth <= self.depth:
            node.depth = self.depth + 1

    # Simulation analysis properties and method
    @property
    def is_done(self):
        return self._is_done

    @is_done.setter
    def is_done(self, value: bool):
        self._is_done = value
        
    @property
    def pending_event(self):
        return self._pending_event

    @pending_event.setter
    def pending_event(self, event):
        self._pending_event = event

    def remove_dependency(self):
        """Removes dependency of self node from the parents' children list."""
        if self.parents:
            for parent in self.parents:
                if self in parent.children:
                    parent.children.remove(self)


""" Class for analyzing given model's inference performance in hardware """
class Analyzer:
    def __init__(self, model, hw_arch, data_bitwidth: int):
        self.model = model
        self.hw_arch = hw_arch
        self.data_bitwidth = data_bitwidth  # TODO Uniform for simplicity.. could be tied to individual data tensors for non-uniform analysis

        # TODO!!! FOR SUPPORT OF NONUNIFORM BITWIDTH ADJUST.. (at least one spatial array should be able to compute the bw..)
        # Assert that the datawidth of each MatmulArray is at least as large as the data_bitwidth
        for matmul_block in hw_arch.matmul_blocks:
            assert matmul_block.data_bitwidth >= self.data_bitwidth, f"Matmul array {matmul_block.name} has datawidth {matmul_block.data_bitwidth} which is less than the model data bitwidth {self.data_bitwidth}!"

        # Assert that the wordsize of each MatmulArray is at least as large as the data_bitwidth
        for memory_block in hw_arch.memory_blocks + [hw_arch.dram]:
            assert memory_block.word_size >= self.data_bitwidth, f"Memory block {memory_block.name} has word size {memory_block.word_size} which is less than the model data bitwidth {self.data_bitwidth}!"

        # WARNING if any memory element's buswidth is lower than the data_bitwidth
        for memory_block in hw_arch.memory_blocks + [hw_arch.dram]:
            if memory_block.bus_bitwidth < self.data_bitwidth:
                print(f"WARNING: Memory block {memory_block.name} has bus_bitwidth {memory_block.bus_bitwidth} which is less than the model data bitwidth {self.data_bitwidth}, resulting in inneficient memory accesses.")

        self.graph = ExecutionGraph()
        self.req_inputs_to_nodes = {}  # Maps the required input variables to a list of nodes that require them (input variable is the output of some other node further down the graph); used for proper node connections
        self.build_execution_graph(self.model.plan, self.model.batch_size)  # Execution graph used for simulation analysis
        self.assign_encoding_ranges()

    def __str__(self) -> str:
        return (f"<Inference Analyzer \n\n"
                f" Model: {self.model} \n\n"
                f" Accelerator: {self.hw_arch.__str__()[:-1]} >\n\n>")

    def assign_encoding_ranges(self):
        for node in self.graph.all_nodes:
            if not node.children:
                node.encoding_range = (0.0, 1.0 - ((node.depth-1) / self.graph.max_depth)) if self.graph.max_depth != 0 else (0.0, 1.0)
            else:
                if node == self.graph.root:
                    node.encoding_range = (1.0 - (node.depth / self.graph.max_depth), 1.0) if self.graph.max_depth != 0 else (0.0, 1.0)
                else:
                    node.encoding_range = (1.0 - (node.depth / self.graph.max_depth), 1.0 - ((node.depth-1) / self.graph.max_depth)) if self.graph.max_depth != 0 else (0.0, 1.0)

    def build_execution_graph(self, plan, batch_size):
        if 'type' in plan:  # Sequential or parallel block
            for operation in reversed(plan['operations']):
                self.build_execution_graph(operation, batch_size)
        else:  # Actual computation operation
            computation = plan.get('computation')
            inputs = self.extract_inputs(computation)
            input_data = {input_var: self.get_parameter_data(input_var) for input_var in inputs}
            output = plan.get('output')
            output_data = {output: self.get_parameter_data(output)} if output else {}

            node = ExecutionNode(plan.get('name'), computation, batch_size, self.data_bitwidth, output, input_data, output_data)
            self.graph.leaves.append(node)  # Assume new node is a leaf initially
            self.graph.all_nodes.append(node)  # Add node to all_nodes list
            
            for i in inputs:
                if i not in self.graph.tensors_needed:
                    self.graph.tensors_needed[i] = 0
                self.graph.tensors_needed[i] += 1

            if node.output in self.req_inputs_to_nodes.keys():  # If the output is an input to some other node, connect them
                for dependent_node in self.req_inputs_to_nodes[node.output]:
                    if node not in dependent_node.children:
                        dependent_node.add_child(node)
                        if dependent_node in self.graph.leaves:
                            self.graph.leaves.remove(dependent_node)  # The parent node is not a leaf anymore (there is data dependency)

            # Iterate over the node's input variables and update the req_inputs_to_nodes dictionary to be then used for proper connection to the node whose output is one of the required inputs (data-dependency)
            for input_var in inputs:
                if input_var not in self.req_inputs_to_nodes:
                    self.req_inputs_to_nodes[input_var] = []
                self.req_inputs_to_nodes[input_var].append(node)

            self.graph.max_depth = max(self.graph.max_depth, node.depth)
            
            if self.graph.root is None:  # The root node is the final operation in the execution plan
                self.graph.tensors_needed[output] = 1
                self.graph.root = node
                self.graph.root._is_root = True

    def get_parameter_data(self, param_name):
        """ Helper method to extract parameter data (dimensions and type). """
        params = self.model.parameters.get('parameters', {})
        for param_type, param_dict in params.items():
            if param_name in param_dict:
                return {
                    'dimensions': param_dict[param_name]['dimensions'],
                    'type': param_type
                }
        return {
            'dimensions': 'Unknown',
            'type': 'Unknown'
        }

    def extract_inputs(self, computation_str):
        """ Extract potential input variable names from the computation string. """
        # NOTE add more then just matmul and add if required or... add inputs list straight to plans within layers/models...
        ignored_words = {'matmul', 'add', 'concat'}
        words = re.findall(r'[a-z_0-9]+', computation_str)
        # Filter out ignored words, preserving the original order (IMPORTANT! to preserve the original matrix-matrix computation order)
        inputs = [i for i in words if i not in ignored_words]
        return inputs

    def visualize_graph(self, filename: str='execution_graph'):
        dot = Digraph(comment='Execution Plan', node_attr={'shape': 'ellipse', 'fontsize': '12', 'height': '1'})
        dot.attr(rankdir='BT')
        self._build_graph(dot, self.graph.root)
        dot.render(filename, format='pdf')

    def _build_graph(self, graph, node, added_nodes=None):
        if added_nodes is None:  # Set to avoid duplication of nodes and edges
            added_nodes = set()

        if node.name not in added_nodes:
            label = (f"{node.name}\n"
                    f"Batches: {node.batch_size}\n"
                    f"Computation: {node.computation if node.computation else 'No computation'}\n"
                    f"Output: {node.output if node.output else 'No output'}\n"
                    f"Input data: {node.input_data}\n"
                    f"Output data: {node.output_data}\n"
                    f"Encoding range: {node.encoding_range}")
            graph.node(node.name, label=label)
            added_nodes.add(node.name)  # Add node name to set to avoid duplication
        
            for child in node.children:
                graph.edge(child.name, node.name)            
                self._build_graph(graph, child, added_nodes)


    def run_simulation_analysis(self, verbose: bool = False, engine_type='static', **kwargs):
        """Method for executing simulation analysis of the execution graph on the hardware accelerator.

        This method configures and executes a simulation of the accelerator's performance. The simulation can be tailored
        using different engine types (e.g., static simulation, genetic algorithm).
        The runtime results of the simulation are used to generate area and energy estimation reports via external plug-ins (e.g. Accelergy).

        Notes:
            - The simulation results populate runtime statistics (number of compute/idle cycles and memory transfer actions)
              in the hardware architecture, which are used to generate energy/area reports.

        Args:
            verbose (bool): If True, prints detailed simulation progress and results. Defaults to False.
            engine_type (str): The type of simulation engine to use. Options are 'static' or 'genetic'. Defaults to 'static'.
        """
        if engine_type == 'static':
            engine = StaticSimulationEngine(model=self.model, execution_graph=self.graph, hw_arch=self.hw_arch, verbose=verbose, **kwargs)
        elif engine_type == 'genetic':  # TODO use PYMOO or DEAP for genetic algorithm
            raise NotImplementedError("Simulation Engine for Genetic Algorithm not yet implemented")
            #engine = GeneticAlgorithmSimulationEngine(model=self.model, execution_graph=self.graph, hw_arch=self.hw_arch, verbose=verbose, initiation_seed=initiation_seed, scheduling_seed=scheduling_seed)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        engine.initialize_simulation()

        # Once analysis is done, the HW arch is populated with runtime stats, which are used to generate action counts for accelergy along with the underlying hw arch description
        self.hw_arch.get_accelergy_description(out_fname="accelergy_arch")
        self.hw_arch.generate_action_counts(out_fname="action_counts")

        command = [
            "accelergy",
            "accelergy_arch.yaml",
            "action_counts.yaml",
            "compound_components.yaml",
            "-o", "ACCELERGY_OUTPUT"
        ]
        
        with open(os.devnull, 'w') as devnull:
            subprocess.run(command, stdout=devnull, stderr=devnull)

        # Fill component area stats with the generated stats
        art = load_yaml('ACCELERGY_OUTPUT/ART.yaml')
        for item in art["ART"]["tables"]:
            for key, value in self.hw_arch.comp_names_map.items():
                if key in item['name'] and key != self.hw_arch.dram.name:  # We dont want the area of DRAM since it is offchip
                    value.area += item['area']
                    self.hw_arch.area += item['area']

        # Fill component energy stats with the generated stats
        ert = load_yaml('ACCELERGY_OUTPUT/energy_estimation.yaml')            
        for item in ert["energy_estimation"]["components"]:
            for key, value in self.hw_arch.comp_names_map.items():
                if key in item['name']:
                    value.energy += item['energy']
                    self.hw_arch.energy += item['energy']

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
