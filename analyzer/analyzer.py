from graphviz import Digraph
import re
from analyzer.core.simulator.simulator import StaticSimulationEngine, GeneticAlgorithmSimulationEngine
from analyzer.utils.utils import best_factors, compute_tile_bounds, TensorsNeededTracker, safe_label


""" Classes for building up the execution tree from model under test's execution plan """
class ExecutionGraph:
    def __init__(self):
        self.root = None
        self.leaves = []
        self.all_nodes = []
        self.max_depth = 0
        self.tensors_needed = TensorsNeededTracker()


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
                f"output={self.output}, input_data={self.input_data}, output_data={self.output_data}, depth={self.depth}>")

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
    def __init__(self, model, hw_arch, data_bitwidth: int, num_subops: int = 1):
        assert num_subops > 0, "Number of sub-operations must be greater than 0!"
        self.model = model
        self.hw_arch = hw_arch
        self.engine = None
        self.data_bitwidth = data_bitwidth  # TODO Uniform for simplicity.. could be tied to individual data tensors for non-uniform analysis
        self.num_subops = num_subops  # TODO make automated.. Number of sub operations to split each operation into (for parallel execution)
        self._params = self.model.parameters.get('parameters', {})
        # TODO ... HERE.. STATIC CHECKS FOR MEMORY SIZES IF THEY CAN HOLD THE DATA!!!

        # Assert that the wordsize of each MatmulArray is at least as large as the data_bitwidth
        for memory_block in hw_arch.memory_blocks + [hw_arch.dram]:
            assert memory_block.word_size >= self.data_bitwidth, f"Memory block {memory_block.name} has word size {memory_block.word_size} which is less than the model data bitwidth {self.data_bitwidth}!"

        # WARNING if any memory element's buswidth is lower than the data_bitwidth
        for memory_block in hw_arch.memory_blocks + [hw_arch.dram]:
            if memory_block.bus_bitwidth < self.data_bitwidth:
                print(f"WARNING: Memory block {memory_block.name} has bus_bitwidth {memory_block.bus_bitwidth} which is less than the model data bitwidth {self.data_bitwidth}, resulting in inneficient memory accesses.")
        
        # TODO future!!! FOR SUPPORT OF NONUNIFORM BITWIDTH ADJUST.. (at least one spatial array should be able to compute the bw..)
        # Assert that the datawidth of each MatmulArray is at least as large as the data_bitwidth
        for matmul_block in hw_arch.matmul_blocks:
            assert matmul_block.data_bitwidth >= self.data_bitwidth, f"Matmul array {matmul_block.name} has datawidth {matmul_block.data_bitwidth} which is less than the model data bitwidth {self.data_bitwidth}!"
            # TODO CHECK AUTO INTERCONNECTION FOR COMPLEX ARCHITECTURES
            if matmul_block.parent_component.auto_interconnect is True and matmul_block._auto_interconnect_set is False:
                matmul_block.find_and_assign_memories()

        # TODO add static check for each on chip's memory capacity to hold each tile's data (in and out)
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
            # TODO now naive global op splitting (only if subops > 1, otherwise same as original)
            computation = plan.get('computation')
            top_level_op = self.extract_top_operation(computation)
            inputs = self.extract_inputs(computation)
            output = plan.get('output')

            if self.num_subops > 1 and top_level_op != "concat":  # Add concatenation operation for the sub operation outputs
                subop_outputs = {f"{output}_subop_{i}": self.get_parameter_data(output, "out", i) for i in range(self.num_subops)}  # TODO
                comp_op = f"concat({', '.join(subop_outputs)})"
                out_data = {output: self.get_parameter_data(output, "concat")} if output else {}
                node = ExecutionNode(f"{plan.get('name')} Outputs Concatenation", comp_op, batch_size, self.data_bitwidth, output, subop_outputs, out_data)
                self.graph.leaves.append(node)  # Assume new node is a leaf initially
                self.graph.all_nodes.append(node)  # Add node to all_nodes list
                self.graph.max_depth = max(self.graph.max_depth, node.depth)

                if node.output in self.req_inputs_to_nodes.keys():  # If the output is an input to some other node, connect them
                    for dependent_node in self.req_inputs_to_nodes[node.output]:
                        if node not in dependent_node.children:
                            dependent_node.add_child(node)
                            if dependent_node in self.graph.leaves:
                                self.graph.leaves.remove(dependent_node)  # The parent node is not a leaf anymore (there is data dependency)

                # Iterate over the node's input variables and update the req_inputs_to_nodes dictionary to be then used for proper connection to the node whose output is one of the required inputs (data-dependency)
                for input_var in subop_outputs:
                    if input_var not in self.req_inputs_to_nodes:
                        self.req_inputs_to_nodes[input_var] = []
                    self.req_inputs_to_nodes[input_var].append(node)
                    self.graph.tensors_needed.increase_count(input_var)

                if self.graph.root is None:  # The root node is the final operation in the execution plan
                    self.graph.tensors_needed.increase_count(output)
                    self.graph.root = node
                    self.graph.root._is_root = True
            
            if top_level_op == "concat":
                input_data = {input_var: self.get_parameter_data(input_var, "concat") for input_var in inputs}
                output_data = {output: self.get_parameter_data(output, "concat")} if output else {}

                node = ExecutionNode(plan.get('name'), computation, batch_size, self.data_bitwidth, output, input_data, output_data)
                self.graph.leaves.append(node)  # Assume new node is a leaf initially
                self.graph.all_nodes.append(node)  # Add node to all_nodes list

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
                    self.graph.tensors_needed.increase_count(input_var)

                self.graph.max_depth = max(self.graph.max_depth, node.depth)
                
                if self.graph.root is None:  # The root node is the final operation in the execution plan
                    self.graph.tensors_needed.increase_count(output)
                    self.graph.root = node
                    self.graph.root._is_root = True
            else:
                for subop_id in range(self.num_subops):
                    # TODO WHAT ABOUT THE CONCAT!!!!!
                    transpose_b = self.matrix_b_needs_transpose(inputs[0], inputs[1])
                    input_data = {input_var: self.get_parameter_data(input_var, f"in_{chr(97 + i)}", subop_id, transpose_b) for i, input_var in enumerate(inputs)}
                    out_name = f"{output}_subop_{subop_id}" if self.num_subops > 1 else output
                    output_data = {out_name: self.get_parameter_data(output, "out", subop_id)} if output else {}
                    op_name = f"{plan.get('name')}_subop_{subop_id}" if self.num_subops > 1 else plan.get('name')

                    node = ExecutionNode(op_name, computation, batch_size, self.data_bitwidth, out_name, input_data, output_data)
                    self.graph.leaves.append(node)  # Assume new node is a leaf initially
                    self.graph.all_nodes.append(node)  # Add node to all_nodes list

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
                        self.graph.tensors_needed.increase_count(input_var)

                    self.graph.max_depth = max(self.graph.max_depth, node.depth)
                    
                    if self.graph.root is None:  # The root node is the final operation in the execution plan
                        self.graph.tensors_needed.increase_count(output)
                        self.graph.root = node
                        self.graph.root._is_root = True

    

    def matrix_b_needs_transpose(self, input_a: str, input_b: str) -> bool:
        """Determines if the second matrix (input_b) needs transposing for matmul."""
        dims_a = None
        dims_b = None

        # Search for dimensions in the model
        for param_dict in self._params.values():
            if input_a in param_dict:
                dims_a = param_dict[input_a]['dimensions']
            if input_b in param_dict:
                dims_b = param_dict[input_b]['dimensions']

        if not dims_a or not dims_b:
            raise ValueError("Missing dimensions for matmul input(s)")

        # Classic matmul: A (M x K), B (K x N)
        if dims_a[1] == dims_b[0]:
            return False  # No transpose needed
        elif dims_a[1] == dims_b[1]:
            return True   # Transpose B to (N x K)
        else:
            raise ValueError(f"Incompatible dimensions for matmul: {dims_a} x {dims_b}")

    def get_parameter_data(self, param_name: str, type: str, subop_id: int = 1, transpose_b: bool = False):
        # TODO prob enum.. now type is: out, in_a, in_b, concat
        """ Helper method to extract parameter data (dimensions and type). """
        for param_type, param_dict in self._params.items():
            if param_name in param_dict:
                dim_x, dim_y = param_dict[param_name]['dimensions']
                if transpose_b and type == "in_b":
                    dim_y, dim_x = dim_x, dim_y  # Swap dimensions for transpose

                tiles_x, tiles_y = best_factors(self.num_subops)
                row = subop_id // tiles_y
                col = subop_id % tiles_y

                offset_x, end_x = compute_tile_bounds(dim_x, tiles_x, row)
                offset_y, end_y = compute_tile_bounds(dim_y, tiles_y, col)

                tile_shape = (end_x - offset_x, end_y - offset_y)
                    
                if type == "concat":  # Reserved for concatenation operation
                    return {
                        'dimensions': (dim_x, dim_y),
                        'tile_shape': (dim_x, dim_y),
                        'offset': (0, 0),
                        'data_category': param_type
                    }
                elif type == "in_a":  # Tile only along rows (first dim)
                    return {
                        'dimensions': (dim_x, dim_y),
                        'tile_shape': (tile_shape[0], dim_y),
                        'offset': (offset_x, 0),
                        'data_category': param_type
                    }
                elif type == "in_b":  # Tile only along columns (second dim)
                    return {
                        'dimensions': (dim_x, dim_y) if not transpose_b else (dim_y, dim_x),
                        'tile_shape': (dim_x, tile_shape[1]) if not transpose_b else (tile_shape[1], dim_x),
                        'offset': (0, offset_y) if not transpose_b else (offset_y, 0),
                        'data_category': param_type
                    }
                else:  # For subtiled output
                    return {
                        'dimensions': tile_shape,
                        'tile_shape': tile_shape,
                        'offset': (0, 0),
                        'data_category': param_type
                    }
        return {
            'dimensions': 'Unknown',
            'tile_shape': 'Unknown',
            'offset': 'Unknown',
            'data_category': 'Unknown'
        }

    def extract_top_operation(self, computation_str):  # TODO make more robust later with recursive support for parsing nested ops (like softmax(matmul(x, y), z))
        """
        Extracts the top-level operation from a computation string.
        e.g., "add(matmul(x, y), z)" → "add"
        """
        computation_str = computation_str.strip()
        match = re.match(r'([a-z_][a-z_0-9]*)\s*\(', computation_str)
        return match.group(1) if match else None

    def extract_inputs(self, computation_str):
        """ Extract potential input variable names from the computation string. """
        # NOTE add more then just matmul and add if required or... add inputs list straight to plans within layers/models...
        ignored_words = {'matmul', 'add', 'concat'}
        words = re.findall(r'[a-z_0-9]+', computation_str)
        # Filter out ignored words, preserving the original order (IMPORTANT! to preserve the original matrix-matrix computation order)
        inputs = [i for i in words if i not in ignored_words]
        return inputs

    def _build_graph(self, graph, node, added_nodes=None):
        if added_nodes is None:  # Set to avoid duplication of nodes and edges
            added_nodes = set()

        if node.name not in added_nodes:
            label = (f"{safe_label(node.name)}\n"
                    f"Batches: {node.batch_size}\n"
                    f"Computation: {safe_label(node.computation) if node.computation else 'No computation'}\n"
                    f"Output: {safe_label(node.output) if node.output else 'No output'}\n"
                    f"Input data: {safe_label(node.input_data)}\n"
                    f"Output data: {safe_label(node.output_data)}\n"
                    f"Encoding range: {node.encoding_range}")
            graph.node(node.name, label=label)
            added_nodes.add(node.name)  # Add node name to set to avoid duplication
        
            for child in node.children:
                graph.edge(child.name, node.name)            
                self._build_graph(graph, child, added_nodes)

    def visualize_graph(self, filename: str='execution_graph'):
        dot = Digraph(comment='Execution Plan', node_attr={'shape': 'ellipse', 'fontsize': '12', 'height': '1'})
        dot.attr(rankdir='BT')
        self._build_graph(dot, self.graph.root)
        dot.render(filename, format='pdf')

    def run_simulation_analysis(self, verbose: bool = False, engine_type='static', deterministic=True, store_to_tmp=False, **kwargs):
        """Method for executing simulation analysis of the execution graph on the hardware accelerator.

        This method configures and executes a simulation of the accelerator's performance. The simulation can be tailored
        using different engine types (e.g., static simulation, genetic algorithm).
        The runtime results of the simulation are used to generate area and energy estimation reports via external plug-ins (e.g. Accelergy).

        Notes:
            - The simulation results populate runtime statistics (number of compute/idle cycles and memory transfer actions)
              in the hardware architecture, which are used to generate energy/area reports.

        Args:
            verbose (bool): If True, prints detailed simulation progress and results. Defaults to False.
            engine_type (str): The type of simulation engine to use. Options are 'static' or 'dynamic'. Defaults to 'static'.
            deterministic (bool): Decides whether all random choices (i.e. choose from many viable options for victim) will be controlled by the permutation seed or just random. Defaults to True.
            store_to_tmp (bool): Decides whether to store Accelergy outputs to tmp or in local directory.
        """
        if engine_type == 'static':
            self.engine = StaticSimulationEngine(model=self.model, execution_graph=self.graph, hw_arch=self.hw_arch, verbose=verbose, deterministic=deterministic, store_to_tmp=store_to_tmp, **kwargs)
        elif engine_type == 'dynamic':  # TODO dynamically reschedule operations during execution in-between spatial arrays...
            raise NotImplementedError("Simulation Engine for Dynamic Scheduler not yet implemented")
            #engine = GeneticAlgorithmSimulationEngine(model=self.model, execution_graph=self.graph, hw_arch=self.hw_arch, verbose=verbose, initiation_seed=initiation_seed, scheduling_seed=scheduling_seed)
        elif engine_type == 'genetic':  # TODO use PYMOO or DEAP for genetic algorithm
            raise NotImplementedError("Simulation Engine for Genetic Algorithm not yet implemented")
            #engine = GeneticAlgorithmSimulationEngine(model=self.model, execution_graph=self.graph, hw_arch=self.hw_arch, verbose=verbose, initiation_seed=initiation_seed, scheduling_seed=scheduling_seed)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
        self.engine.initialize_simulation()

    def reset_graph(self):
        self.graph = ExecutionGraph()
        self.req_inputs_to_nodes = {}
        self.build_execution_graph(self.model.plan, self.model.batch_size)
        self.assign_encoding_ranges()

    def reset(self):
        self.hw_arch.reset_stats()
        self.reset_graph()
        self.engine.reset(self.graph)
