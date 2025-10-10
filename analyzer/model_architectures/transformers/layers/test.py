from analyzer.core.model_architectures.transformer_blocks.layer import TransformerLayer

class Test(TransformerLayer):    
    def __init__(self, name: str = "test", sequence_length: int = 256, embedding_dim: int = 768, layer_dim: int = 768, batch_size: int = 1, add_bias: bool = False, parent: object = None, **kwargs):
        super().__init__(name, sequence_length, embedding_dim, layer_dim, batch_size, add_bias, parent, **kwargs)

        """ STATS """
        # Initialize parameters for query
        self.num_static_parameters = 1 * (embedding_dim * layer_dim + layer_dim) if add_bias else 1 * embedding_dim * layer_dim
        # Calculate the number of MACs
        linear_transforms = batch_size * 1 * sequence_length * embedding_dim * layer_dim
        #self_attention_comps = batch_size * 2 * sequence_length * sequence_length * layer_dim
        self.num_macs = linear_transforms #+ self_attention_comps

        """ PARAMETERS AND EXECUTION PLAN """
        # Define parameters and execution plan
        # TODO now we ignore softmax (attention scores are the result of softmax on query ⋅ key) and layer normalizations
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def define_parameters(self):
        """Defines the parameters for the self attention layer."""
        static_params = {
            f"{self.name}_query_weight": {"dimensions": (self.embedding_dim, self.layer_dim), "total": self.embedding_dim * self.layer_dim}
        }

        if self.add_bias:
            static_params.update({
                f"{self.name}_query_bias": {"dimensions": (1, self.layer_dim), "total": self.layer_dim}
            })
            
        dynamic_params = {
            f"input": {"dimensions": (self.sequence_length, self.embedding_dim), "total": self.sequence_length * self.embedding_dim},
            f"{self.name}_query": {"dimensions": (self.sequence_length, self.layer_dim), "total": self.sequence_length * self.layer_dim}
        }
        return {"parameters": {"static": static_params, "dynamic": dynamic_params}}

    def define_plan(self):
        """Defines the execution plan for the self attention layer."""
        plan = {
            "type": "sequential",  # Sequential execution block
            "operations": [
                {
                    "type": "parallel",  # Parallel operations for linear transformations of Q, K, V matrices
                    "operations": [
                        {
                            "name": f"<{self.name}> Query Projection",
                            "computation": f"add(matmul(input, {self.name}_query_weight), {self.name}_query_bias)" if self.add_bias else f"matmul(input, {self.name}_query_weight)",
                            "output": f"{self.name}_query"
                        }
                    ]
                }
            ]
        }
        return plan
