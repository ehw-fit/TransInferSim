from analyzer.core.model_architectures.transformer_blocks.layer import TransformerLayer

class SelfAttention(TransformerLayer):    
    def __init__(self, name: str = "sa", sequence_length: int = 256, embedding_dim: int = 768, layer_dim: int = 768, batch_size: int = 1, add_bias: bool = False, **kwargs):
        super().__init__(name, sequence_length, embedding_dim, layer_dim, batch_size, add_bias, **kwargs)

        """ STATS """
        # Initialize parameters for query, key, value matrices
        self.num_static_parameters = 3 * (embedding_dim * layer_dim + layer_dim) if add_bias else 3 * embedding_dim * layer_dim
        # Calculate the number of MACs
        linear_transforms = batch_size * 3 * sequence_length * embedding_dim * layer_dim
        self_attention_comps = batch_size * 2 * sequence_length * sequence_length * layer_dim
        self.num_macs = linear_transforms + self_attention_comps

        """ PARAMETERS AND EXECUTION PLAN """
        # Define parameters and execution plan
        # TODO now we ignore softmax (attention scores are the result of softmax on query ⋅ key) and layer normalizations
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def define_parameters(self):
        """Defines the parameters for the self attention layer."""
        static_params = {
            f"{self.name}_query_weight": {"dimensions": (self.layer_dim, self.embedding_dim), "total": self.layer_dim * self.embedding_dim},
            f"{self.name}_key_weight": {"dimensions": (self.layer_dim, self.embedding_dim), "total": self.layer_dim * self.embedding_dim},
            f"{self.name}_value_weight": {"dimensions": (self.layer_dim, self.embedding_dim), "total": self.layer_dim * self.embedding_dim}
        }

        if self.add_bias:
            static_params.update({
                f"{self.name}_query_bias": {"dimensions": (1, self.layer_dim), "total": self.layer_dim},
                f"{self.name}_key_bias": {"dimensions": (1, self.layer_dim), "total": self.layer_dim},
                f"{self.name}_value_bias": {"dimensions": (1, self.layer_dim), "total": self.layer_dim}
            })
            
        dynamic_params = {
            f"input": {"dimensions": (self.sequence_length, self.embedding_dim), "total": self.sequence_length * self.embedding_dim},
            f"{self.name}_query": {"dimensions": (self.sequence_length, self.layer_dim), "total": self.sequence_length * self.layer_dim},
            f"{self.name}_key": {"dimensions": (self.sequence_length, self.layer_dim), "total": self.sequence_length * self.layer_dim},
            f"{self.name}_value": {"dimensions": (self.sequence_length, self.layer_dim), "total": self.sequence_length * self.layer_dim},
            f"{self.name}_attention_score": {"dimensions": (self.sequence_length, self.sequence_length), "total": self.sequence_length * self.sequence_length},
            f"{self.name}_output": {"dimensions": (self.sequence_length, self.layer_dim), "total": self.sequence_length * self.layer_dim}
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
                        },
                        {
                            "name": f"<{self.name}> Key Projection",
                            "computation": f"add(matmul(input, {self.name}_key_weight), {self.name}_key_bias)" if self.add_bias else f"matmul(input, {self.name}_key_weight)",
                            "output": f"{self.name}_key"
                        },
                        {
                            "name": f"<{self.name}> Value Projection",
                            "computation": f"add(matmul(input, {self.name}_value_weight), {self.name}_value_bias)" if self.add_bias else f"matmul(input, {self.name}_value_weight)",
                            "output": f"{self.name}_value"
                        }
                    ]
                },
                {
                    "type": "sequential",  # Sequential execution block for attention computations
                    "operations": [
                        {
                            "name": f"<{self.name}> Attention Score Calculation",
                            "computation": f"matmul({self.name}_query, {self.name}_key)",
                            "output": f"{self.name}_attention_score"
                        },
                        {
                            "name": f"<{self.name}> Output Computation",
                            "computation": f"matmul({self.name}_attention_score, {self.name}_value)",
                            "output": f"{self.name}_output"
                        }
                    ]
                }
            ]
        }
        return plan
