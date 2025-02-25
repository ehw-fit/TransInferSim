from analyzer.core.model_architectures.transformer_blocks.layer import TransformerLayer


class FeedForwardNetwork(TransformerLayer):
    def __init__(self, name: str = "ffn", sequence_length: int = 256, embedding_dim: int = 768, layer_dim: int = 3072, batch_size: int = 1, add_bias: bool = False, **kwargs):
        super().__init__(name, sequence_length, embedding_dim, layer_dim, batch_size, add_bias, **kwargs)

        """ STATS """
        # Two linear transformations: embedding_dim -> layer_dim -> embedding_dim
        self.num_static_parameters = (embedding_dim * layer_dim + layer_dim) + (layer_dim * embedding_dim + embedding_dim) if add_bias else (embedding_dim * layer_dim) + (layer_dim * embedding_dim)
        # Calculate the number of MACs (first and second non-linear transformations)
        self.num_macs = 2 * (batch_size * sequence_length * embedding_dim * layer_dim)

        """ PARAMETERS AND EXECUTION PLAN """
        # Define parameters and execution plan
        # TODO now we ignore softmax (attention scores are the result of softmax on query ⋅ key) and layer normalizations
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()


    def define_parameters(self):
        """Defines the parameters for the feed forward network layer."""
        static_params = {
            f"{self.name}_first_transform_weight": {"dimensions": (self.layer_dim, self.embedding_dim), "total": self.layer_dim * self.embedding_dim},
            f"{self.name}_second_transform_weight": {"dimensions": (self.embedding_dim, self.layer_dim), "total": self.layer_dim * self.embedding_dim},
        }

        if self.add_bias:
            static_params.update({
                f"{self.name}_first_transform_bias": {"dimensions": (1, self.layer_dim), "total": self.layer_dim},
                f"{self.name}_second_transform_bias": {"dimensions": (1, self.embedding_dim), "total": self.embedding_dim}
            })

        dynamic_params = {
            f"{self.name}_input": {"dimensions": (self.sequence_length, self.embedding_dim), "total": self.sequence_length * self.embedding_dim},
            f"{self.name}_first_transform": {"dimensions": (self.sequence_length, self.layer_dim), "total": self.sequence_length * self.layer_dim},
            f"{self.name}_output": {"dimensions": (self.sequence_length, self.embedding_dim), "total": self.sequence_length * self.embedding_dim}
        }
        return {"parameters": {"static": static_params, "dynamic": dynamic_params}}

    def define_plan(self):
        """Defines the execution plan for the feed forward network layer."""
        
        plan = {
            "type": "sequential",  # Sequential execution block for the linear projectiosn within feed forward network
            "operations": [
                {
                    "name": f"<{self.name}> First Linear Transformation",
                    "computation": f"add(matmul({self.name}_input, {self.name}_first_transform_weight), {self.name}_first_transform_bias)" if self.add_bias else f"matmul({self.name}_input, {self.name}_first_transform_weight)",
                    "output": f"{self.name}_first_transform"
                },
                {
                    "name": f"<{self.name}> Second Linear Transformation",
                    "computation": f"add(matmul({self.name}_first_transform, {self.name}_second_transform_weight), {self.name}_second_transform_bias)" if self.add_bias else f"matmul({self.name}_first_transform, {self.name}_second_transform_weight)",
                    "output": f"{self.name}_output"
                }
            ]
        }
        return plan
