from analyzer.core.model_architectures.transformer_blocks.layer import TransformerLayer
from .self_attention import SelfAttention

class MultiHeadSelfAttention(TransformerLayer):
    def __init__(self, name: str = "mhsa", sequence_length: int = 256, embedding_dim: int = 768, num_heads: int = 12, batch_size: int = 1, add_bias: bool = False, **kwargs):
        super().__init__(name, sequence_length, embedding_dim, embedding_dim, batch_size, add_bias, **kwargs)
        assert num_heads > 0
        self.num_heads = num_heads
        # Single-head: MHSA reduces down to a single SelfAttention instance
        if self.num_heads == 1:
            self.self_attention = SelfAttention(f"{name}_head_1", sequence_length, embedding_dim, embedding_dim, batch_size, add_bias, **kwargs)
            self.parameters = self.self_attention.parameters
            self.num_static_parameters = self.self_attention.num_static_parameters
            self.num_macs = self.self_attention.num_macs
            self.plan = self.self_attention.plan
            return

        head_dim = embedding_dim  // num_heads
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        # Init multiple self-attention heads (sub-layers)
        for k in range(1, num_heads+1):
            self.add_layer(SelfAttention(f"{name}_head_{k}", sequence_length, embedding_dim, head_dim, batch_size, add_bias, **kwargs))

        """ STATS """
        # Calculate total parameters by summing parameters of all heads and adding the output projection
        out_proj_parameters = embedding_dim * embedding_dim + embedding_dim if add_bias else embedding_dim * embedding_dim
        self.num_static_parameters = sum(h.num_static_parameters for h in self.layers) + out_proj_parameters
        # Calculate total computations
        self.num_macs = sum(h.num_macs for h in self.layers) + batch_size * sequence_length * embedding_dim * embedding_dim

        """ PARAMETERS AND EXECUTION PLAN """
        # Define parameters and execution plan
        # TODO now we ignore softmax (attention scores are the result of softmax on query ⋅ key) and layer normalizations
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str[:-1]} Heads: {self.num_heads}>"
    
    def define_parameters(self):
        """Aggregate parameters from all heads and include output projection parameters for the multi head self attention layer."""
        static_params = {}
        dynamic_params = {}

        # Aggregate parameters from all heads
        for head in self.layers:
            for key, value in head.parameters['parameters']['static'].items():
                static_params[key] = value
            for key, value in head.parameters['parameters']['dynamic'].items():
                dynamic_params[key] = value
        
        # Add output projection parameters
        static_params.update({
            f"{self.name}_output_projection_weight": {"dimensions": (self.embedding_dim, self.embedding_dim), "total": self.embedding_dim * self.embedding_dim}
        })
        
        if self.add_bias:
            static_params.update({
                f"{self.name}_output_projection_bias": {"dimensions": (1, self.embedding_dim), "total": self.embedding_dim}
            })
        
        dynamic_params.update({
            f"{self.name}_concatenated_output": {"dimensions": (self.sequence_length, self.embedding_dim), "total": self.sequence_length * self.embedding_dim},
            f"{self.name}_output": {"dimensions": (self.sequence_length, self.embedding_dim), "total": self.sequence_length * self.embedding_dim}
        })
        return {"parameters": {"static": static_params, "dynamic": dynamic_params}}

    def define_plan(self):
        """Aggregate execution plans from all heads and include output projection for the multi head self attention layer."""
        head_plans = [head.plan for head in self.layers]
        plan = {
            "type": "sequential",
            "operations": [
                {
                    "type": "parallel",
                    "operations": head_plans
                },
                {
                    "type": "sequential",
                    "operations": [
                        {
                            "name": f"<{self.name}> Concatenation",
                            "computation": f"concat({', '.join([f'{head.name}_output' for head in self.layers])})",
                            "output": f"{self.name}_concatenated_output"
                        },
                        {
                            "name": f"<{self.name}> Output Projection",
                            "computation": f"add(matmul({self.name}_concatenated_output, {self.name}_output_projection_weight), {self.name}_output_projection_bias)" if self.add_bias else f"matmul({self.name}_concatenated_output, {self.name}_output_projection_weight)",
                            "output": f"{self.name}_output"
                        }
                    ]
                }
            ]
        }
        return plan
