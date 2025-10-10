from analyzer.core.model_architectures.transformer_blocks.layer import TransformerLayer
from .latent_attention import LatentAttentionHead

class MultiHeadLatentAttention(TransformerLayer):
    def __init__(self, name="mla", sequence_length=256, embedding_dim=768, num_heads=12, q_latent_dim=1536, kv_latent_dim=512, batch_size=1, add_bias=False, parent: object = None, **kwargs):
        super().__init__(name, sequence_length, embedding_dim, embedding_dim, batch_size, add_bias, parent, **kwargs)
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.num_heads = num_heads
        self.q_latent_dim = q_latent_dim
        self.kv_latent_dim = kv_latent_dim
        self.head_dim = embedding_dim // num_heads

        # Init multiple latent attention heads (sub-layers)
        for i in range(1, num_heads + 1):
            self.add_layer(LatentAttentionHead(f"{name}_head_{i}", sequence_length, q_latent_dim, kv_latent_dim, self.head_dim, batch_size, add_bias, self, **kwargs))

        """ STATS """
        self.num_static_parameters = sum(h.num_static_parameters for h in self.layers)
        # Final output projection, Q down-projection, KV down-projection, and RoPE key projection
        weight_count = (embedding_dim * embedding_dim) + (embedding_dim * q_latent_dim) + (embedding_dim * kv_latent_dim) + (embedding_dim * self.head_dim)
        bias_count = embedding_dim + q_latent_dim + kv_latent_dim + self.head_dim if self.add_bias else 0
        self.num_static_parameters += (weight_count + bias_count)

        self.num_macs = sum(h.num_macs for h in self.layers)
        # Linear transforms (Q down, KV down, RoPE key down) and final output projection
        down_projections = self.batch_size * self.sequence_length * ((self.embedding_dim * self.q_latent_dim) + (self.embedding_dim * self.kv_latent_dim) + (self.embedding_dim * self.head_dim))
        final_out_projection = self.batch_size * self.sequence_length * self.embedding_dim * self.embedding_dim
        self.num_macs += (down_projections + final_out_projection)

        """ PARAMETERS AND EXECUTION PLAN """
        # TODO now we ignore softmax and rope, todo!
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str[:-1]} Query Latent Dim: {self.q_latent_dim} Key Value Latent Dim: {self.kv_latent_dim} Head Dim: {self.head_dim}>"
    
    def define_parameters(self):
        """Aggregate parameters from all heads and include additional laten projection and output projection parameters for the multi head latent attention layer."""
        static_params = {}
        dynamic_params = {}

        # Aggregate parameters from all heads
        for head in self.layers:
            for key, value in head.parameters['parameters']['static'].items():
                static_params[key] = value
            for key, value in head.parameters['parameters']['dynamic'].items():
                dynamic_params[key] = value

        # Add additional params before/after computing the heads
        static_params.update({
            f"{self.name}_output_projection_weight": {"dimensions": (self.embedding_dim, self.embedding_dim), "total": self.embedding_dim * self.embedding_dim},
            f"{self.name}_query_down_projection_weight": {"dimensions": (self.embedding_dim, self.q_latent_dim), "total": self.embedding_dim * self.q_latent_dim},
            f"{self.name}_key_value_down_projection_weight": {"dimensions": (self.embedding_dim, self.kv_latent_dim), "total": self.embedding_dim * self.kv_latent_dim},
            f"{self.name}_rope_key_weight": {"dimensions": (self.embedding_dim, self.head_dim), "total": self.embedding_dim * self.head_dim}
        })

        if self.add_bias:
            static_params.update({
                f"{self.name}_output_projection_bias": {"dimensions": (1, self.embedding_dim), "total": self.embedding_dim},                
                f"{self.name}_query_down_projection_bias": {"dimensions": (1, self.q_latent_dim), "total": self.q_latent_dim},
                f"{self.name}_key_value_down_projection_bias": {"dimensions": (1, self.kv_latent_dim), "total": self.kv_latent_dim},
                f"{self.name}_rope_key_bias": {"dimensions": (1, self.head_dim), "total": self.head_dim}
            })

        dynamic_params.update({
            f"input": {"dimensions": (self.sequence_length, self.embedding_dim), "total": self.sequence_length * self.embedding_dim},
            f"{self.name}_latent_query": {"dimensions": (self.sequence_length, self.q_latent_dim), "total": self.sequence_length * self.q_latent_dim},
            f"{self.name}_latent_key_value": {"dimensions": (self.sequence_length, self.kv_latent_dim), "total": self.sequence_length * self.kv_latent_dim},
            f"{self.name}_rope_key": {"dimensions": (self.sequence_length, self.head_dim), "total": self.sequence_length * self.head_dim},
            f"{self.name}_concatenated_output": {"dimensions": (self.sequence_length, self.embedding_dim), "total": self.sequence_length * self.head_dim},
            f"{self.name}_output": {"dimensions": (self.sequence_length, self.embedding_dim), "total": self.sequence_length * self.embedding_dim}
        })
        return {"parameters": {"static": static_params, "dynamic": dynamic_params}}

    def define_plan(self):
        """Aggregate execution plans from all heads and include latent and output projections for the multi head latent attention layer."""
        head_plans = [head.plan for head in self.layers]
        
        # TODO Kr PARALLEL TO INSIDE THE HEADS?
        plan = {
            "type": "sequential",
            "operations": [
                {
                    "type": "parallel",
                    "operations": [
                        {
                            "name": f"<{self.name}> Latent Query projection",
                            "computation": f"add(matmul(input, {self.name}_query_down_projection_weight), {self.name}_query_down_projection_bias)" if self.add_bias else f"matmul(input, {self.name}_query_down_projection_weight)",
                            "output": f"{self.name}_latent_query"
                        },
                        {
                            "name": f"<{self.name}> Latent Key Value Projection",
                            "computation": f"add(matmul(input, {self.name}_key_value_down_projection_weight), {self.name}_key_value_down_projection_bias)" if self.add_bias else f"matmul(input, {self.name}_key_value_down_projection_weight)",
                            "output": f"{self.name}_latent_key_value"
                        },
                        {
                            "name": f"<{self.name}> RoPE Key Embeddings",   # TODO WE ignore ROPE COST and modelling now!
                            "computation": f"add(matmul(input, {self.name}_rope_key_weight), {self.name}_rope_key_bias)" if self.add_bias else f"matmul(input, {self.name}_rope_key_weight)",
                            "output": f"{self.name}_rope_key"
                        }
                    ]
                },
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