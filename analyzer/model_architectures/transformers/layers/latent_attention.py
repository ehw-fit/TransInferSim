from analyzer.core.model_architectures.transformer_blocks.layer import TransformerLayer

class LatentAttentionHead(TransformerLayer):
    def __init__(self, name: str = "la", sequence_length: int = 256, q_latent_dim: int = 1536, kv_latent_dim: int = 512, layer_dim: int = 64, batch_size: int = 1, add_bias: bool = False, parent: object = None, **kwargs):
        assert parent is not None, f"{self.__class__.__name__} cannot be instantiated as a top-level component. It must have a MultiHeadLatentAttention parent object."
        super().__init__(name, sequence_length, q_latent_dim, layer_dim, batch_size, add_bias, parent, **kwargs)
        self.q_latent_dim = q_latent_dim
        self.kv_latent_dim = kv_latent_dim

        """ STATS """
        weight_count = 2 * (layer_dim * q_latent_dim) + 2 * (layer_dim * kv_latent_dim)
        bias_count = 4 * layer_dim if add_bias else 0
        self.num_static_parameters = weight_count + bias_count

        # Linear transforms (Q up, rope Q up, K up, V up)
        linear_transforms = (batch_size * sequence_length * (2 * (q_latent_dim * layer_dim) + 2 * (kv_latent_dim * layer_dim)))

        # Attention computations: QK + (scores)V
        attention_score_calc = (batch_size * sequence_length * sequence_length * 2 * layer_dim)
        attention_output_calc = (batch_size * sequence_length * sequence_length * layer_dim)
        self_attention_comps = attention_score_calc + attention_output_calc
        self.num_macs = linear_transforms + self_attention_comps
        
        """ PARAMETERS AND EXECUTION PLAN """
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def define_parameters(self):
        """Defines the parameters for the latent attention layer."""
        static_params = {
            f"{self.name}_up_query_weight": {"dimensions": (self.layer_dim, self.q_latent_dim), "total": self.layer_dim * self.q_latent_dim},
            f"{self.name}_rope_up_query_weight": {"dimensions": (self.layer_dim, self.q_latent_dim), "total": self.layer_dim * self.q_latent_dim},
            f"{self.name}_up_key_weight": {"dimensions": (self.layer_dim, self.kv_latent_dim), "total": self.layer_dim * self.kv_latent_dim},
            f"{self.name}_up_value_weight": {"dimensions": (self.layer_dim, self.kv_latent_dim), "total": self.layer_dim * self.kv_latent_dim}
        }

        if self.add_bias:
            static_params.update({
                f"{self.name}_up_query_bias": {"dimensions": (1, self.layer_dim), "total": self.layer_dim},
                f"{self.name}_rope_up_query_bias": {"dimensions": (1, self.layer_dim), "total": self.layer_dim},
                f"{self.name}_up_key_bias": {"dimensions": (1, self.layer_dim), "total": self.layer_dim},
                f"{self.name}_up_value_bias": {"dimensions": (1, self.layer_dim), "total": self.layer_dim}
            })

        dynamic_params = {
            f"{self.name}_query": {"dimensions": (self.sequence_length, self.layer_dim), "total": self.sequence_length * self.layer_dim},
            f"{self.name}_key": {"dimensions": (self.sequence_length, self.layer_dim), "total": self.sequence_length * self.layer_dim},
            f"{self.name}_value": {"dimensions": (self.sequence_length, self.layer_dim), "total": self.sequence_length * self.layer_dim},
            f"{self.name}_rope_query": {"dimensions": (self.sequence_length, self.layer_dim), "total": self.sequence_length * self.layer_dim},
            f"{self.name}_concatenated_query": {"dimensions": (self.sequence_length, self.layer_dim * 2), "total": self.sequence_length * self.layer_dim * 2},
            f"{self.name}_concatenated_key": {"dimensions": (self.sequence_length, self.layer_dim * 2), "total": self.sequence_length * self.layer_dim * 2},
            f"{self.name}_attention_score": {"dimensions": (self.sequence_length, self.sequence_length), "total": self.sequence_length * self.sequence_length},
            f"{self.name}_output": {"dimensions": (self.sequence_length, self.layer_dim), "total": self.sequence_length * self.layer_dim}
        }
        return {"parameters": {"static": static_params, "dynamic": dynamic_params}}

    def define_plan(self):
        """Defines the execution plan for the latent attention layer."""
        plan = {
            "type": "sequential",
            "operations": [
                {
                    "type": "parallel",
                    "operations": [
                        {
                            "name": f"<{self.name}> Query Up Projection",
                            "computation": f"add(matmul({self.parent.name}_latent_query, {self.name}_up_query_weight), {self.name}_up_query_bias)" if self.add_bias else f"matmul({self.parent.name}_latent_query, {self.name}_up_query_weight)",
                            "output": f"{self.name}_query"
                        },
                        {
                            "name": f"<{self.name}> RoPE Query Up Projection",
                            "computation": f"add(matmul({self.parent.name}_latent_query, {self.name}_rope_up_query_weight), {self.name}_rope_up_query_bias)" if self.add_bias else f"matmul({self.parent.name}_latent_query, {self.name}_rope_up_query_weight)",
                            "output": f"{self.name}_rope_query"
                        },
                        {
                            "name": f"<{self.name}> Key Up Projection",
                            "computation": f"add(matmul({self.parent.name}_latent_key_value, {self.name}_up_key_weight), {self.name}_up_key_bias)" if self.add_bias else f"matmul({self.parent.name}_latent_key_value, {self.name}_up_key_weight)",
                            "output": f"{self.name}_key"
                        },
                        {
                            "name": f"<{self.name}> Value Up Projection",
                            "computation": f"add(matmul({self.parent.name}_latent_key_value, {self.name}_up_value_weight), {self.name}_up_value_bias)" if self.add_bias else f"matmul({self.parent.name}_latent_key_value, {self.name}_up_value_weight)",
                            "output": f"{self.name}_value"
                        },
                    ]
                },
                {
                    "type": "parallel",
                    "operations": [
                        {
                            "name": f"<{self.name}> Query Concatenation",
                            "computation": f"concat({self.name}_query, {self.name}_rope_query)",
                            "output": f"{self.name}_concatenated_query"
                        },
                        {
                            "name": f"<{self.name}> Key Concatenation",
                            "computation": f"concat({self.name}_key, {self.parent.name}_rope_key)",
                            "output": f"{self.name}_concatenated_key"
                        }
                    ]
                },
                {
                    "type": "sequential",
                    "operations": [
                        {
                            "name": f"<{self.name}> Attention Score Calculation",
                            "computation": f"matmul({self.name}_concatenated_query, {self.name}_concatenated_key)",
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