from analyzer.core.model_architectures.transformer_blocks.layer import TransformerLayer

class GroupedQueryAttention(TransformerLayer):
    def __init__(
        self,
        name: str = "gqa",
        sequence_length: int = 256,
        embedding_dim: int = 768,
        num_heads: int = 12,
        num_key_value_heads: int = 2,
        batch_size: int = 1,
        add_bias: bool = False,
        parent: object = None,
        **kwargs
    ):
        super().__init__(name, sequence_length, embedding_dim, embedding_dim, batch_size, add_bias, parent, **kwargs)

        assert num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"

        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = embedding_dim // num_heads
        self.kv_dim = num_key_value_heads * self.head_dim
        self.kv_repeat = num_heads // num_key_value_heads

        D = self.embedding_dim
        S = self.sequence_length
        B = self.batch_size
        KV = self.kv_dim

        """ STATS """
        # Weight parameters
        weight_count = (D * D) + (D * KV) + (D * KV) + (D * D)
        # Bias parameters
        bias_count = (D + KV + KV + D) if self.add_bias else 0
        self.num_static_parameters = weight_count + bias_count

        # MACs: Projections (Q, K, V, Out)
        proj_macs = B * S * ( (D * D) + (D * KV) + (D * KV) + (D * D) )
        # MACs: Attention matmuls (Score calculation + Context aggregation)
        attn_macs = 2 * B * S * S * D
        
        self.num_macs = proj_macs + attn_macs

        """ PARAMETERS AND EXECUTION PLAN """
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def define_parameters(self):
        D = self.embedding_dim
        S = self.sequence_length
        KV = self.kv_dim

        static_params = {
            f"{self.name}_query_weight": {"dimensions": (D, D), "total": D * D},
            f"{self.name}_key_weight": {"dimensions": (D, KV), "total": D * KV},
            f"{self.name}_value_weight": {"dimensions": (D, KV), "total": D * KV},
            f"{self.name}_output_projection_weight": {"dimensions": (D, D), "total": D * D},
        }

        if self.add_bias:
            static_params.update({
                f"{self.name}_query_bias": {"dimensions": (1, D), "total": D},
                f"{self.name}_key_bias": {"dimensions": (1, KV), "total": KV},
                f"{self.name}_value_bias": {"dimensions": (1, KV), "total": KV},
                f"{self.name}_output_projection_bias": {"dimensions": (1, D), "total": D},
            })

        dynamic_params = {
            "input": {"dimensions": (S, D), "total": S * D},
            f"{self.name}_query": {"dimensions": (S, D), "total": S * D},
            f"{self.name}_key": {"dimensions": (S, KV), "total": S * KV},
            f"{self.name}_value": {"dimensions": (S, KV), "total": S * KV},
            f"{self.name}_key_repeated": {"dimensions": (S, D), "total": S * D},
            f"{self.name}_value_repeated": {"dimensions": (S, D), "total": S * D},
            f"{self.name}_attention_score": {"dimensions": (S, S), "total": S * S},
            f"{self.name}_context": {"dimensions": (S, D), "total": S * D},
            f"{self.name}_output": {"dimensions": (S, D), "total": S * D},
        }

        return {"parameters": {"static": static_params, "dynamic": dynamic_params}}

    def define_plan(self):
        key_repeat_list = ", ".join([f"{self.name}_key"] * self.kv_repeat)
        value_repeat_list = ", ".join([f"{self.name}_value"] * self.kv_repeat)

        # Helper to format computation string based on bias
        def compute_str(op, in1, in2, bias_name):
            if self.add_bias:
                return f"add({op}({in1}, {in2}), {bias_name})"
            return f"{op}({in1}, {in2})"

        plan = {
            "type": "sequential",
            "operations": [
                {
                    "type": "parallel",
                    "operations": [
                        {
                            "name": f"<{self.name}> Query Projection",
                            "computation": compute_str("matmul", "input", f"{self.name}_query_weight", f"{self.name}_query_bias"),
                            "output": f"{self.name}_query",
                        },
                        {
                            "name": f"<{self.name}> Key Projection",
                            "computation": compute_str("matmul", "input", f"{self.name}_key_weight", f"{self.name}_key_bias"),
                            "output": f"{self.name}_key",
                        },
                        {
                            "name": f"<{self.name}> Value Projection",
                            "computation": compute_str("matmul", "input", f"{self.name}_value_weight", f"{self.name}_value_bias"),
                            "output": f"{self.name}_value",
                        },
                    ],
                },
                {
                    "type": "sequential",
                    "operations": [
                        {
                            "name": f"<{self.name}> Repeat K",
                            "computation": f"broadcast({key_repeat_list})",
                            "output": f"{self.name}_key_repeated",
                        },
                        {
                            "name": f"<{self.name}> Repeat V",
                            "computation": f"broadcast({value_repeat_list})",
                            "output": f"{self.name}_value_repeated",
                        },
                        {
                            "name": f"<{self.name}> Attention Score",
                            "computation": f"matmul({self.name}_query, {self.name}_key_repeated)",
                            "output": f"{self.name}_attention_score",
                        },
                        {
                            "name": f"<{self.name}> Context",
                            "computation": f"matmul({self.name}_attention_score, {self.name}_value_repeated)",
                            "output": f"{self.name}_context",
                        },
                        {
                            "name": f"<{self.name}> Output Projection",
                            "computation": compute_str("matmul", f"{self.name}_context", f"{self.name}_output_projection_weight", f"{self.name}_output_projection_bias"),
                            "output": f"{self.name}_output",
                        },
                    ],
                },
            ],
        }
        return plan