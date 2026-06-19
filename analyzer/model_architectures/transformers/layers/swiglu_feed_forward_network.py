from analyzer.core.model_architectures.transformer_blocks.layer import TransformerLayer


class SwiGLUFeedForwardNetwork(TransformerLayer):
    """
    SwiGLU-style (surrogate to accomodate gate)
      up:   D -> F      (full width)
      gate: D -> F/2
      combined: concat(up, gate) -> (3F/2)
      down: (3F/2) -> D

    Nonlinearities and elementwise gating are ignored
    (same abstraction level as rest of the simulator).
    """

    def __init__(
        self,
        name: str = "swiglu_ffn",
        sequence_length: int = 256,
        embedding_dim: int = 768,
        layer_dim: int = 3072,   # F
        batch_size: int = 1,
        add_bias: bool = False,
        parent: object = None,
        **kwargs
    ):
        F = layer_dim
        assert F % 2 == 0, "layer_dim (F) must be even for gate=F/2."
        self.gate_dim = F // 2
        self.up_dim = F
        self.combined_dim = self.up_dim + self.gate_dim  # 3F/2
        
        super().__init__(name, sequence_length, embedding_dim, self.up_dim, batch_size, add_bias, parent, **kwargs)
        
        

        D = embedding_dim
        B = batch_size
        S = sequence_length

        """ STATS """
        # Params: up (D*F) + gate (D*F/2) + down (D*(3F/2))
        if add_bias:
            self.num_static_parameters = (
                (D * self.up_dim + self.up_dim) +
                (D * self.gate_dim + self.gate_dim) +
                (D * self.combined_dim + D)
            )
        else:
            self.num_static_parameters = (
                (D * self.up_dim) +
                (D * self.gate_dim) +
                (D * self.combined_dim)
            )

        # MACs: three matmuls
        # self.num_macs = 3 * (B * S * D * F)
        
        # MACs: B*S*D*F + B*S*D*(F/2) + B*S*D*(3F/2) = 3*B*S*D*F
        self.num_macs = (
            (B * S * D * self.up_dim) +
            (B * S * D * self.gate_dim) +
            (B * S * D * self.combined_dim)
        )

        """ PARAMETERS AND EXECUTION PLAN """
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def define_parameters(self):
        D = self.embedding_dim
        S = self.sequence_length

        static_params = {
            f"{self.name}_up_weight":   {"dimensions": (self.up_dim, D), "total": self.up_dim * D},
            f"{self.name}_gate_weight": {"dimensions": (self.gate_dim, D), "total": self.gate_dim * D},
            f"{self.name}_down_weight": {"dimensions": (D, self.combined_dim), "total": D * self.combined_dim},
        }

        dynamic_params = {
            f"{self.name}_input":    {"dimensions": (S, D), "total": S * D},
            f"{self.name}_up":       {"dimensions": (S, self.up_dim), "total": S * self.up_dim},
            f"{self.name}_gate":     {"dimensions": (S, self.gate_dim), "total": S * self.gate_dim},
            f"{self.name}_combined": {"dimensions": (S, self.combined_dim), "total": S * self.combined_dim},
            f"{self.name}_output":   {"dimensions": (S, D), "total": S * D},
        }

        return {"parameters": {"static": static_params, "dynamic": dynamic_params}}

    def define_plan(self):
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
                            "name": f"<{self.name}> Up Projection",
                            "computation": f"matmul({self.name}_input, {self.name}_up_weight)",
                            "output": f"{self.name}_up",
                        },
                        {
                            "name": f"<{self.name}> Gate Projection",
                            "computation": f"matmul({self.name}_input, {self.name}_gate_weight)",
                            "output": f"{self.name}_gate",
                        },
                    ],
                },
                {
                    "name": f"<{self.name}> Combine (concat)",
                    "computation": f"concat({self.name}_up, {self.name}_gate)",
                    "output": f"{self.name}_combined",
                },
                {
                    "name": f"<{self.name}> Down Projection",
                    "computation": compute_str("matmul", f"{self.name}_combined", f"{self.name}_down_weight", f"{self.name}_down_bias"),
                    "output": f"{self.name}_output",
                },
            ],
        }
        return plan
