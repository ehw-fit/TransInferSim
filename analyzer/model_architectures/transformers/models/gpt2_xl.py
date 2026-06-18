from analyzer.core.model_architectures.transformer_blocks.model import TransformerModel
from ..layers.encoder import Encoder
import json


class GPT2XL(TransformerModel):
    """
    Decoder-only Transformer (GPT-2 XL) expressed in the same structural style as other models:
    - Stack of decoder blocks (here reusing Encoder: MHSA + FFN)
    - No positional embedding phase
    - No nonlinearities / softmax / layernorm / residuals (consistent with your current blocks)

    Canonical GPT-2 XL hyperparams (commonly cited):
      num_layers=48, embedding_dim=1600, ffn_layer_dim=6400, num_heads=25.
    """

    def __init__(
        self,
        name: str = "gpt2_xl",
        sequence_length: int = 1024,
        num_layers: int = 48,
        embedding_dim: int = 1600,
        ffn_layer_dim: int = 6400,
        num_heads: int = 25,
        batch_size: int = 1,
        add_bias: bool = False,
        **kwargs
    ):
        super().__init__(name, sequence_length, embedding_dim, batch_size, add_bias, **kwargs)

        # Add "decoder" layers (structurally identical to your Encoder blocks under this abstraction)
        for i in range(1, num_layers + 1):
            self.add_layer(
                Encoder(
                    f"decoder_layer_{i}",
                    sequence_length,
                    embedding_dim,
                    ffn_layer_dim,
                    num_heads,
                    batch_size,
                    add_bias,
                    self,
                    **kwargs
                )
            )

        """ STATS """
        self.num_static_parameters = sum(l.num_static_parameters for l in self.layers)
        self.num_macs = sum(l.num_macs for l in self.layers)

        """ PARAMETERS AND EXECUTION PLAN """
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def define_parameters(self):
        """Aggregate parameters from all layers to define the parameters for the GPT-2 XL model."""
        static_params = {}
        dynamic_params = {}

        for i, layer in enumerate(self.layers):
            static_layer_params = layer.parameters["parameters"]["static"]
            dynamic_layer_params = layer.parameters["parameters"]["dynamic"]

            # Do not add duplicated "input" tensors for layers after the first
            for key, value in static_layer_params.items():
                if ("input" not in key and i > 0) or i == 0:
                    static_params[key] = value

            for key, value in dynamic_layer_params.items():
                if ("input" not in key and i > 0) or i == 0:
                    dynamic_params[key] = value

        return {"parameters": {"static": static_params, "dynamic": dynamic_params}}

    def define_plan(self):
        """Aggregate execution plans from all layers to define the execution plan for the GPT-2 XL model."""
        operations = []

        for i, layer in enumerate(self.layers):
            layer_plan_str = json.dumps(layer.plan)

            if i > 0:
                previous_ffn_output = f"{self.layers[i - 1].layers[-1].name}_output"
                layer_plan_str = layer_plan_str.replace("input", previous_ffn_output)

            layer_plan = json.loads(layer_plan_str)
            operations.append(layer_plan)

        return {"type": "sequential", "operations": operations}
