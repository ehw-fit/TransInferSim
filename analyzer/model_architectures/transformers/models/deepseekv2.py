from analyzer.core.model_architectures.transformer_blocks.model import TransformerModel
from ..layers.multi_head_self_attention import MultiHeadSelfAttention
from ..layers.multi_head_latent_attention import MultiHeadLatentAttention
from ..layers.encoder import Encoder  # TODO add decoder
import json

class DeepSeekV2(TransformerModel):
    """
    An decoder-only model with `num_layers` repeated blocks.
    Each block can use either MHSA or MLA, controlled by `attention_type`.
    """
    def __init__(self, name: str = "deepseek_v2", sequence_length: int = 4096, num_layers: int = 60, embedding_dim: int = 5120, ffn_layer_dim: int = 25600, num_heads: int = 40, batch_size: int = 1, add_bias: bool = False, q_latent_dim: int = 3072, kv_latent_dim: int = 1024, attention_type = MultiHeadLatentAttention, **kwargs):
        super().__init__(name, sequence_length, embedding_dim, batch_size, add_bias, **kwargs)
        assert attention_type in [MultiHeadSelfAttention, MultiHeadLatentAttention], "attention_type must be either MultiHeadSelfAttention or MultiHeadLatentAttention."
        # TODO similarly add choice for FFN or MoE! now fixed MoE within encoder block
        # Instantiate N identical encoder blocks
        for i in range(1, num_layers + 1):
            self.add_layer(Encoder(f"decoder_block_{i}", sequence_length, embedding_dim, ffn_layer_dim, num_heads, batch_size, add_bias, parent=self, attention_type=attention_type, q_latent_dim=q_latent_dim, kv_latent_dim=kv_latent_dim, **kwargs))

        """ STATS """
        self.num_static_parameters = sum(layer.num_static_parameters for layer in self.layers)
        self.num_macs = sum(layer.num_macs for layer in self.layers)

        """ PARAMETERS AND EXECUTION PLAN """
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def define_parameters(self):
        """Aggregate parameters from all layers to define the parameters for the deepseekv2-encoder model."""
        static_params = {}
        dynamic_params = {}
        # Aggregate parameters from all layers
        for i, layer in enumerate(self.layers):
            static_layer_params = layer.parameters['parameters']['static']
            dynamic_layer_params = layer.parameters['parameters']['dynamic']
            
            # Do not add the output of the first layer as input to the second layer (duplicate parameters)
            for key, value in static_layer_params.items():
                if ("input" not in key and i > 0) or i == 0:
                    static_params[key] = value

            for key, value in dynamic_layer_params.items():
                if ("input" not in key and i > 0) or i == 0:
                    dynamic_params[key] = value
        return {"parameters": {"static": static_params, "dynamic": dynamic_params}}

    def define_plan(self):
        """Aggregate execution plans from all layers to define the execution plan for the deepseekv2-encoder model."""
        operations = []
        for i, layer in enumerate(self.layers):
            # Convert each block's plan to JSON string to perform the replacement
            layer_plan_str = json.dumps(layer.plan)
            
            # Replace "input" only for layers after the first
            if i > 0:
                # The output that should replace the input placeholder in the current layer's plan
                previous_ffn_output = f"{self.layers[i-1].layers[-1].name}_output"
                # Replace the placeholder 'input' with the actual output of the previous layer's FFN
                layer_plan_str = layer_plan_str.replace("input", previous_ffn_output)
            
            # Convert back to dict
            layer_plan_dict = json.loads(layer_plan_str)
            operations.append(layer_plan_dict)
        
        return {
            "type": "sequential",
            "operations": operations
        }