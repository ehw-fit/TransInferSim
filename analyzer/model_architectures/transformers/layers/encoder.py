from analyzer.core.model_architectures.transformer_blocks.layer import TransformerLayer
from .multi_head_self_attention import MultiHeadSelfAttention
from .feed_forward_network import FeedForwardNetwork
import json

class Encoder(TransformerLayer):
    def __init__(self, name: str = "encoder", sequence_length: int = 256, embedding_dim: int = 768, layer_dim: int = 3072, num_heads: int = 12, batch_size: int = 1, add_bias: bool = False, **kwargs):
        super().__init__(name, sequence_length, embedding_dim, layer_dim, batch_size, add_bias, **kwargs)
        # Add multi-head self-attention and feed-forward network as components
        self.add_layer(MultiHeadSelfAttention(f"{name}_mhsa", sequence_length, embedding_dim, num_heads, batch_size, add_bias, **kwargs))
        self.add_layer(FeedForwardNetwork(f"{name}_ffn", sequence_length, embedding_dim, layer_dim, batch_size, add_bias, **kwargs))

        """ STATS """
        self.num_static_parameters = sum(l.num_static_parameters for l in self.layers)
        self.num_macs = sum(l.num_macs for l in self.layers)

        """ PARAMETERS AND EXECUTION PLAN """
        # Define parameters and execution plan
        # TODO now we ignore softmax (attention scores are the result of softmax on query ⋅ key) and layer normalizations
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def define_parameters(self):
        """Aggregate parameters from all layers."""
        static_params = {}
        dynamic_params = {}
        for layer in self.layers:
            static_layer_params = layer.parameters['parameters']['static']
            dynamic_layer_params = layer.parameters['parameters']['dynamic']
            
            # Prevent adding duplicate parameters (the input to the ffn, as it is already added as the output of the mhsa)
            for key, value in static_layer_params.items():
                if f"{self.layers[1].name}_input" not in key:
                    static_params[key] = value

            for key, value in dynamic_layer_params.items():
                if f"{self.layers[1].name}_input" not in key:
                    dynamic_params[key] = value

        return {"parameters": {"static": static_params, "dynamic": dynamic_params}}

    def define_plan(self):
        """Aggregate execution plans from all layers."""
        plan = {
            "type": "sequential",
            "operations": []
        }
        for layer in self.layers:
            plan['operations'].append(layer.plan)

        plan_str = json.dumps(plan)
        plan_str = plan_str.replace(f"{self.layers[1].name}_input", f"{self.layers[0].name}_output")
        plan = json.loads(plan_str)

        return plan
