from analyzer.core.model_architectures.transformer_blocks.model import TransformerModel
from ..layers.grouped_query_attention import GroupedQueryAttention
from ..layers.swiglu_feed_forward_network import SwiGLUFeedForwardNetwork
from ..layers.feed_forward_network import FeedForwardNetwork
from ..layers.encoder import Encoder  # TODO add decoder
import json

class DeepSeekR1_1_5B(TransformerModel):
    """
    DeepSeek-R1-Distill-Qwen-1.5B
    A dense, decoder-only model based on the Qwen-2.5-1.5B architecture.
    """
    def __init__(self, 
                 name: str = "deepseek_r1_distill_1.5b", 
                 sequence_length: int = 4096, 
                 num_layers: int = 28,           # Qwen2.5-1.5B standard
                 embedding_dim: int = 1536,      # Hidden size
                 ffn_layer_dim: int = 8960,      # FFN intermediate size
                 num_heads: int = 12,            # Number of attention heads
                 num_key_value_heads: int = 2,
                 batch_size: int = 1, 
                 add_bias: bool = False, 
                 attention_type = GroupedQueryAttention, # 1.5B uses GQA
                 ffn_type = FeedForwardNetwork,  # todo add real swiglu
                 **kwargs):
        
        super().__init__(name, sequence_length, embedding_dim, batch_size, add_bias, **kwargs)
        
        # Instantiate 28 identical encoder blocks (Dense, no MoE)
        for i in range(1, num_layers + 1):
            self.add_layer(Encoder(
                f"decoder_block_{i}", 
                sequence_length, 
                embedding_dim, 
                ffn_layer_dim, 
                num_heads, 
                batch_size, 
                add_bias, 
                parent=self, 
                attention_type=attention_type, 
                ffn_type=ffn_type,
                num_key_value_heads=num_key_value_heads,
                **kwargs
            ))
    
    

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