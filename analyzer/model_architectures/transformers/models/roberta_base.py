from analyzer.core.model_architectures.transformer_blocks.model import TransformerModel
from ..layers.encoder import Encoder
import json


class RobertaBase(TransformerModel):
    def __init__(self, name: str = "roberta_base", sequence_length: int = 256, num_layers: int = 12, embedding_dim: int = 768, ffn_layer_dim: int = 3072, num_heads: int = 12, batch_size: int = 1, add_bias: bool = False, **kwargs):
        super().__init__(name, sequence_length, embedding_dim, batch_size, add_bias, **kwargs)

        # Add encoder layers
        for i in range(1, num_layers+1):
            self.add_layer(Encoder(f"encoder_layer_{i}", sequence_length, embedding_dim, ffn_layer_dim, num_heads, batch_size, add_bias, self, **kwargs))
        
        self.num_static_parameters = sum(l.num_static_parameters for l in self.layers)
        self.num_macs = sum(l.num_macs for l in self.layers)

        """ PARAMETERS AND EXECUTION PLAN """
        # Define parameters and execution plan
        # TODO now we ignore softmax (attention scores are the result of softmax on query ⋅ key) and layer normalizations
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def define_parameters(self):
        """Aggregate parameters from all layers to define the parameters for the roberta-base model."""
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
        """Aggregate execution plans from all layers to define the execution plan for the roberta-base model."""
        operations = []
        for i, layer in enumerate(self.layers):
            # Convert the layer's plan to JSON string to perform the replacement
            layer_plan_str = json.dumps(layer.plan)

            # Replace "input" only for layers after the first
            if i > 0:
                # The output that should replace the input placeholder in the current layer's plan
                previous_ffn_output = f"{self.layers[i-1].layers[-1].name}_output"
                # Replace the placeholder 'input' with the actual output of the previous layer's FFN
                layer_plan_str = layer_plan_str.replace("input", previous_ffn_output)
            
            # Convert the modified string back to a dictionary and add to the overall operations
            layer_plan = json.loads(layer_plan_str)
            operations.append(layer_plan)

        # Create the final plan dictionary
        plan = {
            "type": "sequential",
            "operations": operations
        }
        return plan
