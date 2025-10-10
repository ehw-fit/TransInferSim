from analyzer.core.model_architectures.transformer_blocks.model import TransformerModel
from ..layers.encoder import Encoder
import json


class ViTTiny(TransformerModel):
    def __init__(self, name: str = "vit_tiny", image_size: int = 224, patch_size: int = 16, num_layers: int = 12, embedding_dim: int = 192, ffn_layer_dim: int = 768, num_heads: int = 3, batch_size: int = 1, add_bias: bool = False, **kwargs):
        # Number of patches is derived from image size and patch size
        self.num_patches = (image_size // patch_size) ** 2  # e.g., 14 * 14 = 196 for 224x224 image with 16x16 patches
        self.patch_size = patch_size
        super().__init__(name, self.num_patches, embedding_dim, batch_size, add_bias, **kwargs)

        # Add encoder layers
        for i in range(1, num_layers+1):
            self.add_layer(Encoder(f"encoder_layer_{i}", self.num_patches, embedding_dim, ffn_layer_dim, num_heads, batch_size, add_bias, self, **kwargs))
        
        self.num_static_parameters = sum(l.num_static_parameters for l in self.layers)
        self.num_macs = sum(l.num_macs for l in self.layers)

        """ PARAMETERS AND EXECUTION PLAN """
        # Define parameters and execution plan
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def define_parameters(self):
        """Aggregate parameters from all layers to define the parameters for the ViT-Tiny model."""
        static_params = {}
        dynamic_params = {}
        for i, layer in enumerate(self.layers):
            static_layer_params = layer.parameters['parameters']['static']
            dynamic_layer_params = layer.parameters['parameters']['dynamic']
            
            for key, value in static_layer_params.items():
                if ("input" not in key and i > 0) or i == 0:
                    static_params[key] = value

            for key, value in dynamic_layer_params.items():
                if ("input" not in key and i > 0) or i == 0:
                    dynamic_params[key] = value

        return {"parameters": {"static": static_params, "dynamic": dynamic_params}}

    def define_plan(self):
        """Aggregate execution plans from all layers to define the execution plan for the ViT-Tiny model."""
        operations = []
        for i, layer in enumerate(self.layers):
            layer_plan_str = json.dumps(layer.plan)

            if i > 0:
                previous_ffn_output = f"{self.layers[i-1].layers[-1].name}_output"
                layer_plan_str = layer_plan_str.replace("input", previous_ffn_output)
            
            layer_plan = json.loads(layer_plan_str)
            operations.append(layer_plan)

        plan = {
            "type": "sequential",
            "operations": operations
        }
        return plan