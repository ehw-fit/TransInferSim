from abc import ABC, abstractmethod
import yaml
from .layer import TransformerLayer

# Custom representer for serialization of Python tuple objects
def tuple_representer(dumper, data):
    return dumper.represent_list(data)

yaml.add_representer(tuple, tuple_representer)


class TransformerModel(ABC):
    """Abstract class for a transformer model for hardware inference analysis.

    Args:
        name (str): The name of the model.
        sequence_length (int): The length of the input sequence.
        embedding_dim (int): The dimensionality of the embedding.
        batch_size (int): The batch size for the model.
        add_bias (bool): Whether to account for bias parameters.
    """
    def __init__(self, name: str, sequence_length: int, embedding_dim: int, batch_size: int, add_bias: bool, **kwargs):
        assert sequence_length > 0
        assert embedding_dim > 0
        assert batch_size > 0

        self.name = name
        self.type = self.__class__.__name__
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.layers = []
        self.num_static_parameters = 0
        self.num_macs = 0
        self.batch_size = batch_size
        self.add_bias = add_bias

        # Set to keep track of all layer names to ensure uniqueness
        self._names = set()
    
    def __str__(self):
        return (f"<class={self.type} name={self.name}| "
                f"Sequence Length: {self.sequence_length} "
                f"Embedding Dim: {self.embedding_dim} "
                f"Batch Size: {self.batch_size} "
                f"Layers: {len(self.layers)} "
                f"Parameters: {self.num_static_parameters} "
                f"MACs: {self.num_macs} "
                f"Use bias: {self.add_bias}>")
    
    @abstractmethod
    def define_parameters(self):
        """Defines the parameters for the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def define_plan(self):
        """Defines the computational plan for the model. Must be implemented by subclasses."""
        pass

    @staticmethod
    def print_as_yaml_lib(data, file_path: str = None):
        """Prints the dictionary in YAML format using PyYAML library or writes the contents to a file.

        Args:
            data (dict): The data to be printed or saved to a file in YAML format.
            file_path (str, optional): The file path to write the YAML data. If None, the function prints to stdout.
        """
        yaml_data = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    file.write(yaml_data)
            except IOError as e:
                print(f"An error occurred while writing to the file: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        else:
            print(yaml_data)

    def get_all_layer_names(self, layer: TransformerLayer):
        """
        Recursively collects names of all layers and their sublayers, ensuring all names are unique.
        
        Args:
            layer (TransformerLayer): The layer from which to retrieve the names of sublayers.
        
        Raises:
            ValueError: If duplicate layer names are found.
        """
        def recurse_layers(current_layer):
            if current_layer.name in self._names:
                raise ValueError(f"Duplicate layer name detected: {current_layer.name}")
            self._names.add(current_layer.name)
            for sublayer in current_layer.layers:
                recurse_layers(sublayer)
        recurse_layers(layer)

    def add_layer(self, layer: TransformerLayer):
        """
        Adds a layer to the transformer model, ensuring that the layer name is unique.

        Args:
            layer (TransformerLayer): The layer to be added.
        """
        # Update the set of unique names from this layer's sublayers and check for naming duplicates
        self.get_all_layer_names(layer)
        self.layers.append(layer)

    def total_parameters(self):
        """Calculates the total number of parameters in the transformer model."""
        return self.num_static_parameters

    def total_computations(self):
        """Calculates the total number of computations for the transformer model."""
        return self.num_macs
