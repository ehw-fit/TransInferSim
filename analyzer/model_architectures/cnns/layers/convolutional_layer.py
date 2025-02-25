from analyzer.core.model_architectures.convolutional_blocks.layer import Layer

class ConvolutionalLayer(Layer):
    def __init__(self, name: str, input_shape: tuple, output_channels: int,
                 kernel_size: tuple, stride: tuple, padding: tuple,
                 batch_size: int, add_bias: bool, im2col: bool = True):
        super().__init__(name, batch_size, add_bias)
        # Specific attributes for convolutional layers
        self.input_shape = input_shape  # (height_in, width_in, channels_in)
        self.output_channels = output_channels
        self.kernel_size = kernel_size  # (kernel_height, kernel_width)
        self.stride = stride  # (stride_height, stride_width)
        self.padding = padding  # (pad_height, pad_width)
        self.im2col = im2col

        """ STATS """
        # Calculate output dimensions
        self.output_height = ((self.input_shape[0] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0]) + 1
        self.output_width = ((self.input_shape[1] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1]) + 1
        self.output_shape = (self.output_height, self.output_width, self.output_channels)

        # Calculate the number of parameters (weights and biases)
        kernel_height, kernel_width = self.kernel_size
        channels_in = self.input_shape[2]
        self.num_static_parameters = self.output_channels * channels_in * kernel_height * kernel_width
        if self.add_bias:
            self.num_static_parameters += self.output_channels  # One bias per output channel

        # Calculate the number of MAC ops
        self.num_macs = self.batch_size * self.output_channels * self.output_height * self.output_width * channels_in * kernel_height * kernel_width

        """ PARAMETERS AND EXECUTION PLAN """
        # Define parameters and execution plan
        self.parameters = self.define_parameters()
        self.plan = self.define_plan()

    def __str__(self):
        base_str = super().__str__()
        return (base_str[:-1] +
                f" Input Shape: {self.input_shape} "
                f"Output Shape: {self.output_shape} "
                f"Output Channels: {self.output_channels} "
                f"Kernel Size: {self.kernel_size} "
                f"Stride: {self.stride} "
                f"Padding: {self.padding} "
                f"im2col: {self.im2col}>")

    def define_parameters(self):
        """Defines the parameters for the convolutional layer."""
        if self.im2col:
            kernel_height, kernel_width = self.kernel_size
            channels_in = self.input_shape[2]
            weight_dimensions = (kernel_height * kernel_width * channels_in, self.output_channels)
        
            static_params = {
                f"{self.name}_weight": {"dimensions": weight_dimensions, "total": weight_dimensions[0] * weight_dimensions[1]}
            }

            if self.add_bias:
                static_params[f"{self.name}_bias"] = {"dimensions": (1, self.output_channels), "total": self.output_channels}

            dynamic_params = {
                f"{self.name}_input_unfolded": {"dimensions": (self.output_height * self.output_width, kernel_height * kernel_width * channels_in), "batch": self.batch_size, "total": self.batch_size * self.output_height * self.output_width * kernel_height * kernel_width * channels_in},
                f"{self.name}_output": {"dimensions": (self.output_height * self.output_width, self.output_channels), "batch": self.batch_size, "total": self.batch_size * self.output_height * self.output_width * self.output_channels}
            }

            return {"parameters": {"static": static_params, "dynamic": dynamic_params}}
        else:
            raise NotImplementedError("Standard convolution without im2col is not yet implemented")


    def define_plan(self):
        """Defines the execution plan for the convolutional layer."""
        if self.im2col:
            # Names for variables
            input_var = f"{self.name}_input_unfolded"
            weight_var = f"{self.name}_weight"
            bias_var = f"{self.name}_bias" if self.add_bias else None
            output_var = f"{self.name}_output"
            
            # Execution plan using only matmul and add operations
            plan = {
                "type": "sequential",
                "operations": [
                    {
                        "name": f"<{self.name}> Convolution MatMul",
                        "computation": f"matmul({input_var}, {weight_var})",
                        "output": f"{self.name}_matmul_output"
                    },
                    {
                        "name": f"<{self.name}> Add Bias" if self.add_bias else None,
                        "computation": f"add({self.name}_matmul_output, {bias_var})" if self.add_bias else None,
                        "output": output_var if self.add_bias else f"{self.name}_matmul_output"
                    }
                ]
            }

            # Remove None entries if bias is not added
            plan["operations"] = [op for op in plan["operations"] if op["name"] is not None]
            return plan
        else:
            raise NotImplementedError("Standard convolution without im2col is not yet implemented")
