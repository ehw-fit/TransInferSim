# InferSim
TransInferSim is a simulation framework for analyzing Transformer NN inference on hardware. This repository includes various plugins and tools for energy estimation using the Accelergy framework.

## Features
- Analyzes Transformer NN inference on hardware
- Integrates with Accelergy for energy estimation
- Includes various plugins for Accelergy's flexibility

### Reference
TODO

## Installation
To get started with TransInferSim, follow these steps:

### Prerequisites
- Python 3.8 or higher

This project requires Graphviz to be installed on your system.
On Ubuntu/Debian, you can install it using:

```sh
sudo apt-get install graphviz
```

### Clone and build the Repository
Clone the repository and its submodules and build using pip:

```sh
git clone --recurse-submodules https://github.com/ehw-fit/TransInferSim
cd TransInferSim
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
./scripts/setup_submodules.sh
pip install .
```

### Usage
You can find an example run in the `example.py` script, which demonstrates how to instantiate a transformer model or layer of your choice along with a showcase of an example hardware specification. The script then runs an inference simulation, and the runtime performance statistics are saved to a `stats_out.txt` file.

### Licence
This project is licensed under the MIT License - see the LICENSE file for details.
