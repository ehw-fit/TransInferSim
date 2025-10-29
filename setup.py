from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()

setup(
    name='transinfersim',
    version='0.1',
    description='Simulation framework for analyzing Transformer NN inference on HW',
    author='Jan Klhufek',
    author_email='iklhufek@fit.vut.cz',
    url='https://github.com/ehw-fit/TransInferSim',
    packages=find_packages(),
    install_requires=read_requirements(),
)