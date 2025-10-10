from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess
import shutil


def read_requirements():
    with open('requirements.txt') as req:
        return req.readlines()


class InstallCommand(install):
    """Customized setuptools install command to handle submodules."""
    def run(self):
        # Initialize submodules
        subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])

        # Build and install each submodule
        submodules = [
            'accelergy',
            'accelergy_plugins/accelergy-aladdin-plug-in',
            'accelergy_plugins/accelergy-library-plug-in',
            'accelergy_plugins/accelergy-cacti-plug-in',
            'accelergy_plugins/accelergy-adc-plug-in',
            'accelergy_plugins/accelergy-neurosim-plugin',
        ]

        for submodule in submodules:
            os.chdir(submodule)
            if submodule == 'accelergy_plugins/accelergy-cacti-plug-in':
                subprocess.check_call(['make', 'build'])
            if submodule == 'accelergy_plugins/accelergy-neurosim-plugin':
                subprocess.check_call([os.sys.executable, 'setup.py', 'build_ext'])
            subprocess.check_call([os.sys.executable, '-m', 'pip', 'install', '.'])
            os.chdir('../..' if 'accelergy_plugins' in submodule else '..')

        super().run()

        # Clean the build lib (it is just a Python module)
        build_lib_dir = os.path.join('build', 'lib')
        if os.path.exists(build_lib_dir):
            shutil.rmtree(build_lib_dir)

setup(
    name='transinfersim',
    version='0.1',
    description='Simulation framework for analyzing Transformer NN inference on HW',
    author='Jan Klhufek',
    author_email='iklhufek@fit.vut.cz',
    url='https://github.com/ehw-fit/TransInferSim',
    packages=find_packages(),
    install_requires=read_requirements(),
    cmdclass={
        'install': InstallCommand,
    }
)