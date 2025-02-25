from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess
import shutil

def read_requirements():
    with open('requirements.txt') as req:
        return req.readlines()

class InstallCommand(install):
    """Customized setuptools install command to handle downloading submodules."""
    def run(self):
        # URLs for the submodules to be downloaded
        submodule_urls = {
            'accelergy': 'https://github.com/Accelergy-Project/accelergy.git',
            'accelergy_plugins/accelergy-aladdin-plug-in': 'https://github.com/Accelergy-Project/accelergy-aladdin-plug-in.git',
            'accelergy_plugins/accelergy-library-plug-in': 'https://github.com/Accelergy-Project/accelergy-library-plug-in.git',
            'accelergy_plugins/accelergy-cacti-plug-in': 'https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git',
            'accelergy_plugins/accelergy-adc-plug-in': 'https://github.com/Accelergy-Project/accelergy-adc-plug-in.git',
            'accelergy_plugins/accelergy-neurosim-plugin': 'https://github.com/Accelergy-Project/accelergy-neurosim-plugin.git',
        }

        # Download and install each submodule
        for submodule, url in submodule_urls.items():
            if os.path.exists(submodule):
                print(f"Directory '{submodule}' already exists. Pulling latest changes.")
                subprocess.check_call(['git', '-C', submodule, 'pull'])
            else:
                print(f"Cloning '{submodule}' from {url}.")
                subprocess.check_call(['git', 'clone', '--recurse-submodules', url, submodule])

        print("Starting build steps...")
        for submodule, _ in submodule_urls.items():
            os.chdir(submodule)
            if submodule == 'accelergy_plugins/accelergy-cacti-plug-in':
                print(f"Running 'make build' in {submodule}...")
                subprocess.check_call(['make', 'build'])
            if submodule == 'accelergy_plugins/accelergy-neurosim-plugin':
                print(f"Building {submodule} with setup.py...")
                subprocess.check_call([os.sys.executable, 'setup.py', 'build_ext'])
            
            print(f"Installing {submodule}...")
            subprocess.check_call([os.sys.executable, '-m', 'pip', 'install', '.'])
            os.chdir('../..' if 'accelergy_plugins' in submodule else '..')

        super().run()

        # Clean the build lib (it is just a Python module)
        build_lib_dir = os.path.join('build', 'lib')
        if os.path.exists(build_lib_dir):
            shutil.rmtree(build_lib_dir)

setup(
    name='TransInferSim',
    version='0.1',
    description='Simulation framework for analyzing transformer inference on HW',
    author='Jan Klhufek',
    author_email='iklhufek@fit.vut.cz',
    url='https://git.fit.vutbr.cz/iklhufek/TransInferSim.git',
    packages=find_packages(),
    install_requires=read_requirements(),
    cmdclass={
        'install': InstallCommand,
    }
)