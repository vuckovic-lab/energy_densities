'''Setup to install the local slope package via pip

Version: 08.01.2024
Author: Elias Polak'''

from setuptools import setup, find_packages
import subprocess


NAME            = 'ec_MP2'
DESCRIPTION     = 'Functionalities for the evaluation of the MP2 correlation energy density'
URL             = 'https://github.com/vuckovic-lab/energy_densities'
AUTHOR          = 'Elias Polak' 
AUTHOR_EMAIL    = 'elias_93@hotmail.de'


#Obtain current version
def get_git_version(default="0.1"):
    try:
        version = subprocess.check_output(["git", "tag"], stderr=subprocess.STDOUT).strip().decode('utf-8')
        version_list = version.split()
        return version_list[-1]
    except subprocess.CalledProcessError:
        version = default
        return version
VERSION = get_git_version()

setup(
    name=NAME,
    description=DESCRIPTION,
    url=URL,
    licence='GNU',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    version="1.0", 
    
    packages=find_packages(exclude=['*Examples*', '*test*','*Data_results*','*build*','*local_slope.egg-info*']), #Exclude directories
    install_requires=[
        # List your package dependencies here
        # e.g., 'requests >= 2.22.0'
        'build >= 1.0.3',
        'twine >= 4.0.2',
        'setuptools >= 68.0.0',
        'jax >= 0.4.20',
        'jaxlib >= 0.4.20',
        'pyscf >= 2.3.0',
        'opt-einsum >= 3.3.0'
    ],
)