'''Setup to install the exact exchange package via pip

Version: 23.11.23
Author: Elias Polak'''

from setuptools import setup, find_packages
import subprocess


NAME            = 'exact_exchange'
DESCRIPTION     = 'Functionalities for the evaluation of the local slope based MP2 correlation energy density'
URL             = 'https://github.com/vuckovic-lab/energy_densities.git'
AUTHOR          = 'Elias Polak' 
AUTHOR_EMAIL    = 'elias_93@hotmail.de'


setup(
    name=NAME,
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    version="1.0", 
    
    packages=find_packages(exclude=['*Examples*', '*test*','*Data_results*']), #Packages only 
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