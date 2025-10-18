'''Setup to install the local slope package via pip'''

from setuptools import setup, find_packages


NAME            = 'ec_MP2'
DESCRIPTION     = 'Functionalities for the evaluation of the MP2 correlation energy density'
URL             = 'https://github.com/vuckovic-lab/energy_densities'
AUTHOR          = 'Elias Polak' 
AUTHOR_EMAIL    = 'elias_93@hotmail.de'

setup(
    name=NAME,
    description=DESCRIPTION,
    url=URL,
    licence='GNU',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL, 
    
    packages=find_packages(exclude=['*Examples*','*ec_MP2.egg-info*']), 
    install_requires=[
        'build >= 1.0.3',
        'twine >= 4.0.2',
        'setuptools >= 68.0.0',
        'jax >= 0.4.20',
        'jaxlib >= 0.4.20',
        'pyscf >= 2.3.0',
        'opt-einsum >= 3.3.0'
    ],
)