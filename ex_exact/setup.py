'''Setup to install the exact exchange package via pip'''

from setuptools import setup, find_packages


NAME            = 'ex_exact'
DESCRIPTION     = 'Functionalities for the evaluation of the exact exchange energy density'
URL             = 'https://github.com/vuckovic-lab/energy_densities.git'
AUTHOR          = 'Elias Polak' 
AUTHOR_EMAIL    = 'elias_93@hotmail.de'


setup(
    name=NAME,
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    version="0.1", 
    
    packages=find_packages(exclude=['*Examples*','*build*','*ec_MP2.egg-info*']), #Packages only 
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