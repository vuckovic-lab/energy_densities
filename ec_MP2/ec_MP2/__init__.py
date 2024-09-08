'''

MP2 correlation energy density generator

Documentation with examples are found on the github repository:
https://github.com/vuckovic-lab/energy_densities

Available functions:

    >>> import ec_MP2 as ec
    >>> ec_mp2_args()    #Extraction of necessary arguments 
    >>> ec_mp2_kwargs()  #Extraction of optional arguments 
    >>> ec_ref_eval()    #Calculation of the PySCF MP2 correlation energy
    >>> ec_mp2_cs()      #Closed-shell MP2 correlation energy density generator
    >>> ec_mp2_os()      #Opposite spin (os) MP2 correlation energy density generator
'''


#Importing all relevant functions for the evaluation
from ec_MP2.ec_args_kwargs import ec_mp2_args, ec_mp2_kwargs
from ec_MP2.ec_ref_mp2 import ec_ref_eval
from ec_MP2.ec_mp2 import ec_mp2_cs, ec_mp2_os




