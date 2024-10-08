'''

Exact exchange based energy density

Documentation with examples are found on the github repository:
https://github.com/vuckovic-lab/energy_densities.git

Available functions:

    >>> import exact_exchange as ex
    >>> args = ex.ex_args()             #Extraction of necessary arguments 
    >>> kwargs = ex.ex_kwargs()         #Extraction of optional arguments 
    >>> ref = ex.ex_ref_eval()          #Calculation of the corresponding reference exchange energy
    >>> Ex = ex.ex_eval(*args,*kwargs)  #Evaluation of exact exchange energy density
    >>> Ex.array                        #Exact exchange based energy density array
    >>> Ex.energy                       #Exact exchange based energy
'''


#Importing all relevant functions for the evaluation
from ex_exact.ex_args_kwargs import ex_args, ex_kwargs
from ex_exact.ex_funcs import ex_eval
from ex_exact.ex_ref import ex_ref_eval




