# exact_exchange
Project repository for the evaluation of the exact exchange -- Functionalities to obtain the exact exchange based exchange energy and energy density function. 


- Exact exchange is taken from:
    > [S. Vuckovic, T. J. P. Irons, A. Savin, A. M. Teale, and P. Gori-Giorgi, “Exchange-Correlation Functionals via Local Interpolation along the Adiabatic Connection”, Journal of Chemical Theory and Computation 12, 2598-2610 (2016)](https://pubs.acs.org/doi/10.1021/acs.jctc.6b00177).

# Startup
The following is a quick tutorial on how to get going with git, download the latest repository and install the functionalities.

1. Clone the latest version on the github repository:

        git clone https://github.com/vuckovic-lab/exact_exchange.git


The installation via pip will add the functionalities to the global python path. Without installation they have to be added manually. 

2. Go to the downloaded github repository with the functionalities:

        cd exact_exchange/

3. Install the local **exact_exchange** package via pip:

        pip install .

## Requirements:

- build >= 1.0.3
- twine >= 4.0.2
- setuptools >= 68.0.0
- jax >= 0.4.20
- jaxlib >= 0.4.20
- pyscf >= 2.3.0
- opt-einsum >= 3.3.0

## Usage
The python functionalities are employed with the import command (see the content section below for a list of available functions):

        import exact_exchange as ex


# Content 
The repository contains the main python functionalities for the exact exchange evaluation along with some examples for illustration and for testing. 

## exact_exchange functions
Available python functions for the local slope evaluation: 

- **ex_args** 

    Extracts the input arguments from a SCF kernel calculations for ex_eval function. 
    Also creates or reads optionally a 3D mesh grid and adds artificial uniform weights.
    Make sure to use * to unpack.
    
        Input: 
        mf           : SCF class calculation
        mol          : GTO molecular structure
        Abasis       : String specifying the auxiliary basis set
        grids=3      : pyscf.dft generated grid with a specified grid level or a user defined 3D grid.
        
        Output:
        args    : List of output objects, which are needed for the ec_mp2 functions
                  0:dm, 1:mol, 2:Amol, 3:coords, 4:weights

- **ex_kwargs**

    Extracting the optional arguments for the ex_eval class function. Make sure to use * to unpack.
    
        Input:
        batch_size = 0           : Batch size to run the evaluation on, 0 is for no parallelization
        DF = True                : Density fitting option
        verbose = False          : Additional printings of time and memory statements
        
        Output:
        kwargs                   : List of optional arguments
                                   0:batch_size, 1:DF, 2:verbose    


- **ex_ref_eval** 
        
    Exact evaluation of the exchange energy directly from the density matrix.
    
        Input: 
        mf  : Kernel of a RKS/RHF-Calculation
        
        Output:
        Ex  : Exchange energy value 

- **ex_eval(\*args,\*kwargs)** 

    Evalation of the exact exchange energy density on a given grid of coords.
    Density fitting, or batchwise parallelization are optional arguments.

        *args* 
        dm      : Density matrix from a scf calculation (#basis,#basis)
        mol     : gto molecular structure incorporating the basis set.
        Amol    : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        
        coords  : Grid coordinates (#coords,3)
        weights : Grid weights (#coords,3)
        
        
        **kwargs
        batch_size = 0    : Batch size to run the evaluation on, 0 is for no parallelization
        DF = True         : Density fitting option
        verbose = False   : Additional printings of time and memory statements
        

   Returns a class object that contains two main properties:
        
        Ex = ex_eval(*args,*kwargs)
        Ex.energy                       #Exact exchange based exchange energy
        Ex.array                        #Exact exchange based exchange energy density

## Examples
Provides an example scripts upon evaluation of the exact exchange energy density. 

- **main.py** : Simple script to run the exact exchange evaluation and a corresponding reference calculation to compare the exchange energies. 

# Contact
Author: [Elias-Py09](https://github.com/Elias-Py09) 

Group-Homepage: [Vuckovic group](https://www.unifr.ch/chem/en/research/groups/vuckovic-group/)


