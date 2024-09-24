# MP2 Correlation Energy Density Generator
This is the project repository for the evaluation of the MP2 correlation energy density. It contains functionalities to obtain the MP2 correlation energy and energy density function. 

Version 0.4 (31/05/2024)

# Quickstart
- The repository can be cloned to the local directory using the *git clone* command. 

       git clone https://github.com/vuckovic-lab/energy_densities 

- Installation is possible with the *pip* command.

       pip install .

- The functions are directly called from the python files:

        from ec_MP2.ec_args_kwargs import ec_mp2_args, ec_mp2_kwargs
        from ec_MP2.ec_ref_mp2 import ec_ref_eval
        from ec_MP2.ec_mp2 import ec_mp2_cs, ec_mp2_os

- The available functions are

        ec_mp2_args()    #Extraction of necessary arguments 
        ec_mp2_kwargs()  #Extraction of optional arguments 
        ec_ref_eval()    #Calculation of the PySCF MP2 correlation energy
        ec_mp2_cs()      #Closed-shell MP2 correlation energy density generator
        ec_mp2_os()      #Opposite spin (os) MP2 correlation energy density generator
        
## Requirements:

- build >= 1.0.3
- twine >= 4.0.2
- setuptools >= 68.0.0
- jax >= 0.4.20
- jaxlib >= 0.4.20
- pyscf >= 2.3.0
- opt-einsum >= 3.3.0

# Content 
This repository contains the main python functions for along with some examples for illustrations and for testing. 

## Input functionalities
The script *ec_args_kwargs* contains two main functions to extract necessary and optional arguments.

- **ec_mp2_args** 

    Extracts the input arguments from a SCF kernel calculation for the ec_mp2 class functions. Also creates or reads optionally a 3D mesh grid and adds artificial uniform weights. Make sure to use * to unpack.
        
        Input: 
        mf           : SCF class calculation
        mol          : GTO molecular structure
        Abasis       : String specifying the auxiliary basis set
        grids=3      : pyscf.dft generated grid with a specified grid level or a user defined 3D grid.
        
        Output:
        args         : List of output objects, needed for the ec_mp2 functions
                        0:dm, 1:mol, 2:Amol, 3:mo_coeff, 4:mo_occ,
                        5:mo_energies, 6:coords, 7:weights

- **ec_mp2_kwargs**

   Extracting the optional arguments for the ec_mp2 class functions. Make sure to use * to unpack.
                
        Input:
        batch_size = 0           : Grid batch size to run the evaluation on, 0 is for no parallelization
        DF = True                : Density fitting option
        verbose = False          : Additional printings of time and memory statements
        optimal_contract = False : Optimized contraction algorithm for the opt_einsum summation path
        max_num_array = None     : Maximum number of elements in a temporary array for the optimal contraction path, requires a postive integer.
        frozen_core = False      : Frozen core orbital option
        spinorb = False          : Specification of the spin orbital usage of frozen core orbitals, False for R/U is fine
        num_core = 'auto'        : Amount of frozen orbitals, 'auto' selects the core ones.
        kappa = 'inf'            : Laplace transform regularization parameter for the doubles amplitudes, 'inf' for no regularization (original MP2 expression)
        
        Output:
        kwargs                   : List of optional arguments
                                   0:batch_size, 1:DF, 2:verbose, 3:optimal_contract,
                                   4:max_num_array, 5:frozen_core, 6:spinorb, 7:num_core, 8:kappa   

## Reference evaluation
The script *ec_ref_mp2* containts the function to evaluate the MP2 correlation energy in PySCF.

- **ec_ref_eval** 
        
   Reference correlation energy from a corresponding MP2 calculation.  
                
        Input: 
        mf                  : Kernel of the RKS/RHF-calculation
        mol                 : GTO molecular structure
        DF=False            : Density fitting option for the MP2 calculation 
        Abasis=None         : String specifying the auxiliary basis set
        frozen_core = False : Frozen core orbital option
        spinorb = False     : Specification of the spin orbital usage of frozen core orbitals, False for R/U is fine
        num_core = 0        : Amount of frozen orbitals, 'auto' selects the core ones.
        
        Output: 
        Ec_ref          : Reference MP2 correlation energy
        Ec_ref_os       : Opposite spin based MP2 correlation energy
        Ec_ref_ss       : Same spin based MP2 correlation energy

## ec_MP2 functions
The script *ec_mp2* containts the python class functions for the MP2 energy density generator

- **ec_mp2_cs(\*args,\*kwargs)** 

    Evaluation of the MP2 correlation energy density for a closed shell system. Density fitting, parallelization, frozen core orbitals or kappa regularization are optional arguments. Requires a RKS or RHF Pyscf class object.

        *args 
        dm          : Density matrix from a scf calculation (#basis,#basis)
        mol         : gto molecular structure incorporating the basis set.
        Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        mo_coeff    : Coefficient matrix of the atomic orbitals (#basis,#basis)
        mo_occ      : Occupation numbers (#basis,)
        mo_energies : Orbital energies (#basis,)
        coords      : Grid coordinates (#coords,3)
        weights     : Grid weights (#coords,1)
        
        **kwargs
        batch_size = 0           : Grid batch size to run the evaluation on, 0 is for no parallelization
        DF = True                : Density fitting option
        verbose = False          : Additional printings of time and memory statements
        optimal_contract = False : Optimized contraction algorithm for the opt_einsum summation path
        max_num_array = None     : Maximum number of elements in a temporary array for the optimal contraction path, requires a postive integer.
        frozen_core = False      : Frozen core orbital option
        spinorb = False          : Specification of the spin orbital usage of frozen core orbitals, False for R/U is fine
        num_core = 'auto'        : Amount of frozen orbitals, 'auto' selects the core ones.
        kappa = 'inf'            : Laplace transform regularization parameter for the doubles amplitudes, 'inf' for no regularization (original MP2 expression)

   Returns a class object that contains two main properties:
        
        Ec = ec_mp2_cs(*args,*kwargs)
        Ec.energy                       #MP2 correlation energy
        Ec.array                        #MP2 correlation energy density

- **ec_mp2_os(\*args,\*kwargs)** 

    Evaluation of the opposite spin based MP2 correlation energy density. 
    Density fitting, batchwise parallelization, frozen core orbitals or kappa 
    regularization are optional arguments. 

        *args
        dm          : Density matrix from a scf calculation (#basis,#basis)
        mol         : gto molecular structure incorporating the basis set.
        Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis).
        mo_coeff    : Coefficient matrix of the atomic orbitals (#basis,#basis)
        mo_occ      : Occupation numbers (#basis,)
        mo_energies : Orbital energies (#basis,)
        coords      : Grid coordinates (#coords,3)
        weights     : Grid weights (#coords,3)

        **kwargs
        batch_size = 0           : Grid batch size to run the evaluation on, 0 is for no parallelization
        DF = True                : Density fitting option
        verbose = False          : Additional printings of time and memory statements
        optimal_contract = False : Optimized contraction algorithm for the opt_einsum summation path
        max_num_array = None     : Maximum number of elements in a temporary array for the optimal contraction path, requires a postive integer.
        frozen_core = False      : Frozen core orbital option
        spinorb = False          : Specification of the spin orbital usage of frozen core orbitals, False for R/U is fine
        num_core = 'auto'        : Amount of frozen orbitals, 'auto' selects the core ones.
        kappa = 'inf'            : Laplace transform regularization parameter for the doubles amplitudes, 'inf' for no regularization (original MP2 expression)
   
   Returns a class object that contains two main properties:
        
        Ec_os = ec_mp2_os(*args,*kwargs)
        Ec_os.energy          #Opposite spin MP2 correlation energy
        Ec_os.array           #Opposite spin MP2 correlation energy density



## Examples
Provides some initial example scripts upon evaluation of the correlation energy or correlation energy density generator. 

- **main.py** : Simple script to run the generator and a corresponding PySCF MP2 calculation to compare the correlation energies. 


# Reference
MP2 correlation energy density for closed-shell systems derived from [S. Vuckovic, T. J. P. Irons, A. Savin, A. M. Teale, and P. Gori-Giorgi, “Exchange-Correlation Functionals via Local Interpolation along the Adiabatic Connection”, Journal of Chemical Theory and Computation 12, 2598-2610 (2016)](https://pubs.acs.org/doi/10.1021/acs.jctc.6b00177).

# Contact
Author: [Elias-Py09](https://github.com/Elias-Py09) 

Group-Homepage: [Vuckovic group](https://www.stefanvuckovic.com/)

Version: 24.09.2024