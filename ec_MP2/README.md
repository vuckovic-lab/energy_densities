# MP2 Correlation Energy Density Generator
This is the project repository for the evaluation of the MP2 correlation energy density. It contains functionalities to obtain the MP2 correlation energy and energy density function. 

Version (18/10/2025)

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
        ec_mp2_cs()      #Closed-shell MP2 correlation energy density 
        ec_mp2_os()      #Opposite-spin (os) MP2 correlation energy density 
        ec_ump2_os()     #Opposite-spin UMP2 correlation energy density 
        ec_ump2_ss()     #Same-spin (ss) UMP2 correlation energy density 

        
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
        grids=3      : pyscf.dft generated grid with a specified grid level or a user defined 3D grid
        DF=None      : Density fitting string specifying the auxiliary basis set (None for no density fitting)
        
        Output:
        args    : List of output objects, which are needed for the ec_mp2 functions
                0:dm, 1:mol, 2:Amol, 3:mo_coeff, 4:mo_occ, 5:mo_energies, 6:coords, 7:weights
   
- **ec_mp2_kwargs**

    Extracting the optional arguments for the ec_mp2 class function. Make sure to use * to unpack.
    
        Input:
        batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
        kappa = 'inf'           : Laplace transform regularization parameter for the doubles amplitudes,
                                'inf' for no regularization (original MP2 expression)
        optimal_contract = 0    : Optimized contraction algorithm for the opt_einsum summation path
                                (0 for no opt_einsum, else positive integer of elements in a temporary array 
                                for the optimal contraction path).
        frozen_core = 0         : Frozen core orbital option (0 for no frozen core, 'auto' selects core orbitals
                                automatically, else positive integer below number of orbitals)
                                ('inf' for no regularization, else positive integer)
        verbose = False         : Additional printings of time and memory statements
        
        Output:
        kwargs                  : List of optional arguments
                                0:batch_size, 1:kappa, 2:optimal_contract, 3:frozen_core, 4:verbose

## ec_MP2 functions
The script *ec_mp2* containts the python class functions for the MP2 energy density generator

- **ec_mp2_cs(\*args,\*kwargs)** 

    Evaluation of the MP2 correlation energy density for a closed shell system. Density fitting, parallelization, frozen core orbitals or kappa regularization are optional arguments. Requires a RKS or RHF Pyscf class object.

       *args* 
        dm          : Density matrix from an closed-shell scf calculation (#basis,#basis)
        mol         : gto molecular structure incorporating the basis set
        Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        mo_coeff    : Coefficient matrix of the atomic orbitals (#basis,#basis)
        mo_occ      : Occupation numbers (#basis,)
        mo_energies : Orbital energies (#basis,)
        coords      : Grid coordinates (#coords,3)
        weights     : Grid weights (#coords,1)
        
        **kwargs
        batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
        kappa = 'inf'           : Laplace transform regularization parameter for the doubles amplitudes
                                  ('inf' for no regularization, else positive integer).
        optimal_contract = 0    : Optimized contraction algorithm option (0 by default, else positive integer
                                  representing number of elements in temporary array for the optimal contraction path).
        frozen_core = 0         : Frozen core orbital option (0 by default, 'auto' selects core orbitals
                                  automatically, else positive integer below number of orbitals).
        verbose = False         : Additional printings of time and memory statements
        

- **ec_mp2_os(\*args,\*kwargs)** 

    Evaluation of the opposite-spin based MP2 correlation energy density. 
    Density fitting, batchwise parallelization, frozen core orbitals or kappa 
    regularization are optional arguments. Requires a RKS or RHF Pyscf class object.

        *args* 
        dm          : Density matrix from a closed-shell scf calculation (#basis,#basis)
        mol         : gto molecular structure incorporating the basis set
        Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        mo_coeff    : Coefficient matrix of the atomic orbitals (#basis,#basis)
        mo_occ      : Occupation numbers (#basis,)
        mo_energies : Orbital energies (#basis,)
        coords      : Grid coordinates (#coords,3)
        weights     : Grid weights (#coords,1)
        
        **kwargs
        batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
        kappa = 'inf'           : Laplace transform regularization parameter for the doubles amplitudes
                                  ('inf' for no regularization, else positive integer).
        optimal_contract = 0    : Optimized contraction algorithm option (0 by default, else positive integer
                                  representing number of elements in temporary array for the optimal contraction path).
        frozen_core = 0         : Frozen core orbital option (0 by default, 'auto' selects core orbitals
                                  automatically, else positive integer below number of orbitals).
        verbose = False         : Additional printings of time and memory statements

## ec_MP2 functions
The script *ec_ump2* containts the python class functions for the UMP2 energy density generator

- **ec_ump2_os(\*args,\*kwargs)** 
    Evaluation of the opposite-spin based UMP2 correlation energy densities. Density fitting, batchwise parallelization, frozen core orbitals or kappa regularization are optional arguments. Requires a UKS or UHF Pyscf class object.

        *args* 
        dm          : Density matrix from an open-shell scf calculation (2,#basis,#basis)
        mol         : gto molecular structure incorporating the basis set
        Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        mo_coeff    : Coefficient matrix of the atomic orbitals (2,#basis,#basis)
        mo_occ      : Occupation numbers (2,#basis,)
        mo_energies : Orbital energies (2,#basis,)
        coords      : Grid coordinates (#coords,3)
        weights     : Grid weights (#coords,1)
        
        **kwargs
        batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
        kappa = 'inf'           : Laplace transform regularization parameter for the doubles amplitudes
                                  ('inf' for no regularization, else positive integer).
        optimal_contract = 0    : Optimized contraction algorithm option (0 by default, else positive integer
                                  representing number of elements in temporary array for the optimal contraction path).
        frozen_core = 0         : Frozen core orbital option (0 by default, 'auto' selects core orbitals
                                  automatically, else positive integer below number of orbitals).
        verbose = False         : Additional printings of time and memory statements

- **ec_ump2_ss(\*args,\*kwargs)** 
    Evaluation of the same-spin based UMP2 correlation energy densities. Density fitting, batchwise parallelization, frozen core orbitals or kappa regularization are optional arguments. Requires a UKS or UHF Pyscf class object.

        *args* 
        dm          : Density matrix from an open-shell scf calculation (2,#basis,#basis)
        mol         : gto molecular structure incorporating the basis set
        Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        mo_coeff    : Coefficient matrix of the atomic orbitals (2,#basis,#basis)
        mo_occ      : Occupation numbers (2,#basis,)
        mo_energies : Orbital energies (2,#basis,)
        coords      : Grid coordinates (#coords,3)
        weights     : Grid weights (#coords,1)
        
        **kwargs
        batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
        kappa = 'inf'           : Laplace transform regularization parameter for the doubles amplitudes
                                  ('inf' for no regularization, else positive integer).
        optimal_contract = 0    : Optimized contraction algorithm option (0 by default, else positive integer
                                  representing number of elements in temporary array for the optimal contraction path).
        frozen_core = 0         : Frozen core orbital option (0 by default, 'auto' selects core orbitals
                                  automatically, else positive integer below number of orbitals).
        verbose = False         : Additional printings of time and memory statements
        
        

## Examples
Provides some initial example scripts upon evaluation of the correlation energy or correlation energy density generator. 

- **main.py** : Simple script to run the generator and a corresponding PySCF MP2 calculation to compare the correlation energies. 


# Reference
- MP2 correlation energy density for closed-shell systems derived from [S. Vuckovic, T. J. P. Irons, A. Savin, A. M. Teale, and P. Gori-Giorgi, “Exchange-Correlation Functionals via Local Interpolation along the Adiabatic Connection”, Journal of Chemical Theory and Computation 12, 2598-2610 (2016)](https://pubs.acs.org/doi/10.1021/acs.jctc.6b00177).

- **ec_MP2** package employed in [E. Polak, H. Zhao, S. Vuckovic, "Real-space machine learning of correlation density functionals", ChemRxiv (2024)](https://doi.org/10.26434/chemrxiv-2024-zk6hp-v2) *(submitted to Nature Communications)*.


# Contact
Author: [Elias-Py09](https://github.com/Elias-Py09) 

Group-Homepage: [Vuckovic group](https://www.stefanvuckovic.com/)

Version: 18.10.2025