# exact_exchange
Project repository for the evaluation of the exact exchange -- Functionalities to obtain the exact exchange based exchange energy and energy density function. 

- Exact exchange is taken from:
    > [S. Vuckovic, T. J. P. Irons, A. Savin, A. M. Teale, and P. Gori-Giorgi, “Exchange-Correlation Functionals via Local Interpolation along the Adiabatic Connection”, Journal of Chemical Theory and Computation 12, 2598-2610 (2016)](https://pubs.acs.org/doi/10.1021/acs.jctc.6b00177).

# Quickstart
- The repository can be cloned to the local directory using the *git clone* command. 

       git clone https://github.com/vuckovic-lab/energy_densities 

- Installation is possible with the *pip* command.

       pip install .

- The functions are directly called from the python files:

        from ex_exact.ex_args_kwargs import ex_args, ex_kwargs
        from ex_exact.ex_ref import ex_ref_eval
        from ex_exact.ex_funcs import ex_eval

- The available functions are

        ex_args()     #Extraction of necessary arguments 
        ex_kwargs()   #Extraction of optional arguments 
        ex_ref_eval() #Calculation of the PySCF HF exchange energy
        ex_eval()      #Exact exchange energy density generator

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

## Energy density generator
The main python module is initialize from the whole exact_exchange directory.

- **ex_hf_gen**

   Evaluation of the exact exchange energy density
    
        *args
        mf                      : SCF class calculation
        mol                     : GTO molecular structure
        
        *kwargs
        grids=3                 : pyscf.dft generated grid with a specified grid level or a user defined 3D grid.
        batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
        DF=None                 : Density fitting (DF) string specifying the auxiliary basis set (None for no DF)
        optimal_contract = 0    : Optimized contraction algorithm option (0 by default, else positive integer
                                        representing number of elements in temporary array for the optimal contraction path).
        verbose = False         : Additional printings of time and memory statements
        
        Output:
        Ex_energy   : Total exchange energy in a.u.
        ex_array    : Exchange energy density arrays evaluated on the given grid.
        
        Example:
        >>> basis = 'def2svp'
        >>> Abasis = 'def2svpri' 
        >>> grid_level = 4
        >>> xc_fun = 'hf'
        >>> atom_geom = 'Al 0 0 0'
        >>> spin = 1
        >>> charge = 0
        >>> mol = gto.Mole()
        >>> mol.atom  = atom_geom
        >>> mol.basis = basis
        >>> mol.spin = spin 
        >>> mol.charge = charge
        >>> mol.build()
        >>> mf = dft.UKS(mol)
        >>> mf.grids.level = grid_level
        >>> mf.xc = xc_fun
        >>> mf.kernel()
        >>> Ex_energy,ex_array = ex_hf_gen(mf,mol,grids=grid_level,DF=Abasis)
        >>> print('Exchange energy: %s' % Ex_energy)

## Input functionalities
The script *ex_args_kwargs* contains two main functions that extract necessary and optional input arguments.

- **ex_args** 

    Extracts the input arguments from a SCF kernel calculations for the generator. 
    Also creates or reads optionally a 3D mesh grid and adds artificial uniform weights.
    Make sure to use * to unpack.
    
        Input: 
        mf           : SCF class calculation
        mol          : GTO molecular structure
        Abasis       : String specifying the auxiliary basis set
        grids=3      : pyscf.dft generated grid with a specified grid level or a user defined 3D grid.
        DF=None      : Density fitting string specifying the auxiliary basis set (None for no density fitting)
        
        Output:
        args    : List of output objects, which are needed for the ec_mp2 functions
                0:dm, 1:mol, 2:Amol, 3:coords, 4:weights
        
- **ex_kwargs**

    Collects the optional arguments for the generator. Make sure to use * to unpack.
        
        Input:
        batch_size = 0           : Batch size to run the evaluation on, 0 is for no parallelization
        DF = True                : Density fitting option
        optimal_contract=0       : Optimized contraction algorithm for the opt_einsum summation path
                                (0 for no opt_einsum, else positive integer of elements in a temporary array 
                                for the optimal contraction path). 
        verbose = False          : Additional printings of time and memory statements
        
        Output:
        kwargs                   : List of optional arguments
                                0:batch_size, 1:DF, 2:verbose

## Reference evaluation
The script *ex_ref* containts the function to evaluate the PySCF based HF exchange energy.

- **ex_ref** 
        
    Exact evaluation of the exchange energy directly from the density matrix.
        
        Input: 
        mf  : Kernel of a RKS/RHF-Calculation
        
        Output:
        Ex  : Exchange energy value

## HF exact exchange energy density generator
The script *ex_funcs* contains the python class functions for the HF exact exchange energy density generator of closed- and open-shell systems. 

- **ex_rhf(\*args,\*kwargs)** 

    Evalation of the exact exchange based energy density for closed-shell (restricted) systems.
    Density fitting or batchwise parallelization are optional arguments.

        *args* 
        dm      : Density matrix from a scf calculation (#basis,#basis)
        mol     : gto molecular structure incorporating the basis set.
        Amol    : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        
        coords  : Grid coordinates (#coords,3)
        weights : Grid weights (#coords,3)
        
        **kwargs
        batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
        optimal_contract = 0    : Optimized contraction algorithm option (0 by default, else positive integer
                                  representing number of elements in temporary array for the optimal contraction path).
        verbose = False         : Additional printings of time and memory statements
        
- **ex_uhf(\*args,\*kwargs)** 

    Evalation of the exact exchange based energy density for open-shell (restricted) systems.
    Density fitting or batchwise parallelization are optional arguments.

        *args* 
        dm      : Density matrix from a scf calculation (2,#basis,#basis)
        mol     : gto molecular structure incorporating the basis set.
        Amol    : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        
        coords  : Grid coordinates (#coords,3)
        weights : Grid weights (#coords,3)
        
        **kwargs
        batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
        optimal_contract = 0    : Optimized contraction algorithm option (0 by default, else positive integer
                                  representing number of elements in temporary array for the optimal contraction path).
        verbose = False         : Additional printings of time and memory statements
                


## Exact-exchange energy density generators
The following class functions are within the *ex_funcs.py* script. They provide the evaluation of closed-shell (*ex_rhf*) and open-shell (*ex_uhf*) systems. 

- **ex_rhf(\*args,*\kwargs)**

    Evalation of the exact exchange based energy density for closed-shell (restricted) systems. Density fitting or batchwise parallelization are optional arguments.

        *args* 
        dm      : Density matrix from a scf calculation (#basis,#basis)
        mol     : gto molecular structure incorporating the basis set.
        Amol    : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        
        coords  : Grid coordinates (#coords,3)
        weights : Grid weights (#coords,3)
        
        **kwargs
        batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
        optimal_contract = 0    : Optimized contraction algorithm option (0 by default, else positive integer representing number of elements in temporary array for the optimal contraction path).
        verbose = False         : Additional printings of time and memory statements


- **ex_uhf(\*args,*\kwargs)**

    Evalation of the exact exchange based energy density for open-shell (unrestricted) systems. Density fitting or batchwise parallelization are optional arguments.

        *args* 
        dm      : Density matrix from a scf calculation (#basis,#basis)
        mol     : gto molecular structure incorporating the basis set.
        Amol    : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        
        coords  : Grid coordinates (#coords,3)
        weights : Grid weights (#coords,3)
        
        **kwargs
        batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
        optimal_contract = 0    : Optimized contraction algorithm option (0 by default, else positive integer representing number of elements in temporary array for the optimal contraction path).
        verbose = False         : Additional printings of time and memory statements


## Examples
Provides an example scripts upon evaluation of the exact exchange energy density. 

- **main.py** : Simple script to run the exact exchange evaluation and a corresponding reference calculation to compare the exchange energies. 

# Contact
Author: [Elias-Py09](https://github.com/Elias-Py09) 

Group-Homepage: [Vuckovic group](https://www.unifr.ch/chem/en/research/groups/vuckovic-group/)

Version: 18.10.2025
