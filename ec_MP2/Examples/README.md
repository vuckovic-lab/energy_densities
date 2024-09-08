# local slope examples
This directory contains several examples of the local slope evaluation and its usage. They can be directly run or used as a template to perform further analysis. Inside the *Geometries* directory are some geometry samples as *.txt* files, which are read by the example python scripts. 

## Some notes on the code

- The local slope functionalities are employed with the import command (see the content section of the main [README](https://github.com/vuckovic-lab/local_slope#content) file for a list of available functions):

        import local_slope as ls
    

- Defining *kwargs* is not necessary, but can be included if desired:

        batch_size = 0           : Batch size to run the evaluation on, 0 is for no parallelization
        DF = True                : Density fitting option
        verbose = False          : Additional printings of time and memory statements
        optimal_contract = False : Optimized contraction algorithm for the opt_einsum summation path
        max_num_array = None     : Maximum number of elements in a temporary array for the optimal contraction path, 
                                   requires a postive integer (671.000.000 for max 5GB of memory size per array)
        frozen_core = False      : Frozen core orbital option
        spinorb = False          : Specification of the spin orbital usage of frozen core orbitals, 
                                    False for R/U is fine
        num_core = 'auto'             : Amount of frozen orbitals, 'auto' selects the core ones.
        kappa = 'inf'            : Laplace transform regularization parameter for the doubles amplitudes,
                                   'inf' for no regularization (original MP2 expression)


- Reading the geometry from a *.txt* file:

        atom_geom='Geometries/atom_geom.txt'   
        with open(atom_geom, 'r') as f:
            atom_geom = f.read()

- Necessary arguments for the local slope evaluation are extracted from the [*ec_mp2_args* ](https://github.com/vuckovic-lab/local_slope#Input-functionalities) python function:

        args = ec_mp2_args(mf,mol,Abasis,grids=3)

- Finally, a local slope class object from a closed shell calculation can be obtained via

        Ec=ls.ec_mp2_cs(*args,*kwargs) 

  or only the corresponding opposite spin part:

        Ec_os=ls.ec_mp2_os(*args,*kwargs)

            

# Content

- **main.py** : Simple script to run the local slope evaluation and a corresponding MP2 reference calculation to compare the correlation energies. 

- **local_slope_batches.py** 

    Iterative analysis of the local slope evaluation for different batch sizes. 
    
        verbose = True
    
    is necessary to visualise the analysis. Upon specification of the other optional arguments, a sequence of batch sizes is created, dependent on the total number of grid points and the user defined amount of sequence elements:

        batch_size_steps = 50  

    The script returns the results in terms of time [sec] and memory usage [GB] in a *.csv* file and also as a plot against the sequence of batch sizes.  

- **local_slope_inter.py**   

    Local slope based interaction energy calculation. Reads from the *Geometry* directory the **atom_geom_complex.txt** complex geometry and the **atom_geom_iso.txt** isolated species geometry. 

    Contains a unit conversions script from atomic units to kcal/mol:

        E_int = kcal(int)

    Returns the absolute error of the locals slope based correlation energy as well as the interaction energy error in kcal/mol.
    

- **local_slope_plotting.py** 

    Script to plot the local slope array on a pyscf.dft generated grid with specified grid level or on a user defined grid. The additional python function 
        
        coords_u = user_coord(starts, stop, steps)
        
    creates a 3D grid with points only along the principal axis of the chemical system.  

- **interaction_local_slope.py** 

    Evaluates the interaction local slope (ILS) for a given complex and on a user defined grid. Molecular geometry is created for a diatomic system:

        mol_AB, mol_A, mol_B = dimer_build(atomA,atomB,distance,basis)
    
    where the atoms are specified with a string value and their interatomic distance constructs the complex geometry with equilibrium being at the origin. 
    
    The local slope and ILS array is plotted for different contributions of opposite and same spin configurations within the closed shell evaluation.