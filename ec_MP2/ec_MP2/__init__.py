'''

Local slope modelling of the correlation energy

Documentation with examples are found on the github repository:
https://github.com/vuckovic-lab/energy_densities.git

Available functions:

    >>> import ec_MP2 as ec
    >>> ec.ec_mp2_args()    #Extraction of necessary arguments for the MP2 correlation energy density evaluation
    >>> ec.ec_mp2_kwargs()  #Extraction of optional arguments for the MP2 correlation energy density evaluation
    >>> ec.ec_ref_eval()    #Calculation of the corresponding reference correlation energy
    >>> ec.ec_mp2_cs()      #Closed-shell MP2 correlation energy density 
    >>> ec.ec_mp2_os()      #Opposite-spin (os) MP2 correlation energy density
    >>> ec.ec_ump2_os()     #Opposite-spin UMP2 correlation energy density 
    >>> ec.ec_ump2_ss()     #Same-spin (ss) UMP2 correlation energy density 
'''

#Importing all relevant functions for the evaluation
from ec_MP2.ec_args_kwargs import ec_mp2_args, ec_mp2_kwargs
from ec_MP2.ec_mp2 import ec_mp2_cs, ec_mp2_os
from ec_MP2.ec_ump2 import ec_ump2_os, ec_ump2_ss


#Defining an evaluation generator
def ec_mp2_gen(mf,mol,grids=3,batch_size=0,DF=None,kappa='inf',optimal_contract=0,
               frozen_core=0,verbose=False,i_iter=False):
    """Evaluation of the MP2 correlation energy density (local slope)
    
    *args
    mf                      : SCF class calculation
    mol                     : GTO molecular structure
    
    *kwargs
    grids=3                 : pyscf.dft generated grid with a specified grid level or a user defined 3D grid.
    batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
    DF=None                 : Density fitting (DF) string specifying the auxiliary basis set (None for no DF)
    kappa = 'inf'           : Laplace transform regularization parameter for the doubles amplitudes
                                ('inf' for no regularization, else positive integer).
    optimal_contract = 0    : Optimized contraction algorithm option (0 by default, else positive integer
                                representing number of elements in temporary array for the optimal contraction path).
    frozen_core = 0         : Frozen core orbital option (0 by default, 'auto' selects core orbitals
                                automatically, else positive integer below number of orbitals).
    verbose = False         : Additional printings of time and memory statements
    i_iter  = False         : Additional iteration option reducing memory requirement at the cost of time
    
    Output:
    [Ec_os_energy,Ec_ss_energy], : Tupel of correlation energies from both spin channel contributions in a.u.
    [Ec_os_array,Ec_ss_array]    : Tupel of correlation energy density arrays from both spin channel contributions
                                   evaluated on the given grid.
    
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
    >>> Ec_energies,Ec_arrays = ec_mp2_gen(mf,mol,grids=grid_level,DF=Abasis)
    >>> print('Opposite-spin based correlation energy: %s' % Ec_energies[0])
    >>> print('Same-spin based based correlation energy: %s' % Ec_energies[1])  
    """
    
    #Extracting (optional) arguments for the local slope evaluation
    argss    = ec_mp2_args(mf,mol,grids=grids,DF=DF)
    kwargs  = ec_mp2_kwargs(batch_size=batch_size,verbose=verbose,optimal_contract=optimal_contract,
    frozen_core=frozen_core,kappa=kappa)
    
    dm = argss[0]     #Extract density matrix from args
    mo_occ = argss[4] #Extract occupation numbers from args

    #Check if restricted or unrestricted evaluation using density matrix
    if len(dm.shape) < 3 and len(mo_occ.shape) < 2:  #Check shape of dm and mo_occ for restricted evaluation
            
        Ec_cs = ec_mp2_cs(*argss,*kwargs)
        Ec_os = ec_mp2_os(*argss,*kwargs)
        
        #Evaluation of same spin part
        Ec_ss_array  = Ec_cs.array - Ec_os.array
        Ec_ss_energy = Ec_cs.energy- Ec_os.energy
        
        return [Ec_os.energy,Ec_ss_energy],[Ec_os.array,Ec_ss_array]
        
    else: #Unrestricted evaluation
            
        Ec_os = ec_ump2_os(*argss,*kwargs)
        Ec_ss = ec_ump2_ss(*argss,*kwargs)
            
        return [Ec_os.energy,Ec_ss.energy],[Ec_os.array,Ec_ss.array]



