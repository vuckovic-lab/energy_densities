'''

Exact exchange based energy density

Documentation with examples are found on the github repository:
https://github.com/vuckovic-lab/energy_densities.git

Available functions:

    >>> import exact_exchange as ex
    >>> ex_exact.ex_args()           #Extraction of necessary arguments 
    >>> ex_exact.ex_kwargs()         #Extraction of optional arguments 
    >>> ex_exact.ex_ref()            #Calculation of the reference exchange energy
    >>> ex_exact.ex_rhf()            #Exact exchange energy density for closed-shell
    >>> ex_exact.ex_uhf()            #Exact exchange energy density for open-shell
'''

#Importing all relevant functions for the evaluation
from ex_exact.ex_ref import ex_ref
from ex_exact.ex_args_kwargs import ex_args, ex_kwargs
from ex_exact.ex_funcs import ex_rhf, ex_uhf


#Defining an evaluation generator
def ex_hf_gen(mf,mol,grids=3,batch_size=0,DF=None,optimal_contract=0,verbose=False):
    """Evaluation of the exact exchange energy density
    
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
    """
    
    #Extracting (optional) arguments for the local slope evaluation
    argss    = ex_args(mf,mol,grids=grids,DF=DF)
    kwargs  = ex_kwargs(batch_size=batch_size,optimal_contract=optimal_contract,verbose=verbose)

    dm = argss[0]     #Extract density matrix from args
    
    #Check if restricted or unrestricted evaluation using density matrix
    if len(dm.shape) < 3:  #Check shape of dm for restricted evaluation
        
        Ex= ex_rhf(*argss,*kwargs) 
        Ex_energy = Ex.energy
        ex_array = Ex.array
        
        return [Ex_energy,ex_array]
    
    else: 
       
        Ex = ex_uhf(*argss,*kwargs)
        Ex_energy = Ex.energy
        ex_array = Ex.array
        
        return [Ex_energy,ex_array]







