'''Python scripts to extract and collect fixed and optional arguments for the 
correlation energy density evaluation with the local slope modelling. '''


from pyscf import dft, scf, df
import numpy as np 


#Extracting necessary input args to a list
def ec_mp2_args(mf,mol,grids=3,DF=None):
    '''Extracts the input arguments from a SCF kernel calculations for ec_mp2 evaluation. 
    Also creates or reads optionally a 3D mesh grid and adds artificial uniform weights.
    Make sure to use * to unpack.
    
    Input: 
    mf           : SCF class calculation
    mol          : GTO molecular structure
    grids=3      : pyscf.dft generated grid with a specified grid level or a user defined 3D grid
    DF=None      : Density fitting string specifying the auxiliary basis set (None for no density fitting)
    
    Output:
    args    : List of output objects, which are needed for the ec_mp2 functions
              0:dm, 1:mol, 2:Amol, 3:mo_coeff, 4:mo_occ, 5:mo_energies, 6:coords, 7:weights'''
    
    
    dm          = mf.make_rdm1()                     #Density matrix (#basis,#basis)
    mo_coeff    = mf.mo_coeff                        #Coefficient matrix of the atomic orbitals (#basis,#basis)
    mo_occ      = mf.mo_occ                          #Occupation numbers (#basis,)
    mo_energies = mf.mo_energy                       #Orbital energies (#basis,)
    if DF is not None:
        Amol    = df.addons.make_auxmol(mol, DF)     #Auxiliary basis set generated molecular structure
    else:
        Amol    = None
        
    #Checking the grids input:
    if isinstance(grids,int): #Specified grid level
        if grids < 1 or grids > 10: #Check for grid level integer value
            print('Grid level for the Becke-generated dft grid needs to be an integer between 0 and 10')
            quit()
        else: #Apply grid chosen grid level depending on the mf object
            if isinstance(mf,dft.rks.RKS) or isinstance(mf,dft.uks.UKS): #dft.RKS/UKS class object
                if mf.grids.level == grids: #Check if chosen grid level is used for pyscf calculation
                    coords      = mf.grids.coords       #Grid coordinates    (#coords,3)
                    weights     = mf.grids.weights      #Grid weights        (#coords,3)           
                else: #User defined grid level generated dft grid
                    grid = dft.Grids(mol)
                    grid.level = grids
                    grid.build(mf=mf)
                    coords=grid.coords
                    weights=grid.weights
                      
            elif isinstance(mf,scf.hf.RHF) or isinstance(mf,scf.uhf.UHF): 
                #SCF.RHF/UHF class object requires user defined generated dft grid
                grid = dft.Grids(mol)
                grid.level = grids
                grid.build(mf=mf)
                coords=grid.coords
                weights=grid.weights  
            else: #mf object wrongly specified
                print(''' 'mf' needs to be either a scf RHF/UHF or a dft RKS/UHF class object''')
                quit()
    elif isinstance(grids,np.ndarray) and grids.shape[1]==3: #grids is a user specified 3D grid
        coords = grids
        weights_value = 1.0 #Uniform grid weights
        weights =  np.ones(coords.shape[0]) * weights_value
        
    else: #Wrong grids specification
        print(''' 'grids' needs to be either an integer for the grid level or a user defined 3D grid. ''')
        quit()
   
    #Combine everything to a list
    args = [dm,mol,Amol,mo_coeff,mo_occ,mo_energies,coords,weights]
    
    return args
    
#Extracting optional kwargs to a list
def ec_mp2_kwargs(batch_size=0,kappa='inf',optimal_contract=0,frozen_core=0,verbose=False):
    '''Extracting the optional arguments for the ec_mp2 class function. Make sure to use * to unpack.
    
    Input:
    batch_size = 0           : Batch size to run the evaluation on, 0 is for no parallelization
    kappa = 'inf'            : Laplace transform regularization parameter for the doubles amplitudes,
                               'inf' for no regularization (original MP2 expression)
    optimal_contract = 0     : Optimized contraction algorithm for the opt_einsum summation path
                               (0 for no opt_einsum, else positive integer of elements in a temporary array 
                               for the optimal contraction path).
    frozen_core = 0          : Frozen core orbital option (0 for no frozen core, 'auto' selects core orbitals
                               automatically, else positive integer below number of orbitals)
                               ('inf' for no regularization, else positive integer)
    verbose = False          : Additional printings of time and memory statements
    
    Output:
    kwargs                   : List of optional arguments
                               0:batch_size, 1:kappa, 2:optimal_contract, 3:frozen_core, 4:verbose'''
    
    kwargs = [batch_size,kappa,optimal_contract,frozen_core,verbose]
    
    return kwargs
