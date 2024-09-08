'''Python scripts to extract and collect fixed and optional
arguments for the exact exchange based energy density evaluation 

Author: Elias Polak
Version: 24.11.2023'''


from pyscf import dft, scf, df
import numpy as np 


#Extracting necessary input args to a list
def ex_args(mf,mol,Abasis,grids=3):
    '''Extracts the input arguments from a SCF kernel calculations for ex_eval function. 
    Also creates or reads optionally a 3D mesh grid and adds artificial uniform weights.
    Make sure to use * to unpack.
    
    Input: 
    mf           : SCF class calculation
    mol          : GTO molecular structure
    Abasis       : String specifying the auxiliary basis set
    grids=3      : pyscf.dft generated grid with a specified grid level or a user defined 3D grid.
    
    Output:
    args    : List of output objects, which are needed for the ec_mp2 functions
              0:dm, 1:mol, 2:Amol, 3:coords, 4:weights'''
    
    
    dm          = mf.make_rdm1()                     #Density matrix (#basis,#basis)
    Amol        = df.addons.make_auxmol(mol, Abasis) #Auxiliary basis set generated molecular structure
    
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
                else: #User defined generated dft grid
                    grid = dft.Grids(mol)
                    grid.level = grids
                    grid.build(mf=mf)
                    coords=grid.coords
                    weights=grid.weights
                      
            elif isinstance(mf,scf.hf.RHF) or isinstance(mf,scf.hf.UHF): 
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
        weights = abs(coords) #Uniform random weights
    else: #Wrong grids specification
        print(''' 'grids' needs to be either an integer for the grid level or a user defined 3D grid. ''')
        quit()
   
    #Combine everything to a list
    args = [dm,mol,Amol,coords,weights]
    
    return args
    
#Extracting optional kwargs to a list
def ex_kwargs(batch_size=0,DF=True,verbose=False):
    '''Extracting the optional arguments for the ex_eval class function. Make sure to use * to unpack.
    
    Input:
    batch_size = 0           : Batch size to run the evaluation on, 0 is for no parallelization
    DF = True                : Density fitting option
    verbose = False          : Additional printings of time and memory statements
    
    Output:
    kwargs                   : List of optional arguments
                               0:batch_size, 1:DF, 2:verbose'''
    
    kwargs = [batch_size,DF,verbose]
    
    return kwargs
