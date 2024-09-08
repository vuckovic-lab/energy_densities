''' Script to obtain PySCF based reference MP2 correlation energy.

Author: Elias Polak
Version: 15.11.2023'''

from pyscf import mp
from pyscf.data import elements


#MP2 correlation energy evaluation
def ec_ref_eval(mf, mol, DF=False, Abasis=None, frozen_core=False, spinorb=False, num_core='auto'):
    '''Reference correlation energy from a PySCF-based MP2 calculation 
    
    Input: 
    mf                  : Kernel of the RKS/RHF-calculation
    mol                 : GTO molecular structure
    DF=False            : Density fitting option for the MP2 calculation 
    Abasis=None         : String specifying the auxiliary basis set
    frozen_core=False   : Frozen core orbital option
    spinorb = False     : Specification of the spin orbital usage of frozen core orbitals, False for R/U is fine
    num_core = 'auto'   : Amount of frozen orbitals, 'auto' selects the core ones.
    
    
    Output: 
    Ec_ref      : Reference MP2 correlation energy
    Ec_ref_os   : Opposite spin based MP2 correlation energy
    Ec_ref_ss   : Same spin based MP2 correlation energy
    '''

    print()
    print('MP2 Reference calculation:')
    
    if DF: #Density fitting 
        
        if frozen_core: #Frozen core orbitals
            mp2 = mp.MP2(mf).density_fit(auxbasis=Abasis)
            
            if num_core=='auto': #Automatic frozen core orbitals
                num_core = num_core_orb(mol, spinorb)
            else: #Manually chosen frozen core orbitals
                num_core = num_core
                
            mp2.frozen = num_core
            mp2.kernel()
        
        else: #No frozen core 
            mp2 = mp.MP2(mf).density_fit(auxbasis=Abasis)
            mp2.kernel()

    else: #No density fitting
        
        if frozen_core: #Frozen core orbitals
            mp2 = mp.MP2(mf) 
            
            if num_core=='auto': #Automatic frozen core orbitals
                num_core = num_core_orb(mol, spinorb)
            else: #Manually chosen frozen core orbitals
                num_core = num_core
            
            mp2.frozen = num_core
            mp2.kernel()
            
        else: #No frozen core     
            mp2 = mp.MP2(mf) 
            mp2.kernel()
            
    
    Ec_ref      = mp2.e_corr    #Full reference correlation energy
    Ec_ref_os   = mp2.e_corr_os #Opposite based correlation energy
    Ec_ref_ss   = mp2.e_corr_ss #Same spin based correlation energy

    return Ec_ref, Ec_ref_os, Ec_ref_ss




#Extracting the number of core orbitals
def num_core_orb(mol, spinorb=False):
    '''Returns the number of core orbitals
    
    Input: 
    mol             : GTO molecular structure
    spinorb = False : Specification of the spin orbital usage, False for R/U, True for GMP2, GCCSD, etc.
    
    Output:
    num_core        : Number of core orbitals in the molecule'''
    
    #Number of core orbitals:
    num_core = elements.chemcore(mol, spinorb)
    
    return num_core   