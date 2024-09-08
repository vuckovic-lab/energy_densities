'''Test run script for the MP2 correlation energy density generator'''

#Importing packages
import numpy as np
import ec_MP2 as ec

#Importing pyscf modules
from pyscf import gto, dft

DF=True                     #Density fitting argument
verbose=True            #Additional printing of computation steps and time results
optimal_contract=False      #opt_einsum contraction optimized path
max_num_array=int(670000000)     #Maximum number of elements in a temporary array for optimal contraction path
batch_size=int(5000)             #Batch size parallelization of grid points
frozen_core=True           #Option for frozen core orbitals
spinorb=False               #Usage of spin orbitals when freezing core orbitals
num_core='auto'             #Specification of number of core orbitals
kappa='inf'            #Kappa regularization parameter

kwargs = ec.ec_mp2_kwargs(batch_size,DF,verbose,optimal_contract,
                          max_num_array,frozen_core,spinorb,num_core,kappa) #Extracting optional arguments

print()
print('Starting MP2 correlation energy density test script for a simple Neon atom')
print()

atom_geom='''Ne 0 0 0 '''

basis = 'def2-tzvp'
Abasis = 'def2tzvpri' #aux basis for MP2 correlation energy
mol = gto.M(atom=atom_geom, basis=basis)
grid_level =  3 #Set grid level
mf = dft.RKS(mol) 
mf.grids.level = grid_level  # Grid level
mf.xc = 'hf'  

print('Obtaining orbitals:')
mf.kernel()
print()

args = ec.ec_mp2_args(mf,mol,Abasis,grids=grid_level) #Extracting input arguments

#Local slope evaluation
Ec=ec.ec_mp2_cs(*args,*kwargs)      #Closed shell
Ec_os=ec.ec_mp2_os(*args,*kwargs)   #Opposite spin  
Ec_ss=Ec.energy-Ec_os.energy        #Same spin
Ec_ref=ec.ec_ref_eval(mf,mol,DF,Abasis,frozen_core,spinorb,num_core) #Corresponding MP2 reference

print('Results:')
print('MP2 correlation energy: %s' % Ec.energy)
print('Reference MP2 (refMP2) correlation energy: %s' % Ec_ref[0])
print('Absolute difference: %s' 
      % np.abs(Ec.energy-Ec_ref[0]))
print('Opposite and same spin MP2 correlation energy: %s and %s' 
      % (Ec_os.energy,Ec_ss))
print('Opposite and same spin refMP2 correlation energy: %s and %s' 
      % (Ec_ref[1],Ec_ref[2]))
print('Opposite and same spin absolute difference: %s and %s' 
      % (np.abs(Ec_os.energy-Ec_ref[1]),np.abs(Ec_ss-Ec_ref[2])))

print()
print('Finished successfully the test. You are good to go!')