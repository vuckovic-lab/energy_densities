''' Test script for the exact exchange energy evaluation'''


#Importing pyscf modules
from pyscf import gto, dft
from pyscf.dft import numint

#Importing packages
import numpy as np
import ex_exact as ex

DF=False           #Density fitting argument
verbose=True       #Additional printing of computation steps and time results
batch_size=5000    #Batch size parallelization of grid points

kwargs = ex.ex_kwargs(batch_size,DF,verbose) #Extracting optional arguments


print()
print('Starting exact exchange test script for a simple Neon atom')
print()

atom_geom='''Ne 0 0 0 '''

basis = 'def2-tzvp'
Abasis = 'def2tzvpri' #aux basis for MP2 correlation energy
mol = gto.M(atom=atom_geom, basis=basis)
grid_level =  3 #Set grid level
mf = dft.RKS(mol) 
mf.grids.level = grid_level  # Grid level
mf.xc = 'pbe,pbe'  

print('Obtaining orbitals:')
mf.kernel()
print()

args = ex.ex_args(mf,mol,Abasis,grids=grid_level) #Extracting input arguments

#Exchange energy evaluation
Ex=ex.ex_eval(*args,*kwargs)
Ex_ref=ex.ex_ref_eval(mf)


print()
print('Results:')
print('Exact exchange based energy: %s' % Ex.energy)
print("Reference exchange energy: %s" % Ex_ref)
print('Absolute difference: %s' % np.abs(Ex.energy-Ex_ref))
print()
print('Finished successfully the test exchange energy evaluation. You are good to go!')



