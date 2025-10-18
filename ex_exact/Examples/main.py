''' Test script for the exact exchange energy evaluation'''

from pyscf import gto, dft
import numpy as np
from ex_exact import ex_hf_gen,ex_ref

print()
print('Starting exact exchange test script for a simple Neon atom')
print()

basis = 'def2svp'
Abasis = 'def2universaljkfit' #aux basis for exchange DF
xc_fun = 'hf'

#Atom specifications
atom_geom = 'Al 0 0 0'
spin = 1
charge = 0

#Building mol object
mol = gto.Mole()
mol.atom  = atom_geom
mol.basis = basis
mol.spin = spin 
mol.charge = charge
mol.build()

#SCF resolution using HF
mf = dft.UKS(mol)
mf.xc = xc_fun
mf.kernel()

#Exchange energy evaluation
Ex_energy,ex_array=ex_hf_gen(mf,mol,DF=Abasis)
Ex_ref=ex_ref(mf)


print('Exact exchange based energy: %s' % Ex_energy)
print("Reference exchange energy: %s" % Ex_ref)
print('Absolute difference: %s' % np.abs(Ex_energy-Ex_ref))
print()
print('Finished successfully the test exchange energy evaluation. You are good to go!')


