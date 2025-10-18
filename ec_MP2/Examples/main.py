'''Test run script for the MP2 correlation energy density generator'''

from pyscf import gto, mp, dft
from ec_MP2 import ec_mp2_gen


print()
print('Starting MP2 correlation energy density test script for a simple Neon atom')
print()

#Parameters
basis = 'def2svp'
Abasis = 'def2svpri' 
xc_fun = 'hf'

#Atom specifications
atom_geom = 'Li 0 0 0'
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

#MP2
mp2 = mp.MP2(mf)
mp2.kernel()


#MP2 package energy evaluation
Ec_energies,Ec_arrays = ec_mp2_gen(mf,mol,DF=Abasis)
print(f'MP2 correlation energy: {Ec_energies[0]+Ec_energies[1]}')
print(f'MP2-OS correlation energy: {Ec_energies[0]}')
print(f'MP2-SS correlation energy: {Ec_energies[1]}')

#Printing accuracy results
print('Accuracy of evaluation:')
print(f'MP2-OS Error: {abs(Ec_energies[0]-mp2.e_corr_os)}')
print(f'MP2-SS Error: {abs(Ec_energies[1]-mp2.e_corr_ss)}')
print(f'MP2 Error: {abs(Ec_energies[0]+Ec_energies[1]-mp2.e_corr)}')

print()
print('Successfully finished generator test. You are good to go!')
print()