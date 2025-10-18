''' *** Main script for the UMP2 correlation energy density generator class ***

General formula is taken from:
S. Vuckovic, T. J. P. Irons, A. Savin, A. M. Teale, and P. Gori-Giorgi, “Ex-
change-Correlation Functionals via Local Interpolation along the Adiabatic
Connection”, Journal of Chemical Theory and Computation 12, 2598-2610 (2016).

The following python functionalities evaluate the UMP2 correlation energy density and the corresponding 
UMP2 correlation energy.

The evaluation requires a given grid in .xyz format and the output of a self-consistent HF or DFT treatment
using the Pyscf package. 

Notation:
phi     - molecular orbitals
chi     - atomic orbitals
psi     - auxiliary basis
eps     - orbital energies

ij      - occupied molecular orbitals
ab      - virtual molecular orbitals
klmn    - atomic orbitals
t       - auxiliary orbitals
p       - grid points
T_ijab  - Partial MP2 doubles amplitudes 
V_ijabp - orbital tensor integrals  '''

import numpy as np 
import jax.numpy as jnp
import time
import os
import sys
import opt_einsum as oe
import functools

from pyscf import df, gto, ao2mo, lib
from pyscf.dft import numint
from pyscf.data import elements
from concurrent.futures import ThreadPoolExecutor, as_completed

### Opposite-spin UMP2 correlation energy density for an open shell system ###
'''       
For open shell systems, the opposite-spin based formula reads:
    
e_c,os^{UMP2}(r_p)= 1/(2*rho(r_p))*sum_{ijab}T_{ijab}V_{ijabp}

where T_ijab is the partial MP2 doubles amplitude,

T_{ijab} = (<ij|ab>)/(eps_a+eps_b-eps_i-eps_j),

and V_{ijabp} is the orbital tensor integral,

V_{ijabp} = phi_i(r_p)phi_a(r_p)*int (phi_j(r')*phi_b(r'))/(r_p-r')dr'
  
with molecular orbital spin electron cases:
OS1: i,a = alpha ; j,b = beta 
OS1: i,a = beta  ; j,b = alpha. 
------------------------------------------------------------------------------------
Expansion to atomic orbitals yields a tensor multiplication notation per grid point:
m,n     -  atomic orbital index

V_{ijabp} = phi_i(r_p)phi_a(r_p)*sum_{mn}[C_{mj}*C_{nb} *int (chi_m(r')*chi_n(r'))/(r_p-r')dr']
          = phi_i(r_p)phi_a(r_p)*sum_{mn}[C_{mj}*C_{nb}*A_{mnp}]
------------------------------------------------------------------------------------          
 
'''
class ec_ump2_os:
    ''' Evaluation of the opposite-spin based UMP2 correlation energy densities (outputs for OS1 and OS2 case).
        Density fitting, batchwise parallelization, frozen core orbitals or 
        kappa regularization are optional arguments. 
        
        Requires a UKS or UHF Pyscf class object.
        
        Example:
        >>> kwargs = ec_mp2_kwargs()
        >>> atom_geom = 'Li 0 0 0'
        >>> basis = 'def2-svp' 
        >>> mol = gto.M(atom=atom_geom, basis=basis, spin=1, charge=0)
        >>> mf = dft.UKS(mol) 
        >>> mf.xc = 'hf'  
        >>> mf.kernel()
        >>> args = ec_mp2_args(mf,mol)
        >>> Ec_os=ec_ump2_os(*args, *kwargs)
        >>> print('Opposite-spin based correlation energy: %s' % Ec_os.energy)
        '''

    def __init__(self,dm,mol,Amol,mo_coeff,mo_occ,mo_energies,coords,weights,
                 batch_size=0, kappa='inf',optimal_contract=0, frozen_core=0, verbose=False):
        '''
        *args* 
        dm          : Density matrix from an open-shell scf calculation (2,#basis,#basis)
        mol         : gto molecular structure incorporating the basis set
        Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        mo_coeff    : Coefficient matrix of the atomic orbitals (2,#basis,#basis)
        mo_occ      : Occupation numbers (2,#basis,)
        mo_energies : Orbital energies (2,#basis,)
        coords      : Grid coordinates (#coords,3)
        weights     : Grid weights (#coords,1)
        
        **kwargs
        batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
        kappa = 'inf'           : Laplace transform regularization parameter for the doubles amplitudes
                                  ('inf' for no regularization, else positive integer).
        optimal_contract = 0    : Optimized contraction algorithm option (0 by default, else positive integer
                                  representing number of elements in temporary array for the optimal contraction path).
        frozen_core = 0         : Frozen core orbital option (0 by default, 'auto' selects core orbitals
                                  automatically, else positive integer below number of orbitals).
        verbose = False         : Additional printings of time and memory statements
        '''
        
        #Saving arguments of the class
        self.dm                 = dm
        self.mol                = mol
        self.Amol               = Amol
        self.mo_coeff           = mo_coeff
        self.mo_occ             = mo_occ
        self.mo_energies        = mo_energies
        self.coords             = coords
        self.weights            = weights
        self.batch_size         = batch_size
        self.verbose            = verbose
        self.optimal_contract   = optimal_contract
        self.frozen_core        = frozen_core
        self.kappa              = kappa
        
        #Extracting the number of virtual and occupied orbitals
        self.Nocc, self.Nvirt  = orb_occ_virt(self.mo_occ)    
        
        #========================================#
        #Checking input...             
        if self.Amol is None or isinstance(self.Amol,gto.mole.Mole): #DF option
            pass
        else:
            print('Amol needs to be a string name for the corresponding auxilliary basis set or "None" for no density fitting.')
            sys.exit()()
        
        if isinstance(self.optimal_contract,int)==False or self.optimal_contract < 0: #Einsum contraction option
            print('Optimal contraction argument has to be a positive integer or 0 for no optimal contraction.')
            sys.exit()()
        
        if isinstance(self.batch_size,int)==False or self.batch_size <0:  #Batch size option
            print('batch_size argument hast to be a positive integer or 0 for no parallelization.')
            sys.exit()()    
        
        if isinstance(self.frozen_core,int)==False or self.frozen_core < 0: #Number of frozen core orbitals
            if self.frozen_core == 'auto':
                pass
            else:
                print('''Number of frozen core orbitals must be a positive integer, 'auto' for automatic assignement or 0 for no frozen core approximation.''')
                sys.exit()()
       
        if isinstance(self.kappa, (int,float))==False or self.kappa < 0: #Kappa regularization 
            if self.kappa == 'inf':
                pass
            else:           
                print('''The regularization paramater kappa has to be a positive real number, 0 or 'inf'.''')
                sys.exit()()  

        #========================================#
        # Extra Functions for T_ijab and V_ijabp 
        
        #Two-body integrals of occupied and virtual alpha/beta molecular orbitals
        def two_body_integrals(mo_coeff,mol):
            '''Two-body integral computation of occupied and virtual molecular orbital functions: <ij|ab>.
            
            Input:
            mo_coeff          : Coefficient matrix of the atomic orbitals (2,#basis,#basis)
            mol               : gto molecular geometry
 
            Output: 
            two_integral_OS1  : Two body integral value for OS1 index case   (#alpha_occ_basis,#beta_occ_basis,#alpha_virt_basis,#beta_virt_basis)
            two_integral_OS2  : Two body integral value for OS2 index case   (#beta_occ_basis,#alpha_occ_basis,#beta_virt_basis,#alpha_virt_basis)'''


            #OS1 case:
            if self.frozen_core==0: #Check for frozen core orbital option
                #iajb integrals (iofree):
                two_integral_eval_OS1 = ao2mo.outcore.general_iofree(mol, (mo_coeff[0][:,:self.Nocc[0]], mo_coeff[0][:,self.Nocc[0]:],
            mo_coeff[1][:,:self.Nocc[1]],mo_coeff[1][:,self.Nocc[1]:]),compact=False).reshape(self.Nocc[0],self.Nvirt[0],self.Nocc[1],self.Nvirt[1])
            else:
                #iajb integrals (iofree):
                two_integral_eval_OS1 = ao2mo.outcore.general_iofree(mol, (mo_coeff[0][:,self.frozen_core:self.Nocc[0]], mo_coeff[0][:,self.Nocc[0]:],
            mo_coeff[1][:,self.frozen_core:self.Nocc[1]],mo_coeff[1][:,self.Nocc[1]:]),compact=False).reshape(self.Nocc[0]-self.frozen_core,self.Nvirt[0],self.Nocc[1]-self.frozen_core,self.Nvirt[1])
            
            #ijab integrals:
            two_integral_eval_OS1 = two_integral_eval_OS1.transpose((0,2,1,3));
            
            #OS2 case:
            if self.frozen_core==0: #Check for frozen core orbital option<
                #iajb integrals (iofree):
                two_integral_eval_OS2 = ao2mo.outcore.general_iofree(mol, (mo_coeff[1][:,:self.Nocc[1]], mo_coeff[1][:,self.Nocc[1]:],
            mo_coeff[0][:,:self.Nocc[0]],mo_coeff[0][:,self.Nocc[0]:]),compact=False).reshape(self.Nocc[1],self.Nvirt[1],self.Nocc[0],self.Nvirt[0])
            else:
                #iajb integrals (iofree):
                two_integral_eval_OS2 = ao2mo.outcore.general_iofree(mol, (mo_coeff[1][:,self.frozen_core:self.Nocc[1]], mo_coeff[1][:,self.Nocc[1]:],
            mo_coeff[0][:,self.frozen_core:self.Nocc[0]],mo_coeff[0][:,self.Nocc[0]:]),compact=False).reshape(self.Nocc[1]-self.frozen_core,self.Nvirt[1],self.Nocc[0]-self.frozen_core,self.Nvirt[0])
            
            #ijab integrals:
            two_integral_eval_OS2 = two_integral_eval_OS2.transpose((0,2,1,3));
            
            return two_integral_eval_OS1,two_integral_eval_OS2

        #Partial MP2 doubles amplitude T_ijab
        def part_mp2_amplitude(mol,mo_coeff,mo_energies,mo_occ,kappa):
            '''Evaluation of the partial MP2 doubles amplitude:
            
            T_{ijab}=(<ij|ab>)/(eps_a+eps_b-eps_i-eps_j)
            
            Input:
            mol           : gto molecular geometry
            mo_coeff      : Coefficient matrix of the atomic orbitals (2,#basis,#basis)
            mo_energies   : Orbital energies (2,#basis,)
            mo_occ        : Occupation numbers (2,#basis,)
            kappa = 'inf' : Laplace transform regularization parameter for the doubles amplitudes,
                            'inf' for no regularization (original MP2 expression)
            
            Output:
            T_eval_OS1    : Evaluated partial MP2 doubles amplitude for OS1 index case (#alpha_occ_basis,#beta_occ_basis,#alpha_virt_basis,#beta_virt_basis)
            T_eval_OS2    : Evaluated partial MP2 doubles amplitude for OS2 index case (#beta_occ_basis,#alpha_occ_basis,#beta_virt_basis,#alpha_virt_basis)
            '''
            
            #Orbital energies distribution
            orbs_energies_occ_alpha  = mo_energies[0][mo_occ[0] > 0] #Energies of occupied alpha orbitals
            orbs_energies_occ_beta   = mo_energies[1][mo_occ[1] > 0] #Energies of occupied beta orbitals
            orbs_energies_virt_alpha = mo_energies[0][mo_occ[0] ==0] #Energies of virtual alpha orbitals
            orbs_energies_virt_beta  = mo_energies[1][mo_occ[1] ==0] #Energies of virtual beta orbitals
            
            #Checking for frozen core orbital option
            if self.frozen_core==0:
                Nocc_fc = self.Nocc
            else:
                Nocc_fc = []
                for k in range(len(self.Nocc)):
                    Nocc_fc.append(self.Nocc[k]-self.frozen_core) #Updated number of occupied orbitals
                orbs_energies_occ_alpha = orbs_energies_occ_alpha[self.frozen_core:] #Updated list of orbital energies
                orbs_energies_occ_beta  = orbs_energies_occ_beta[self.frozen_core:] #Updated list of orbital energies
               
            #Denominator of orbital energies [i,j,a,b]:
            '''OS1 and OS2 case are equivalent here: eps_a+eps_b-eps_i-eps_j'''
            Eps=np.zeros([Nocc_fc[0],Nocc_fc[1],self.Nvirt[0],self.Nvirt[1]])

            for i in np.arange(Nocc_fc[0]):
                for j in np.arange(Nocc_fc[1]):
                    for a in np.arange(self.Nvirt[0]):
                        for b in np.arange(self.Nvirt[1]):
                            Eps[i,j,a,b] = orbs_energies_virt_alpha[a] + orbs_energies_virt_beta[b] - orbs_energies_occ_alpha[i] - orbs_energies_occ_beta[j]

            
            #Evaluate two body integrals
            T_ijab_OS1,T_ijab_OS2 = two_body_integrals(mo_coeff,mol)
            
            #Final partial MP2 doubles amplitude
            if kappa == 'inf':
                T_eval_OS1 = T_ijab_OS1 / Eps
                T_eval_OS2 = T_ijab_OS2 / Eps.transpose(1,0,3,2)
            else:
                T_eval_OS1 = T_ijab_OS1 / Eps * ((1-np.exp( - kappa * Eps)) ** 2)        
                T_eval_OS2 = T_ijab_OS2 / Eps.transpose(1,0,3,2) * ((1-np.exp( - kappa * Eps.transpose(1,0,3,2))) ** 2)        
            
            return T_eval_OS1, T_eval_OS2

        #Evaluation of virtual and occupied orbital functions
        def occ_virt_basis(mol, coords, mo_coeff):
            '''Extracting orbital functions from the atomic orbitals (basis set) and molecular orbital coefficients
            
            Input:
            mol         : gto molecular structure incorporating the basis set
            coords      : Grid coordinates (#coords,3)
            mo_coeff    : Coefficient matrix of the alpha/beta atomic orbitals (2,#basis,#basis)
            
            Output:
            mol_orb_occ  : Occupied alpha/beta molecular orbital functions evaluated on the grid (2,#occ-basis, #coords)
            mol_orb_virt : Virtual alpha/beta molecular orbital functions evaluated on the grid (2,#virt-basis,#coords)'''
                    
            #Atomic orbitals evaluated on the (batched) grid
            ao_value = numint.eval_ao(mol, coords, deriv=1)                   #  (#derivatives, #coords, #basis)
            
            #Molecular orbital evaluation
            mol_orb_alpha = np.einsum('ji,pj->ip',mo_coeff[0],ao_value[0])    #  alpha (#basis, #coords)
            mol_orb_beta  = np.einsum('ji,pj->ip',mo_coeff[1],ao_value[0])    #  beta  (#basis, #coords)

            #Check for frozen core orbital option
            if self.frozen_core==0:
                mol_orb_occ_alpha = mol_orb_alpha[:self.Nocc[0],:]            #   (#occ_basis, #coords)
                mol_orb_occ_beta  = mol_orb_beta[:self.Nocc[1],:]             #   (#occ_basis, #coords)
            else:
                mol_orb_occ_alpha = mol_orb_alpha[self.frozen_core:self.Nocc[0],:] #   (#occ_basis-#core_orb, #coords)
                mol_orb_occ_beta = mol_orb_beta[self.frozen_core:self.Nocc[1],:]   #   (#occ_basis-#core_orb, #coords)
                  
            #Extracting virtual orbitals
            mol_orb_virt_alpha = mol_orb_alpha[self.Nocc[0]:,:]               #   (#vir-basis, #coords)
            mol_orb_virt_beta = mol_orb_beta[self.Nocc[1]:,:]                 #   (#vir-basis, #coords)
         
            #Collecting results
            mol_orb_occ  =[mol_orb_occ_alpha,mol_orb_occ_beta]
            mol_orb_virt =[mol_orb_virt_alpha,mol_orb_virt_beta]
            
            return mol_orb_occ, mol_orb_virt
        
        #========================================#
        #Initialized printing:
        
        print('---------------------------------------------------------------')
        print(' Opposite-spin (os) UMP2 correlation energy density evaluation ')
        print('---------------------------------------------------------------')
        
        if self.verbose: #Parameter printing
            print('Evaluation parameters: ')
            print('Batch wise parallelization = ' + str(self.batch_size > 0))
            print('Density fitting = ' + str(isinstance(self.Amol,gto.mole.Mole)))
            print('Optimized einsum path = ' + str(self.optimal_contract > 0))
            print('Frozen core orbitals = ' + str(self.frozen_core))
            print('Initialising evaluation of the necessary components.')
            print(f'Number of [alpha,beta] occupied molecular orbitals is {self.Nocc}.')
            print(f'Number of [alpha,beta] virtual molecular orbitals is {self.Nvirt}.')
            self.contract_size = [] #Preallocate largest memory usage of contraction
               
        #Freezing the core orbitals
        if self.frozen_core=='auto': #Automatic frozen core orbitals
            self.frozen_core = int(num_core_orb(self.mol))
        if self.verbose:
            print()
            print('The first %s out of %s alpha and %s beta occupied orbitals are set frozen'
                  % (self.frozen_core,self.Nocc[0],self.Nocc[1]))
    
        #Starting evaluation
        #====================#
        if self.verbose: #Initial printing
            print()
            print('Starting evaluation...')
            print()
            
        start_time = time.time()
        
        # Partial MP2 doubles amplitude T_ijab (#occ_basis,#occ_basis,#virt_basis,#virt_basis) for OS1 and OS2 case
        T_OS1,T_OS2 = part_mp2_amplitude(self.mol,self.mo_coeff, self.mo_energies,
                               self.mo_occ,self.kappa)
        if self.verbose: #Saving size of T_ijab
            self.T_size1 = (T_OS1.size * T_OS1.itemsize) / (1024**3)  #in GB
            self.T_size2 = (T_OS2.size * T_OS2.itemsize) / (1024**3)  #in GB

        #Extraction of atomic orbital coefficients for alpha/beta electrons 
        if self.frozen_core==0:
            C_occ_alpha  = self.mo_coeff[0][:,:self.Nocc[0]]                 #Occupied orbitals (#basis, #occ_basis_alpha)                
            C_occ_beta   = self.mo_coeff[1][:,:self.Nocc[1]]                 #Occupied orbitals (#basis, #occ_basis_virt)                
        else:
            C_occ_alpha  = self.mo_coeff[0][:,self.frozen_core:self.Nocc[0]] #Occupied orbitals (#basis, #occ_basis_alpha-#core_orb)
            C_occ_beta   = self.mo_coeff[1][:,self.frozen_core:self.Nocc[1]] #Occupied orbitals (#basis, #occ_basis_virt-#core_orb)
        C_virt_alpha = self.mo_coeff[0][:,self.Nocc[0]:]                     #Virtual orbitals (#basis, #virt-basis_alpha)
        C_virt_beta = self.mo_coeff[1][:,self.Nocc[1]:]                      #Virtual orbitals (#basis, #virt-basis_alpha)
        
        #Collecting
        C_occ  = [C_occ_alpha,C_occ_beta]                                 #Occupied orbitals (alpha, beta)
        C_virt = [C_virt_alpha,C_virt_beta]                               #Virtual orbitals (alpha, beta)   
                
        if self.verbose: #Saving size of orbital coefficients
            self.C_occ_size = (C_occ[0].size * C_occ[0].itemsize+C_occ[1].size * C_occ[1].itemsize) / (1024**3) #in GB
            print(f'Memory usage of C_occ {self.C_occ_size:.8f} GB')
            self.C_virt_size = (C_virt[0].size * C_virt[0].itemsize+C_virt[1].size * C_virt[1].itemsize) / (1024**3) #in GB
            print(f'Memory usage of C_virt {self.C_virt_size:.8f} GB')
            self.mo_coeff_size = (self.mo_coeff[0].size * self.mo_coeff[0].itemsize+self.mo_coeff[1].size * self.mo_coeff[1].itemsize) / (1024**3) #in GB
            print(f'Memory usage of mo_coeff {self.mo_coeff_size:.8f} GB')
            print()
        
        #Batch seperation options:    
        if self.batch_size==0: #No parallelization 
            if self.verbose:
                print('No batch size chosen. Evaluation on the whole grid.')  
            self.coords_batches=self.coords 
        
        else: #Batch-wise parallelization           
            # Separate grid points into batches:
            self.coords_batches = [self.coords[i:i + self.batch_size] for i in range(0,self.coords.shape[0],self.batch_size)]  
            self.batches_amount = len(self.coords_batches) #Number of batches
            
            #Print Number of batches:
            if self.verbose:
                print('Separating grid into batches:')
                print('The number of batches is %s for a batch size of %s grid points.' % (self.batches_amount,self.batch_size))
                
            #Read the maximum amount of available cpu per task
            self.cpus_per_task = os.environ.get('SLURM_CPUS_PER_TASK')
            if self.cpus_per_task is not None:
                self.cpus_per_task = int(self.cpus_per_task)
            else:
                self.cpus_per_task = os.cpu_count()
                
            # Print the number of available CPU cores
            if self.verbose:
                print('Batch-wise evaluation on %s available CPU cores.' % self.cpus_per_task)
                
        #Check for density fitting option
        if self.Amol is not None: #Density fitting
            ''' Further expansion of V with a density fitted auxiliary basis set:

            A_{mnp}  = sum_{t}Q_{tmn}int psi_t(r')/(r_p-r')dr'
                     = sum_{t}Q_{tmn}I_tp 
             
            V_{ijab} = phi_i(r_p)*phi_a(r_p)*sum _{mn} C_{mj}C_{nb} A_{mnp}
                     = phi_i(r_p)*phi_a(r_p)*sum _{tmn} C_{mj}C_{nb}Q_{tmn}I_tp '''
            
            #Aux basis electrostatic integrals
            def aux_basis_int(coords, Amol):  
                ''' Tensor integral evaluation of the auxiliary basis set:
                    I_t(r)=int psi_t(r1)/(|r1-r|)dr1
                    evaluated using delta distributions with the Hartree potential
                    
                    Input: 
                    coords  : grid coordinates (N,3),
                    Amol    : gto molecular geometry with aux basis
                
                    Output:
                    integralvalue : Array of integral values for every aux basis at every grid point (#coords,#aux-basis)
                '''

                # Creating fake dirac delta charges for the hartree potential evaluation on the grid points
                fakemol = gto.fakemol_for_charges(coords, expnt=1e+16)
                mol1 = fakemol + Amol
                
                I = mol1.intor('int2c2e', shls_slice=(0,fakemol.nbas,fakemol.nbas,mol1.nbas))
                
                return I             
            
            #Function to evaluate of the correlation energy density array without DF
            def ec_density_eval_DF(T_OS1,T_OS2,mol,mo_coeff,C_occ,C_virt,df_coeff, Amol,coords):
                '''Opposite-spin based UMP2 correlation energy density evaluations from the open-shell formula:
                
                e_c,os^{MP2}(r_p)*rho(r_p)=-0.5* sum_{ijab}[V_{ijab}(r_p)T_{ijab}],
                
                Input:
                T_OS1       : Partial MP2 doubles amplitude (#occ_basis_alpha,#occ_basis_beta,#virt_basis_alpha,#virt_basis_beta)
                T_OS2       : Partial MP2 doubles amplitude (#occ_basis_beta,#occ_basis_alpha,#virt_basis_beta,#virt_basis_alpha)
                mol         : gto molecular structure incorporating the basis set.           
                mo_coeff    : Coefficient matrix of the alpha/beta atomic orbitals (2,#basis,#basis)
                C_occ       : Atomic orbital coefficients of alpha/beta occupied molecular orbitals (2,#basis,#occ_basis)
                C_virt      : Atomic orbital coefficients of alpha/beta virtual molecular orbitals (2,#basis,#virt_basis)
                df_coeff    : coefficient matrix from density fitting (#aux-basis,#basis,#basis)
                Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
                coords      : Given grid coordinates (#coords,3)
                
                Output:               
                ec_OS1      : Opposite-spin based UMP2 correlation energy density array for OS1 index case evaluated with no DF
                ec_OS2      : Opposite-spin based UMP2 correlation energy density array for OS2 index case evaluated with no DF'''
               
                #Extraction of molecular orbital functions
                mol_orb_occ, mol_orb_virt = occ_virt_basis(mol, coords, mo_coeff)
                
                #Extracting Hartree integral
                I_integral=aux_basis_int(coords, Amol)
                
                #OS1
                if self.optimal_contract==0: #No optimized contraction
                    #Sum all 
                    ec_OS1  = oe.contract('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[1],C_virt[1],df_coeff,I_integral,T_OS1)
                else: # Optimized contraction
                    #Sum all 
                    ec_OS1  = oe.contract('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[1],C_virt[1],df_coeff,I_integral,T_OS1,optimize='auto',memory_limit=self.optimal_contract)
                
                    
                if self.verbose: #Saving size of largest intermediate contraction arrays
                    contract_info = oe.contract_path('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[1],C_virt[1],df_coeff,I_integral,T_OS1)
                    self.contract_size.append(contract_info[1].largest_intermediate * 8 / (1024**3)) #in GB
            
                #OS2
                if self.optimal_contract==0: # No Optimized contraction
                    #Sum all 
                    ec_OS2  = oe.contract('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[0],C_virt[0],df_coeff,I_integral,T_OS2)
                else: # Optimized contraction
                    #Sum all 
                    ec_OS2  = oe.contract('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[0],C_virt[0],df_coeff,I_integral,T_OS2,optimize='auto',memory_limit=self.optimal_contract)
                    
                        
                if self.verbose: #Saving size of largest intermediate contraction arrays
                    contract_info = oe.contract_path('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[0],C_virt[0],df_coeff,I_integral,T_OS2)
                    self.contract_size.append(contract_info[1].largest_intermediate * 8 / (1024**3)) #in GB
                            
                return -0.5*ec_OS1, -0.5*ec_OS2
            
            #Density fitting coefficients for the auxliliary basis set (#aux-basis, #basis, #basis)
            self.df_coeff = obtain_df_coef(self.mol, self.Amol) 
            
            if self.verbose: #Saving size of auxiliary basis coefficients
                self.df_coeff_size = self.df_coeff.size * self.df_coeff.itemsize / (1024**3) #in GB
                print()
                print(f'Memory usage of the auxiliary basis set coefficients {self.df_coeff_size:.8f} GB')
                print()
            
            #Evaluating the correlation energy density on given grid
            if self.batch_size==0: #No batches specified
                if self.verbose:
                    print('Running correlation energy density evaluation on the full grid.')
                self.ec_OS1,self.ec_OS2 = ec_density_eval_DF(T_OS1,T_OS2,self.mol,self.mo_coeff,C_occ,C_virt,self.df_coeff, self.Amol,self.coords_batches)

            else: #Batch-wise parallelization of the evaluation
                if self.verbose:
                    print('Running correlation energy density evaluation on batch wise separated grid.')
                    
                #Prestore arguments:
                partial_ec_density_eval_DF = functools.partial(ec_density_eval_DF, T_OS1,T_OS2, self.mol, self.mo_coeff, C_occ, 
                                        C_virt,self.df_coeff, self.Amol)
                
                #Parallelization
                with ThreadPoolExecutor(max_workers=1) as executor: #Set 1 worker per thread for effective parallelization
    
                    #Preallocate lists
                    self.ec_per_batch_OS1 = [None]*len(self.coords_batches) 
                    self.ec_per_batch_OS2 = [None]*len(self.coords_batches) 
                    
                    #Parallelize with respect to number of CPUs
                    for idx in range(0, len(self.coords_batches), self.cpus_per_task):
                        if self.verbose:
                            print(f'Submission for batches {idx} until {idx+self.cpus_per_task}')
                        coords_batch_par=self.coords_batches[idx:idx+self.cpus_per_task] #Collect batches
                        #Submit execution to available CPUs per batch
                        futures_indx = {executor.submit(partial_ec_density_eval_DF,batch): indx for indx, batch in enumerate(coords_batch_par)}
                        for future in as_completed(futures_indx):
                            if self.verbose:
                                print(f'Finished batch evaluation {idx+futures_indx[future]}')
                            #Save results as jnp to ensure immutable    
                            self.ec_per_batch_OS1[futures_indx[future]+idx]=jnp.array(future.result()[0]) 
                            self.ec_per_batch_OS2[futures_indx[future]+idx]=jnp.array(future.result()[1])

                #Combination of the resulting correlation energy density batches
                self.ec_OS1 = jnp.concatenate(self.ec_per_batch_OS1, axis=0)
                self.ec_OS2 = jnp.concatenate(self.ec_per_batch_OS2, axis=0)
    
        else:   #Without density fitting
            ''' 
            Expansion to atomic orbitals yields the tensor multiplication from above.

            V_{ijabp} = phi_i(r_p)*phi_a(r_p)*sum _{mn} C_{mj}C_{nb} A_{mnp}
            '''
                
            #Tensor integral evaluation using Hartree potential 
            def tensor_int_fake(coords, mol):
                '''Tensor integral expression:
                A_{mn}(r)=int chi_m(r2)*chi_n(r2))/(|r-r2|)dr2
                evaluated using delta distributions with the Hartree potential
            
                Input: 
                coords  : grid coordinates (N,3),
                mol     : gto molecular geometry
            
                Output:
                integralvalue : Array of integral values for every atomic-basis at every grid point (#basis,#basis,N)
                '''

                # Creating fake dirac delta charges for the hartree potential evaluation on the grid points
                fakemol = gto.fakemol_for_charges(coords, expnt=1e+16)
                
                integralvalue = df.incore.aux_e2(mol, fakemol) #Evaluating the tensor integral as hartree potential
                
                return integralvalue    
            
            #Function to evaluate of the correlation energy density array without DF
            def ec_density_eval_noDF(T_OS1,T_OS2,mol,mo_coeff,C_occ,C_virt,coords):
                '''Opposite-spin based UMP2 correlation energy density evaluations from the open-shell formula:
                
                e_c,os^{MP2}(r_p)*rho(r_p)=-0.5* sum_{ijab}[V_{ijab}(r_p)T_{ijab}],
                
                Input:
                T_OS1       : Partial MP2 doubles amplitude (#occ_basis_alpha,#occ_basis_beta,#virt_basis_alpha,#virt_basis_beta)
                T_OS2       : Partial MP2 doubles amplitude (#occ_basis_beta,#occ_basis_alpha,#virt_basis_beta,#virt_basis_alpha)
                mol         : gto molecular structure incorporating the basis set.           
                mo_coeff    : Coefficient matrix of the alpha/beta atomic orbitals (2,#basis,#basis)
                C_occ       : Atomic orbital coefficients of alpha/beta occupied molecular orbitals (2,#basis,#occ_basis)
                C_virt      : Atomic orbital coefficients of alpha/beta virtual molecular orbitals (2,#basis,#virt_basis)
                coords      : Given grid coordinates (#coords,3)
                
                Output:               
                ec_OS1      : Opposite-spin based UMP2 correlation energy density array for OS1 index case evaluated with no DF
                ec_OS2      : Opposite-spin based UMP2 correlation energy density array for OS2 index case evaluated with no DF'''
               
                #Extraction of molecular orbital functions
                mol_orb_occ, mol_orb_virt = occ_virt_basis(mol, coords, mo_coeff)
                
                #Extracting Hartree integral
                A_integral=tensor_int_fake(coords, mol)
                
                #OS1
                if self.optimal_contract==0: # No Optimized contraction
                    #Sum all 
                    ec_OS1  = oe.contract('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[1],C_virt[1],A_integral,T_OS1)
                else: # Optimized contraction
                    #Sum all 
                    ec_OS1  = oe.contract('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[1],C_virt[1],A_integral,T_OS1,optimize='auto',memory_limit=self.optimal_contract)
                    
                    
                if self.verbose: #Saving size of largest intermediate contraction arrays
                    contract_info = oe.contract_path('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[1],C_virt[1],A_integral,T_OS1)
                    self.contract_size.append(contract_info[1].largest_intermediate * 8 / (1024**3)) #in GB
            
                #OS2
                if self.optimal_contract==0: # No Optimized contraction
                    #Sum all 
                    ec_OS2  = oe.contract('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[0],C_virt[0],A_integral,T_OS2)
                else: # Optimized contraction
                    #Sum all 
                    ec_OS2  = oe.contract('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[0],C_virt[0],A_integral,T_OS2,optimize='auto',memory_limit=self.optimal_contract)
                
                if self.verbose: #Saving size of largest intermediate contraction arrays
                    contract_info = oe.contract_path('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[0],C_virt[0],A_integral,T_OS2)
                    self.contract_size.append(contract_info[1].largest_intermediate * 8 / (1024**3)) #in GB
                            
                return -0.5*ec_OS1,-0.5*ec_OS2
            
            #Evaluating the correlation energy density on given grid
            if self.batch_size==0: #No batches specified
                if self.verbose:
                    print('Running correlation energy density evaluation on the full grid.')
                self.ec_OS1,self.ec_OS2 = ec_density_eval_noDF(T_OS1,T_OS2,self.mol,self.mo_coeff,C_occ,C_virt,self.coords_batches)

            else: #Batch-wise parallelization of the evaluation
                if self.verbose:
                    print('Running correlation energy density evaluation on batch wise separated grid.')
                    
                #Prestore arguments:
                partial_ec_density_eval_noDF = functools.partial(ec_density_eval_noDF, T_OS1,T_OS2, self.mol, self.mo_coeff, C_occ, C_virt)
                
                #Parallelization
                with ThreadPoolExecutor(max_workers=1) as executor: #Set 1 worker per thread for effective parallelization
                    
                    #Preallocate lists
                    self.ec_per_batch_OS1 = [None]*len(self.coords_batches) 
                    self.ec_per_batch_OS2 = [None]*len(self.coords_batches) 
                    
                    #Parallelize with respect to number of CPUs
                    for idx in range(0, len(self.coords_batches), self.cpus_per_task):
                        if self.verbose:
                            print(f'Submission for batches {idx} until {idx+self.cpus_per_task}')
                        coords_batch_par=self.coords_batches[idx:idx+self.cpus_per_task] #Collect batches
                        #Submit execution to available CPUs per batch
                        futures_indx = {executor.submit(partial_ec_density_eval_noDF,batch): indx for indx, batch in enumerate(coords_batch_par)}
                        for future in as_completed(futures_indx):
                            if self.verbose:
                                print(f'Finished batch evaluation {idx+futures_indx[future]}')
                            #Save results as jnp to ensure immutable    
                            self.ec_per_batch_OS1[futures_indx[future]+idx]=jnp.array(future.result()[0]) 
                            self.ec_per_batch_OS2[futures_indx[future]+idx]=jnp.array(future.result()[1])
                
                #Combination of the resulting correlation energy density batches
                self.ec_OS1 = jnp.concatenate(self.ec_per_batch_OS1, axis=0)
                self.ec_OS2 = jnp.concatenate(self.ec_per_batch_OS2, axis=0)
        
        end_time = time.time()
        print()
        print('Finished evaluation of the opposite-spin based UMP2 correlation energy density.')
        print('Elapsed total evaluation time: %.2f seconds' % np.abs(start_time-end_time))   
        print()
        if self.verbose: #Final printings
            #Save total memory usage of correlation density array:
            self.ec_size_OS1 = (self.ec_OS1.size * self.ec_OS1.itemsize) / (1024**3) #in GB  
            self.ec_size_OS2 = (self.ec_OS2.size * self.ec_OS2.itemsize) / (1024**3) #in GB  
            if self.batch_size > 0: #Save batchwise size of the energy density array:     
                self.ec_per_batch_size = (get_max_memory_object(self.ec_per_batch_OS2)+get_max_memory_object(self.ec_per_batch_OS2)) / (1024**3) #in GB 
            else: #For no batches, batchewise size is equal to full array size.
                self.ec_per_batch_size = (self.ec_size_OS1+self.ec_size_OS2)
            print('Final Memory usage:')
            print(f'T_ijab: {self.T_size1+self.T_size2:.8f} GB')
            print(f'Largest intermediate of the contraction path: {max(self.contract_size):.8f} GB')
            print(f'Correlation energy density array per batch: {self.ec_per_batch_size:.8f} GB')
            print(f'Full correlation energy density array: {self.ec_size_OS1+self.ec_size_OS2:.8f} GB')
            print()
      
    '''Opposite-spin UMP2 correlation energy value'''
    @property
    def energy(self):
        
        #Evaluate the integral with the density function
        Ec_value  = oe.contract('p,p->', self.ec_OS1, self.weights)
        Ec_value += oe.contract('p,p->', self.ec_OS2, self.weights)
        
        return Ec_value
        
    '''Opposite-spin UMP2 correlation energy density evaluated on the grid'''
    @property    
    def array(self):
        '''e_c(r)=1/rho(r) * e_c^{UMP2}(r)'''
        
        #Atomic orbitals evaluated on the grid 
        ao_value = numint.eval_ao(self.mol, self.coords, deriv=1)   
        
        # Evaluate electron density on same grid from atomic orbitals
        rho_alpha = numint.eval_rho(self.mol, ao_value[0], self.dm[0], xctype='LDA')
        rho_beta = numint.eval_rho(self.mol, ao_value[0], self.dm[1], xctype='LDA')

        ec = (self.ec_OS1+self.ec_OS2) / (rho_alpha+rho_beta) 
        return ec           
    
    '''Density weighted opposite-spin MP2 correlation energy density'''
    @property 
    def rho_weighted(self):
        
        ec_rho = (self.ec_OS1+self.ec_OS2) #Density weights already included in evaluation
        
        return ec_rho        
            
### Same-spin UMP2 correlation energy density for an open shell system ###
'''       
For open shell systems, the same-spin based formula reads:
    
e_c,ss^{UMP2}(r_p)= 1/(2*rho(r_p))*sum_{ijab}[T_{ijba}-T_{ijab}][V_{ijabp}-V_{ijbap}]

where T_ijab is the partial MP2 doubles amplitude,

T_{ijab} = (<ij|ab>)/(eps_a+eps_b-eps_i-eps_j),

and V_{ijabp} is the orbital tensor integral,

V_{ijabp} = phi_i(r_p)phi_a(r_p)*int (phi_j(r')*phi_b(r'))/(r_p-r')dr'
  
with molecular orbital spin electron cases:
SS1: i,j,a,b = alpha  
SS2: i,j,a,b = beta 
------------------------------------------------------------------------------------
Expansion to atomic orbitals yields a tensor multiplication notation per grid point:
m,n     -  atomic orbital index

V_{ijabp} = phi_i(r_p)phi_a(r_p)*sum_{mn}[C_{mj}*C_{nb} *int (chi_m(r')*chi_n(r'))/(r_p-r')dr']
          = phi_i(r_p)phi_a(r_p)*sum_{mn}[C_{mj}*C_{nb}*A_{mnp}]
------------------------------------------------------------------------------------          
 
'''
class ec_ump2_ss:
    ''' Evaluation of the same-spin based UMP2 correlation energy densities (outputs for SS1 and SS2 case).
        Density fitting, batchwise parallelization, frozen core orbitals or 
        kappa regularization are optional arguments. 
        
        Example:
        >>> kwargs = ec_mp2_kwargs()
        >>> atom_geom = 'Li 0 0 0'
        >>> basis = 'def2-tzvp'
        >>> Abasis = 'def2tzvpri' 
        >>> mol = gto.M(atom=atom_geom, basis=basis, spin=1, charge=0)
        >>> mf = dft.UKS(mol) 
        >>> mf.xc = 'hf'  
        >>> mf.kernel()
        >>> args = ec_mp2_args(mf,mol,DF=Abasis)
        >>> Ec_ss=ec_ump2_ss(*args, *kwargs)
        >>> print('Same-spin based correlation energy: %s' % Ec_ss.energy)
        '''

    def __init__(self,dm,mol,Amol,mo_coeff,mo_occ,mo_energies,coords,weights,
                 batch_size=0,kappa='inf',optimal_contract=0,frozen_core=0,verbose=False):
        '''
        *args* 
        dm          : Density matrix from an open-shell scf calculation (2,#basis,#basis)
        mol         : gto molecular structure incorporating the basis set
        Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        mo_coeff    : Coefficient matrix of the atomic orbitals (2,#basis,#basis)
        mo_occ      : Occupation numbers (2,#basis,)
        mo_energies : Orbital energies (2,#basis,)
        coords      : Grid coordinates (#coords,3)
        weights     : Grid weights (#coords,1)
        
        **kwargs
        batch_size = 0          : Batch size to run the evaluation on, 0 is for no parallelization
        kappa = 'inf'           : Laplace transform regularization parameter for the doubles amplitudes
                                  ('inf' for no regularization, else positive integer).
        optimal_contract = 0    : Optimized contraction algorithm option (0 by default, else positive integer
                                  representing number of elements in temporary array for the optimal contraction path).
        frozen_core = 0         : Frozen core orbital option (0 by default, 'auto' selects core orbitals
                                  automatically, else positive integer below number of orbitals).
        verbose = False         : Additional printings of time and memory statements
        '''
        
        #Saving arguments of the class
        self.dm                 = dm
        self.mol                = mol
        self.Amol               = Amol
        self.mo_coeff           = mo_coeff
        self.mo_occ             = mo_occ
        self.mo_energies        = mo_energies
        self.coords             = coords
        self.weights            = weights
        self.batch_size         = batch_size
        self.verbose            = verbose
        self.optimal_contract   = optimal_contract
        self.frozen_core        = frozen_core
        self.kappa              = kappa
        
        #Extracting the number of virtual and occupied orbitals
        self.Nocc, self.Nvirt  = orb_occ_virt(self.mo_occ)    
        
        #========================================#
        #Checking input...
        if self.Amol is None or isinstance(self.Amol,gto.mole.Mole): #DF option
            pass
        else:
            print('Amol needs to be a string name for the corresponding auxilliary basis set or "None" for no density fitting.')
            sys.exit()()
        
        if isinstance(self.optimal_contract,int)==False or self.optimal_contract < 0: #Einsum contraction option
            print('Optimal contraction argument has to be a positive integer or 0 for no optimal contraction.')
            sys.exit()()
        
        if isinstance(self.batch_size,int)==False or self.batch_size <0:  #Batch size option
            print('batch_size argument hast to be a positive integer or 0 for no parallelization.')
            sys.exit()()    
        
        if isinstance(self.frozen_core,int)==False or self.frozen_core < 0: #Number of frozen core orbitals
            if self.frozen_core == 'auto':
                pass
            else:
                print('''Number of frozen core orbitals must be a positive integer, 'auto' for automatic assignement or 0 for no frozen core approximation.''')
                sys.exit()()
       
        if isinstance(self.kappa, (int,float))==False or self.kappa < 0: #Kappa regularization 
            if self.kappa == 'inf':
                pass
            else:           
                print('''The regularization paramater kappa has to be a positive real number, 0 or 'inf'.''')
                sys.exit()()  

        #========================================#
        # Extra Functions for T_ijab and V_ijabp 
        
        #Two-body integrals of occupied and virtual alpha/beta molecular orbitals
        def two_body_integrals(mo_coeff,mol):
            '''Two-body integral computation of occupied and virtual molecular orbital functions: <ij|ab>.
            
            Input:
            mo_coeff          : Coefficient matrix of the atomic orbitals (2,#basis,#basis)
            mol               : gto molecular geometry
            
            Output: 
            two_integral_SS1  : Two body integral value for SS1 index case   (#alpha_occ_basis,#alpha_occ_basis,#alpha_virt_basis,#alpha_virt_basis)
            two_integral_SS2  : Two body integral value for SS2 index case   (#beta_occ_basis,#beta_occ_basis,#beta_virt_basis,#beta_virt_basis)'''


            #SS1 case:
            if frozen_core==0: #Check for frozen core orbital option<
                #iajb integrals (iofree):
                two_integral_eval_SS1 = ao2mo.outcore.general_iofree(mol, (mo_coeff[0][:,:self.Nocc[0]], mo_coeff[0][:,self.Nocc[0]:],
            mo_coeff[0][:,:self.Nocc[0]],mo_coeff[0][:,self.Nocc[0]:]),compact=False).reshape(self.Nocc[0],self.Nvirt[0],self.Nocc[0],self.Nvirt[0])
            else:
                #iajb integrals (iofree):
                two_integral_eval_SS1 = ao2mo.outcore.general_iofree(mol, (mo_coeff[0][:,self.frozen_core:self.Nocc[0]], mo_coeff[0][:,self.Nocc[0]:],
            mo_coeff[0][:,self.frozen_core:self.Nocc[0]],mo_coeff[0][:,self.Nocc[0]:]),compact=False).reshape(self.Nocc[0]-self.frozen_core,self.Nvirt[0],self.Nocc[0]-self.frozen_core,self.Nvirt[0])
            #ijab integrals:
            two_integral_eval_SS1 = two_integral_eval_SS1.transpose((0,2,1,3));
            
            #SS2 case:
            if frozen_core==0: #Check for frozen core orbital option<
                #iajb integrals (iofree):
                two_integral_eval_SS2 = ao2mo.outcore.general_iofree(mol, (mo_coeff[1][:,:self.Nocc[1]], mo_coeff[1][:,self.Nocc[1]:],
            mo_coeff[1][:,:self.Nocc[1]],mo_coeff[1][:,self.Nocc[1]:]),compact=False).reshape(self.Nocc[1],self.Nvirt[1],self.Nocc[1],self.Nvirt[1])
            else:
                #iajb integrals (iofree):
                two_integral_eval_SS2 = ao2mo.outcore.general_iofree(mol, (mo_coeff[1][:,self.frozen_core:self.Nocc[1]], mo_coeff[1][:,self.Nocc[1]:],
            mo_coeff[1][:,self.frozen_core:self.Nocc[1]],mo_coeff[1][:,self.Nocc[1]:]),compact=False).reshape(self.Nocc[1]-self.frozen_core,self.Nvirt[1],self.Nocc[1]-self.frozen_core,self.Nvirt[1])
            #ijab integrals:
            two_integral_eval_SS2 = two_integral_eval_SS2.transpose((0,2,1,3));
            
            return two_integral_eval_SS1,two_integral_eval_SS2

        #Partial MP2 doubles amplitude T_ijab
        def part_mp2_amplitude(mol,mo_coeff,mo_energies,mo_occ,kappa):
            '''Evaluation of the partial MP2 doubles amplitude:
            
            T_{ijab}=(<ij|ab>)/(eps_a+eps_b-eps_i-eps_j)
            
            Input:
            mol           : gto molecular geometry
            mo_coeff      : Coefficient matrix of the atomic orbitals (2,#basis,#basis)
            mo_energies   : Orbital energies (2,#basis,)
            mo_occ        : Occupation numbers (2,#basis,)
            kappa = 'inf' : Laplace transform regularization parameter for the doubles amplitudes,
                            'inf' for no regularization (original MP2 expression)
            
            Output:
            T_eval_SS1    : Evaluated partial MP2 doubles amplitude for SS1 index case (#alpha_occ_basis,#alpha_occ_basis,#alpha_virt_basis,#alpha_virt_basis)
            T_eval_SS2    : Evaluated partial MP2 doubles amplitude for SS2 index case (#beta_occ_basis,#beta_occ_basis,#beta_virt_basis,#beta_virt_basis)
            '''
            
            #Orbital energies distribution
            orbs_energies_occ_alpha  = mo_energies[0][mo_occ[0] > 0] #Energies of occupied alpha orbitals
            orbs_energies_occ_beta   = mo_energies[1][mo_occ[1] > 0] #Energies of occupied beta orbitals
            orbs_energies_virt_alpha = mo_energies[0][mo_occ[0] ==0] #Energies of virtual alpha orbitals
            orbs_energies_virt_beta  = mo_energies[1][mo_occ[1] ==0] #Energies of virtual beta orbitals
            
            #Checking for frozen core orbital option
            if self.frozen_core==0:
                Nocc_fc = self.Nocc
            else:
                Nocc_fc = []
                for k in range(len(self.Nocc)):
                    Nocc_fc.append(self.Nocc[k]-self.frozen_core) #Updated number of occupied orbitals
                orbs_energies_occ_alpha = orbs_energies_occ_alpha[self.frozen_core:] #Updated list of orbital energies
                orbs_energies_occ_beta  = orbs_energies_occ_beta[self.frozen_core:] #Updated list of orbital energies
                
            #Denominator of orbital energies [i,j,a,b]:
            Eps_SS1=np.zeros([Nocc_fc[0],Nocc_fc[0],self.Nvirt[0],self.Nvirt[0]])
            Eps_SS2=np.zeros([Nocc_fc[1],Nocc_fc[1],self.Nvirt[1],self.Nvirt[1]])

            #SS1
            for i in np.arange(Nocc_fc[0]):
                for j in np.arange(Nocc_fc[0]):
                    for a in np.arange(self.Nvirt[0]):
                        for b in np.arange(self.Nvirt[0]):
                            Eps_SS1[i,j,a,b] = orbs_energies_virt_alpha[a] + orbs_energies_virt_alpha[b] - orbs_energies_occ_alpha[i] - orbs_energies_occ_alpha[j]
            #SS2
            for i in np.arange(Nocc_fc[1]):
                for j in np.arange(Nocc_fc[1]):
                    for a in np.arange(self.Nvirt[1]):
                        for b in np.arange(self.Nvirt[1]):
                            Eps_SS2[i,j,a,b] = orbs_energies_virt_beta[a] + orbs_energies_virt_beta[b] - orbs_energies_occ_beta[i] - orbs_energies_occ_beta[j]

            
            #Evaluate two body integrals
            T_ijab_SS1,T_ijab_SS2 = two_body_integrals(mo_coeff,mol)
            
            #Final partial MP2 doubles amplitude
            if kappa == 'inf':
                T_eval_SS1 = T_ijab_SS1 / Eps_SS1
                T_eval_SS2 = T_ijab_SS2 / Eps_SS2
            else:
                T_eval_SS1 = T_ijab_SS1 / Eps_SS1 * ((1-np.exp( - kappa * Eps_SS1)) ** 2)        
                T_eval_SS2 = T_ijab_SS2 / Eps_SS2 * ((1-np.exp( - kappa * Eps_SS2)) ** 2)        
            
            return T_eval_SS1, T_eval_SS2

        #Evaluation of virtual and occupied orbital functions
        def occ_virt_basis(mol, coords, mo_coeff):
            '''Extracting orbital functions from the atomic orbitals (basis set) and molecular orbital coefficients
            
            Input:
            mol         : gto molecular structure incorporating the basis set
            coords      : Grid coordinates (#coords,3)
            mo_coeff    : Coefficient matrix of the alpha/beta atomic orbitals (2,#basis,#basis)
            
            Output:
            mol_orb_occ  : Occupied alpha/beta molecular orbital functions evaluated on the grid (2,#occ-basis, #coords)
            mol_orb_virt : Virtual alpha/beta molecular orbital functions evaluated on the grid (2,#virt-basis,#coords)'''
                    
            #Atomic orbitals evaluated on the (batched) grid
            ao_value = numint.eval_ao(mol, coords, deriv=1)                   #  (#derivatives, #coords, #basis)
            
            #Molecular orbital evaluation
            mol_orb_alpha = np.einsum('ji,pj->ip',mo_coeff[0],ao_value[0])    #  alpha (#basis, #coords)
            mol_orb_beta  = np.einsum('ji,pj->ip',mo_coeff[1],ao_value[0])    #  beta  (#basis, #coords)

            #Check for frozen core orbital option
            if frozen_core==0:
                mol_orb_occ_alpha = mol_orb_alpha[:self.Nocc[0],:]            #   (#occ_basis, #coords)
                mol_orb_occ_beta  = mol_orb_beta[:self.Nocc[1],:]             #   (#occ_basis, #coords)
            else:
                mol_orb_occ_alpha = mol_orb_alpha[self.frozen_core:self.Nocc[0],:]     #   (#occ_basis-#core_orb, #coords)
                mol_orb_occ_beta  = mol_orb_beta[self.frozen_core:self.Nocc[1],:]      #   (#occ_basis-#core_orb, #coords)
                 
            #Extracting virtual orbitals
            mol_orb_virt_alpha = mol_orb_alpha[self.Nocc[0]:,:]                #   (#vir-basis, #coords)
            mol_orb_virt_beta  = mol_orb_beta[self.Nocc[1]:,:]                 #   (#vir-basis, #coords)
         
            #Collecting results
            mol_orb_occ  =[mol_orb_occ_alpha,mol_orb_occ_beta]
            mol_orb_virt =[mol_orb_virt_alpha,mol_orb_virt_beta]
            
            return mol_orb_occ, mol_orb_virt
        
        #========================================#
        #Initialized printing:
        
        print('---------------------------------------------------------------')
        print('   Same-spin (ss) UMP2 correlation energy density evaluation   ')
        print('---------------------------------------------------------------')
        
        if self.verbose: #Parameter printing
            print('Evaluation parameters: ')
            print('Batch wise parallelization = ' + str(self.batch_size > 0))
            print('Density fitting = ' + str(isinstance(self.Amol,gto.mole.Mole)))
            print('Optimized einsum path = ' + str(self.optimal_contract > 0))
            print('Frozen core orbitals = ' + str(self.frozen_core))
            print('Initialising evaluation of the necessary components.')
            print(f'Number of [alpha,beta] occupied molecular orbitals is {self.Nocc}.')
            print(f'Number of [alpha,beta] virtual molecular orbitals is {self.Nvirt}.')
            self.contract_size = [] #Preallocate largest memory usage of contraction
               
        #Freezing the core orbitals
        if self.frozen_core=='auto': #Automatic frozen core orbitals
            self.frozen_core = int(num_core_orb(self.mol))
        if self.verbose:
            print()
            print('The first %s out of %s alpha and %s beta occupied orbitals are set frozen'
                  % (self.frozen_core,self.Nocc[0],self.Nocc[1]))
  
        #Starting evaluation
        #====================#
        if self.verbose: #Initial printing
            print()
            print('Starting evaluation...')
            print()
            
        start_time = time.time()
        
        # Partial MP2 doubles amplitude T_ijab (#occ_basis,#occ_basis,#virt_basis,#virt_basis) for SS1 and SS2 case
        T_SS1,T_SS2 = part_mp2_amplitude(self.mol,self.mo_coeff, self.mo_energies,
                               self.mo_occ,self.kappa)
        if self.verbose: #Saving size of T_ijab
            self.T_size1 = (T_SS1.size * T_SS1.itemsize) / (1024**3)  #in GB
            self.T_size2 = (T_SS2.size * T_SS2.itemsize) / (1024**3)  #in GB

        #Extraction of atomic orbital coefficients for alpha/beta electrons 
        if self.frozen_core==0:
            C_occ_alpha  = self.mo_coeff[0][:,:self.Nocc[0]]              #Occupied orbitals (#basis, #occ_basis_alpha)                
            C_occ_beta   = self.mo_coeff[1][:,:self.Nocc[1]]              #Occupied orbitals (#basis, #occ_basis_virt)                
        else:
            C_occ_alpha  = self.mo_coeff[0][:,self.frozen_core:self.Nocc[0]] #Occupied orbitals (#basis, #occ_basis_alpha-#core_orb)
            C_occ_beta   = self.mo_coeff[1][:,self.frozen_core:self.Nocc[1]] #Occupied orbitals (#basis, #occ_basis_virt-#core_orb)
        C_virt_alpha = self.mo_coeff[0][:,self.Nocc[0]:]                  #Virtual orbitals (#basis, #virt-basis_alpha)
        C_virt_beta = self.mo_coeff[1][:,self.Nocc[1]:]                   #Virtual orbitals (#basis, #virt-basis_alpha)
        
        #Collecting
        C_occ  = [C_occ_alpha,C_occ_beta]                                 #Occupied orbitals (alpha, beta)
        C_virt = [C_virt_alpha,C_virt_beta]                               #Virtual orbitals (alpha, beta)   
        
        if self.verbose: #Saving size of orbital coefficients
            self.C_occ_size = (C_occ[0].size * C_occ[0].itemsize+C_occ[1].size * C_occ[1].itemsize) / (1024**3) #in GB
            print(f'Memory usage of C_occ {self.C_occ_size:.8f} GB')
            self.C_virt_size = (C_virt[0].size * C_virt[0].itemsize+C_virt[1].size * C_virt[1].itemsize) / (1024**3) #in GB
            print(f'Memory usage of C_virt {self.C_virt_size:.8f} GB')
            self.mo_coeff_size = (self.mo_coeff[0].size * self.mo_coeff[0].itemsize+self.mo_coeff[1].size * self.mo_coeff[1].itemsize) / (1024**3) #in GB
            print(f'Memory usage of mo_coeff {self.mo_coeff_size:.8f} GB')
            print()
         
        #Batch seperation options:    
        if self.batch_size==0: #No parallelization 
            if self.verbose:
                print('No batch size chosen. Evaluation on the whole grid.')  
            self.coords_batches=self.coords 
        
        else: #Batch-wise parallelization           
            # Separate grid points into batches:
            self.coords_batches = [self.coords[i:i + self.batch_size] for i in range(0,self.coords.shape[0],self.batch_size)]  
            self.batches_amount = len(self.coords_batches) #Number of batches
            
            #Print Number of batches:
            if self.verbose:
                print('Separating grid into batches:')
                print('The number of batches is %s for a batch size of %s grid points.' % (self.batches_amount,self.batch_size))
                
            #Read the maximum amount of available cpu per task
            self.cpus_per_task = os.environ.get('SLURM_CPUS_PER_TASK')
            if self.cpus_per_task is not None:
                self.cpus_per_task = int(self.cpus_per_task)
            else:
                self.cpus_per_task = os.cpu_count()
                
            # Print the number of available CPU cores
            if self.verbose:
                print('Batch-wise evaluation on %s available CPU cores.' % self.cpus_per_task)
                
        
        #Check for density fitting option
        if self.Amol is not None: #Density fitting
            ''' Further expansion of V with a density fitted auxiliary basis set:

            A_{mnp}  = sum_{t}Q_{tmn}int psi_t(r')/(r_p-r')dr'
                     = sum_{t}Q_{tmn}I_tp 
             
            V_{ijab} = phi_i(r_p)*phi_a(r_p)*sum _{mn} C_{mj}C_{nb} A_{mnp}
                     = phi_i(r_p)*phi_a(r_p)*sum _{tmn} C_{mj}C_{nb}Q_{tmn}I_tp '''
                     
            #Aux basis electrostatic integrals
            def aux_basis_int(coords, Amol):  
                ''' Tensor integral evaluation of the auxiliary basis set:
                    I_t(r)=int psi_t(r1)/(|r1-r|)dr1
                    evaluated using delta distributions with the Hartree potential
                    
                    Input: 
                    coords  : grid coordinates (N,3),
                    Amol    : gto molecular geometry with aux basis
                
                    Output:
                    integralvalue : Array of integral values for every aux basis at every grid point (#coords,#aux-basis)
                '''

                # Creating fake dirac delta charges for the hartree potential evaluation on the grid points
                fakemol = gto.fakemol_for_charges(coords, expnt=1e+16)
                mol1 = fakemol + Amol
                
                I = mol1.intor('int2c2e', shls_slice=(0,fakemol.nbas,fakemol.nbas,mol1.nbas))
                
                return I  
            
            #Function to evaluate of the correlation energy density array without DF
            def ec_density_eval_DF(T_SS1,T_SS2,mol,mo_coeff,C_occ,C_virt,df_coeff,Amol,coords):
                '''Same-spin based UMP2 correlation energy density evaluations from the open-shell formula with DF:
                
                e_c,os^{MP2}(r_p)*rho(r_p)=0.5*sum_{ijab}[T_{ijba}-T_{ijab}][V_{ijabp}-V_{ijbap}]
                    =0.5*sum_{ijab}V_{ijabp}[T_{ijba}-T_{ijab}]-0.5*sum_{ijab}V_{ijbap}[T_{ijba}-T_{ijab}],
                
                Input:
                T_SS1       : Partial MP2 doubles amplitude (#occ_basis_alpha,#occ_basis_alpha,#virt_basis_alpha,#virt_basis_alpha)
                T_SS2       : Partial MP2 doubles amplitude (#occ_basis_beta,#occ_basis_beta,#virt_basis_beta,#virt_basis_beta)
                mol         : gto molecular structure incorporating the basis set.           
                mo_coeff    : Coefficient matrix of the alpha/beta atomic orbitals (2,#basis,#basis)
                C_occ       : Atomic orbital coefficients of alpha/beta occupied molecular orbitals (2,#basis,#occ_basis)
                C_virt      : Atomic orbital coefficients of alpha/beta virtual molecular orbitals (2,#basis,#virt_basis)
                df_coeff    : coefficient matrix from density fitting (#aux-basis,#basis,#basis)
                Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
                coords      : Given grid coordinates (#coords,3)
                
                Output:               
                ec_SS1      : Same-spin based UMP2 correlation energy density array for SS1 index case evaluated with no DF
                ec_SS2      : Same-spin based UMP2 correlation energy density array for SS2 index case evaluated with no DF'''
               
                #Extraction of molecular orbital functions
                mol_orb_occ, mol_orb_virt = occ_virt_basis(mol, coords, mo_coeff)
                
                #Extracting Hartree integral
                I_integral = aux_basis_int(coords,Amol)
                
                #Handling partial MP2 doubles amplitude (a<->b)
                T_SS1_transpose = np.transpose(T_SS1,(0,1,3,2))
                T_SS2_transpose = np.transpose(T_SS2,(0,1,3,2))
                
                #T_eval difference: 
                'T_{ijba}-T_{ijab}'
                T_SS1_inter = T_SS1_transpose-T_SS1
                T_SS2_inter = T_SS2_transpose-T_SS2
                
                #SS1
                if self.optimal_contract==0: # No Optimized contraction
                    #First sum
                    ec_SS1  = oe.contract('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[0],C_virt[0],df_coeff,I_integral,T_SS1_inter)
                    #Second sum
                    ec_SS1 -= oe.contract('ip,bp,mj,na,tmn,pt,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[0],C_virt[0],df_coeff,I_integral,T_SS1_inter)
                else: # Optimized contraction
                    #First sum
                    ec_SS1  = oe.contract('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[0],C_virt[0],df_coeff,I_integral,T_SS1_inter,optimize='auto',memory_limit=self.optimal_contract)
                    #Second sum
                    ec_SS1 -= oe.contract('ip,bp,mj,na,tmn,pt,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[0],C_virt[0],df_coeff,I_integral,T_SS1_inter,optimize='auto',memory_limit=self.optimal_contract)
                    
                if self.verbose: #Saving size of largest intermediate contraction arrays
                    contract_info = oe.contract_path('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[0],C_virt[0],df_coeff,I_integral,T_SS1_inter)
                    self.contract_size.append(contract_info[1].largest_intermediate * 8 / (1024**3)) #in GB
            
                #SS2
                if self.optimal_contract==0: # No Optimized contraction
                    #First sum
                    ec_SS2  = oe.contract('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[1],C_virt[1],df_coeff,I_integral,T_SS2_inter)
                    #Second sum
                    ec_SS2 -= oe.contract('ip,bp,mj,na,tmn,pt,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[1],C_virt[1],df_coeff,I_integral,T_SS2_inter)
                else: # Optimized contraction
                    #First sum
                    ec_SS2  = oe.contract('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[1],C_virt[1],df_coeff,I_integral,T_SS2_inter,optimize='auto',memory_limit=self.optimal_contract)
                    #Second sum
                    ec_SS2 -= oe.contract('ip,bp,mj,na,tmn,pt,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[1],C_virt[1],df_coeff,I_integral,T_SS2_inter,optimize='auto',memory_limit=self.optimal_contract)
                
                if self.verbose: #Saving size of largest intermediate contraction arrays
                    contract_info = oe.contract_path('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[0],C_virt[0],df_coeff,I_integral,T_SS1_inter)
                    self.contract_size.append(contract_info[1].largest_intermediate * 8 / (1024**3)) #in GB
                            
                return 0.25*ec_SS1, 0.25*ec_SS2
            
            #Density fitting coefficients for the auxliliary basis set (#aux-basis, #basis, #basis)
            self.df_coeff = obtain_df_coef(self.mol, self.Amol) 
            
            if self.verbose: #Saving size of auxiliary basis coefficients
                self.df_coeff_size = self.df_coeff.size * self.df_coeff.itemsize / (1024**3) #in GB
                print()
                print(f'Memory usage of the auxiliary basis set coefficients {self.df_coeff_size:.8f} GB')
                print()
            
            #Evaluating the correlation energy density on given grid
            if self.batch_size==0: #No batches specified
                if self.verbose:
                    print('Running correlation energy density evaluation on the full grid.')
                self.ec_SS1,self.ec_SS2 = ec_density_eval_DF(T_SS1,T_SS2,self.mol,self.mo_coeff,C_occ,C_virt,self.df_coeff,self.Amol,self.coords_batches)

            else: #Batch-wise parallelization of the evaluation
                if self.verbose:
                    print('Running correlation energy density evaluation on batch wise separated grid.')
                    
                #Prestore arguments:
                partial_ec_density_eval_DF = functools.partial(ec_density_eval_DF, T_SS1,T_SS2, self.mol, self.mo_coeff, C_occ, 
                                        C_virt, self.df_coeff,self.Amol)
                
                #Parallelization
                with ThreadPoolExecutor(max_workers=1) as executor: #Set 1 worker per thread for effective parallelization
                    
                    #Preallocate lists
                    self.ec_per_batch_SS1 = [None]*len(self.coords_batches) 
                    self.ec_per_batch_SS2 = [None]*len(self.coords_batches) 
                    
                    #Parallelize with respect to number of CPUs
                    for idx in range(0, len(self.coords_batches), self.cpus_per_task):
                        if self.verbose:
                            print(f'Submission for batches {idx} until {idx+self.cpus_per_task}')
                        coords_batch_par=self.coords_batches[idx:idx+self.cpus_per_task] #Collect batches
                        
                        #Submit execution to available CPUs per batch
                        futures_indx = {executor.submit(partial_ec_density_eval_DF,batch): indx for indx, batch in enumerate(coords_batch_par)}
                        for future in as_completed(futures_indx):
                            if self.verbose:
                                print(f'Finished batch evaluation {idx+futures_indx[future]}')
                            #Save results as jnp to ensure immutable    
                            self.ec_per_batch_SS1[futures_indx[future]+idx]=jnp.array(future.result()[0]) 
                            self.ec_per_batch_SS2[futures_indx[future]+idx]=jnp.array(future.result()[1])
                
                #Combination of the resulting correlation energy density batches
                self.ec_SS1 = jnp.concatenate(self.ec_per_batch_SS1, axis=0)
                self.ec_SS2 = jnp.concatenate(self.ec_per_batch_SS2, axis=0)
            
        else:   #Without density fitting
            ''' 
            Expansion to atomic orbitals yields the tensor multiplication from above.

            V_{ijabp} = phi_i(r_p)*phi_a(r_p)*sum _{mn} C_{mj}C_{nb} A_{mnp}
            '''
                
            #Tensor integral evaluation using Hartree potential 
            def tensor_int_fake(coords, mol):
                '''Tensor integral expression:
                A_{mn}(r)=int chi_m(r2)*chi_n(r2))/(|r-r2|)dr2
                evaluated using delta distributions with the Hartree potential
            
                Input: 
                coords  : grid coordinates (N,3),
                mol     : gto molecular geometry
            
                Output:
                integralvalue : Array of integral values for every atomic-basis at every grid point (#basis,#basis,N)
                '''

                # Creating fake dirac delta charges for the hartree potential evaluation on the grid points
                fakemol = gto.fakemol_for_charges(coords, expnt=1e+16)
                
                integralvalue = df.incore.aux_e2(mol, fakemol) #Evaluating the tensor integral as hartree potential
                
                return integralvalue    
            
            #Function to evaluate of the correlation energy density array without DF
            def ec_density_eval_noDF(T_SS1,T_SS2,mol,mo_coeff,C_occ,C_virt,coords):
                '''Same-spin based UMP2 correlation energy density evaluations from the open-shell formula:
                
                e_c,os^{MP2}(r_p)*rho(r_p)=0.5*sum_{ijab}[T_{ijba}-T_{ijab}][V_{ijabp}-V_{ijbap}]
                    =0.5*sum_{ijab}V_{ijabp}[T_{ijba}-T_{ijab}]-0.5*sum_{ijab}V_{ijbap}[T_{ijba}-T_{ijab}],
                
                Input:
                T_SS1       : Partial MP2 doubles amplitude (#occ_basis_alpha,#occ_basis_alpha,#virt_basis_alpha,#virt_basis_alpha)
                T_SS2       : Partial MP2 doubles amplitude (#occ_basis_beta,#occ_basis_beta,#virt_basis_beta,#virt_basis_beta)
                mol         : gto molecular structure incorporating the basis set.           
                mo_coeff    : Coefficient matrix of the alpha/beta atomic orbitals (2,#basis,#basis)
                C_occ       : Atomic orbital coefficients of alpha/beta occupied molecular orbitals (2,#basis,#occ_basis)
                C_virt      : Atomic orbital coefficients of alpha/beta virtual molecular orbitals (2,#basis,#virt_basis)
                coords      : Given grid coordinates (#coords,3)
                
                Output:               
                ec_SS1      : Same-spin based UMP2 correlation energy density array for SS1 index case evaluated with no DF
                ec_SS2      : Same-spin based UMP2 correlation energy density array for SS2 index case evaluated with no DF'''
               
                #Extraction of molecular orbital functions
                mol_orb_occ, mol_orb_virt = occ_virt_basis(mol, coords, mo_coeff)
                
                #Extracting Hartree integral
                A_integral=tensor_int_fake(coords, mol)
                
                #Handling partial MP2 doubles amplitude (a<->b)
                T_SS1_transpose = np.transpose(T_SS1,(0,1,3,2))
                T_SS2_transpose = np.transpose(T_SS2,(0,1,3,2))
                
                #T_eval difference: 
                'T_{ijba}-T_{ijab}'
                T_SS1_inter = T_SS1_transpose-T_SS1
                T_SS2_inter = T_SS2_transpose-T_SS2
                
                #SS1
                if self.optimal_contract==0: # No Optimized contraction
                    #First sum
                    ec_SS1  = oe.contract('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[0],C_virt[0],A_integral,T_SS1_inter)
                    #Second sum
                    ec_SS1 -= oe.contract('ip,bp,mj,na,mnp,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[0],C_virt[0],A_integral,T_SS1_inter)
                else: # Optimized contraction
                    #First sum
                    ec_SS1  = oe.contract('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[0],C_virt[0],A_integral,T_SS1_inter,optimize='auto',memory_limit=self.optimal_contract)
                    #Second sum
                    ec_SS1 -= oe.contract('ip,bp,mj,na,mnp,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[0],C_virt[0],A_integral,T_SS1_inter,optimize='auto',memory_limit=self.optimal_contract)
                    
                if self.verbose: #Saving size of largest intermediate contraction arrays
                    contract_info = oe.contract_path('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ[0],mol_orb_virt[0],
                                    C_occ[0],C_virt[0],A_integral,T_SS1_inter)
                    self.contract_size.append(contract_info[1].largest_intermediate * 8 / (1024**3)) #in GB
            
                #SS2
                if self.optimal_contract==0: # No Optimized contraction
                    #First sum
                    ec_SS2  = oe.contract('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[1],C_virt[1],A_integral,T_SS2_inter)
                    #Second sum
                    ec_SS2 -= oe.contract('ip,bp,mj,na,mnp,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[1],C_virt[1],A_integral,T_SS2_inter)
                else: # Optimized contraction
                    #First sum
                    ec_SS2  = oe.contract('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[1],C_virt[1],A_integral,T_SS2_inter,optimize='auto',memory_limit=self.optimal_contract)
                    #Second sum
                    ec_SS2 -= oe.contract('ip,bp,mj,na,mnp,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[1],C_virt[1],A_integral,T_SS2_inter,optimize='auto',memory_limit=self.optimal_contract)
                    
                if self.verbose: #Saving size of largest intermediate contraction arrays
                    contract_info = oe.contract_path('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ[1],mol_orb_virt[1],
                                    C_occ[1],C_virt[1],A_integral,T_SS2_inter)
                    self.contract_size.append(contract_info[1].largest_intermediate * 8 / (1024**3)) #in GB               
            
                return 0.25*ec_SS1, 0.25*ec_SS2
            
            #Evaluating the correlation energy density on given grid
            if self.batch_size==0: #No batches specified
                if self.verbose:
                    print('Running correlation energy density evaluation on the full grid.')
                self.ec_SS1,self.ec_SS2 = ec_density_eval_noDF(T_SS1,T_SS2,self.mol,self.mo_coeff,C_occ,C_virt,self.coords_batches)

            else: #Batch-wise parallelization of the evaluation
                if self.verbose:
                    print('Running correlation energy density evaluation on batch wise separated grid.')
                    
                #Prestore arguments:
                partial_ec_density_eval_noDF = functools.partial(ec_density_eval_noDF, T_SS1,T_SS2, self.mol, self.mo_coeff, C_occ,C_virt)
                
                #Parallelization
                with ThreadPoolExecutor(max_workers=1) as executor: #Set 1 worker per thread for effective parallelization
                    
                    #Preallocate lists
                    self.ec_per_batch_SS1 = [None]*len(self.coords_batches) 
                    self.ec_per_batch_SS2 = [None]*len(self.coords_batches) 
                    
                    #Parallelize with respect to number of CPUs
                    for idx in range(0, len(self.coords_batches), self.cpus_per_task):
                        if self.verbose:
                            print(f'Submission for batches {idx} until {idx+self.cpus_per_task}')
                        coords_batch_par=self.coords_batches[idx:idx+self.cpus_per_task] #Collect batches
                        #Submit execution to available CPUs per batch
                        futures_indx = {executor.submit(partial_ec_density_eval_noDF,batch): indx for indx, batch in enumerate(coords_batch_par)}
                        for future in as_completed(futures_indx):
                            if self.verbose:
                                print(f'Finished batch evaluation {idx+futures_indx[future]}')
                            #Save results as jnp to ensure immutable    
                            self.ec_per_batch_SS1[futures_indx[future]+idx]=jnp.array(future.result()[0]) 
                            self.ec_per_batch_SS2[futures_indx[future]+idx]=jnp.array(future.result()[1])
                
                #Combination of the resulting correlation energy density batches
                self.ec_SS1 = jnp.concatenate(self.ec_per_batch_SS1, axis=0)
                self.ec_SS2 = jnp.concatenate(self.ec_per_batch_SS2, axis=0)
        
        end_time = time.time()
        print()
        print('Finished evaluation of the same-spin based UMP2 correlation energy density.')
        print('Elapsed total evaluation time: %.2f seconds' % np.abs(start_time-end_time))   
        print()
        if self.verbose: #Final printings
            #Save total memory usage of correlation density array:
            self.ec_size_SS1 = (self.ec_SS1.size * self.ec_SS1.itemsize) / (1024**3) #in GB  
            self.ec_size_SS2 = (self.ec_SS2.size * self.ec_SS2.itemsize) / (1024**3) #in GB  
            if self.batch_size > 0: #Save batchwise size of the energy density array:     
                self.ec_per_batch_size = (get_max_memory_object(self.ec_per_batch_SS2)+get_max_memory_object(self.ec_per_batch_SS2)) / (1024**3) #in GB 
            else: #For no batches, batchewise size is equal to full array size.
                self.ec_per_batch_size = (self.ec_size_SS1+self.ec_size_SS2)
            print('Final Memory usage:')
            print(f'T_ijab: {self.T_size1+self.T_size2:.8f} GB')
            print(f'Largest intermediate of the contraction path: {max(self.contract_size):.8f} GB')
            print(f'Correlation energy density array per batch: {self.ec_per_batch_size:.8f} GB')
            print(f'Full correlation energy density array: {self.ec_size_SS1+self.ec_size_SS2:.8f} GB')
            print()

      
    '''Same-spin UMP2 correlation energy value'''
    @property
    def energy(self):
    
        #Evaluate the integral with the density function
        Ec_value  = oe.contract('p,p->', self.ec_SS1, self.weights)
        Ec_value += oe.contract('p,p->', self.ec_SS2, self.weights)
        
        return Ec_value
        
    '''Same-spin UMP2 correlation energy density evaluated on the grid'''
    @property    
    def array(self):
        '''e_c(r)=1/rho(r) * e_c^{UMP2}(r)'''
        
        #Atomic orbitals evaluated on the grid 
        ao_value = numint.eval_ao(self.mol, self.coords, deriv=1)   
        
        # Evaluate electron density on same grid from atomic orbitals
        rho_alpha = numint.eval_rho(self.mol, ao_value[0], self.dm[0], xctype='LDA')
        rho_beta = numint.eval_rho(self.mol, ao_value[0], self.dm[1], xctype='LDA')

        ec = (self.ec_SS1+self.ec_SS2) / (rho_alpha+rho_beta) 
        return ec     
     
    '''Density weighted same-spin MP2 correlation energy density'''
    @property 
    def rho_weighted(self):
        
        ec_rho = (self.ec_SS1+self.ec_SS2) #Density weights already included in evaluation
        
        return ec_rho     
             
#============================================#
#       Additional python functions          #
#============================================#

#Occupied and Virtual alpha/beta orbitals functions 
def orb_occ_virt(mo_occ):
    '''Extracting the number of occupied and virtual alpha/beta molecular orbitals from the occupation numbers.
    
    Input: 
    mo_occ : Molecular orbital occupation numbers (2,#basis,)
    
    Output:
    self.Nocc  : Number of occupied alpha/beta molecular orbitals (2,)
    self.Nvirt : Number of virtual alpha/beta molecular orbitals  (2,)
    
    Example:
    >>>atom_geom = 'Li 0 0 0'
    >>>basis = 'def2-svp'
    >>>mol = gto.M(atom=atom_geom, basis=basis)
    >>>mf = dft.UKS(mol) 
    >>>mf.xc = 'hf'  
    >>>mf.kernel()
    >>>mo_occ=mf.mo_occ
    >>>self.Nocc, self.Nvirt = orb_occ_virt(mo_occ) 
    >>>print(f'Number of occupied alpha and beta orbitals: {self.Nocc}')
    >>>print(f'Number of virtual alpha and beta orbitals: {self.Nvirt}')
    '''
    
    #Extracting the number of total and occupied alpha/beta orbitals
    Nocc  = [mo_occ[0][mo_occ[0] > 0].shape[0],mo_occ[1][mo_occ[1]>0].shape[0]]
    Nvirt = [mo_occ[0][mo_occ[0] == 0].shape[0],mo_occ[1][mo_occ[1] == 0].shape[0]]
    
    return Nocc, Nvirt

#Extracting density fitting coefficients:
def obtain_df_coef(mol, Amol, df_metric="2c2e", batch_size=50, check_2el_error=False, verbose=False):
    '''Computes df_coef for "mol" basis function, with auxiliary basis from Amol
    Author: Stefan Vuckovic

    Arg:
        mol, PySCF Mol, e.g., mol = gto.M(atom = 'Ne 0.0 0.0')
        Amol, PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        
    Kwargs:
        batch_size: int, controls batch size of arrays for linear solver, default 50
        df_metric=, can be "2c2e" (default, more accurate, somewhat slower) or "2c1e"
    
    Returns:
        numpy.ndarray of shape (naux, nao, nao) 


    Examples:

    >>> mol = gto.M(atom='H 0 0 0; Cl 0 0 1', basis='def2-qzvp')
    >>> Amol = df.addons.make_auxmol(mol, 'def2qzvpri')
    '''
                
                # Define a jnp function to solve the linear system
    # @jit
    def solve_linear_system(ints_2c, ints_3c):
        # the shape of ints_2c is (naux, naux)
        # the shape of ints_3c is (nao, nao, naux)
        nao = ints_3c.shape[0]
        naux = ints_2c.shape[0]
        ints_3c_T = ints_3c.reshape(nao * nao, naux).T
        # of shape naux, nao, nao
        return jnp.linalg.solve(ints_2c, ints_3c_T).reshape(naux, nao, nao)
    
    nao = mol.nao
    naux = Amol.nao
    
    if verbose:
        print()
        print("Obtaining density fitting coefficients...")
        print()
        print("Number of atomic orbitals: {}".format(nao))
        print("Number of auxiliary basis functions: {}".format(naux))
        
    start_time = time.time()
    
    # 3c2e integrals for DF
    ints_3c2e = df.incore.aux_e2(mol, Amol, intor="int3c2e")

    # other integrals for DF:
    if df_metric == "2c2e":
        ints_2c2e = Amol.intor("int2c2e")
        ints_2c = ints_2c2e
        ints_3c = ints_3c2e
        arrays_dict = {"ints_3c2e": ints_3c2e, "ints_2c2e": ints_2c2e}
        # print("Density fitting to be done with ints_2c = ints_2c2e and ints_3c = ints_3c2e")
    elif df_metric == "2c1e":
        ints_3c1e = df.incore.aux_e2(mol, Amol, intor="int3c1e")
        ints_2c1e = Amol.intor("int1e_ovlp")
        ints_2c = ints_2c1e
        ints_3c = ints_3c1e
        arrays_dict = {"ints_3c1e": ints_3c1e, "ints_2c1e": ints_2c1e}
        # print("Density fitting to be done with ints_2c = ints_2c1e and ints_3c = ints_3c1e")
    else:
        print("error: df_metric can be either 2c2e or 2c1e")

    # print the sizes of arrays
    for array_name in arrays_dict:
        array = arrays_dict[array_name]
        size_mb = round(array.nbytes / (1024 * 1024), 2)
        if verbose:
            print(
                "Shape of {} is {}, and its size is {} MB".format(
                    array_name, array.shape, size_mb
                )
            )
    if verbose:
        if check_2el_error == True:
            # Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
            df_eri = lib.einsum("ijP,Pkl->ijkl", ints_3c2e, df_coef)
            # Now check the error of DF integrals wrt the normal ERIs
            print(
                "max error in 2-elec integrals due to DF is",
                abs(mol.intor("int2e") - df_eri).max(),
            )
        else:
            pass

    # divide 'ints_3d into sub-arrays of shape(n,n)'
    # Compute the remainder and padding for 'ints_3d' needed
    # Set the size of the sub-arrays to the smaller of sv_df_n and nao
    n = min(batch_size, nao)
    # Print the size of the sub-arrays needed for density fitting
    # print(f"The size of sub-arrays of ints_3c needed for density fitting is {n}")
    # Compute the remainder of dividing nao by n
    remainder = nao % n
    # the amount of padding needed to ensure that the array size is 
    #divisible by n
    d_pad = n - remainder if remainder != 0 else 0

    # Create a new array of zeros with the augmented shape
    ints_3c_aug = np.zeros((nao + d_pad, nao + d_pad, naux))

    # Copy elements from the original array to the augmented array
    ints_3c_aug[:nao, :nao, :] = ints_3c

    # split 'ints_3c_aug' into 'n_sqr' sub-arrays of shape (n,n)
    n_sqr = (nao + d_pad) // n

    list_ind = []  # index for slicing 'ints_3c_aug'
    for i in range(n_sqr):
        start_i = i * n
        end_i = (i + 1) * n
        for j in range(n_sqr):
            start_j = j * n
            end_j = (j + 1) * n
            list_ind.append([start_i, end_i, start_j, end_j])

    df_coef_aug = np.zeros((naux, nao + d_pad, nao + d_pad))

    for k in range(len(list_ind)):
        ind_k = list_ind[k]
        start_i, end_i, start_j, end_j = ind_k[0], ind_k[1], ind_k[2], ind_k[3]
        sub_k = ints_3c_aug[start_i:end_i, start_j:end_j, :]
        part_coef = solve_linear_system(ints_2c, sub_k)
        df_coef_aug[:, start_i:end_i, start_j:end_j] = part_coef

    df_coef = df_coef_aug[:, 0:nao, 0:nao]

    end_time = time.time()
    if verbose:
        print("The density fitting took: %s sec" % np.around(end_time - start_time, decimals=2))

    return df_coef
        
#Getting the memory of the object with maximum memory of a list
def get_max_memory_object(lst):
    max_size = 0
    max_obj = None
    for obj in lst:
        obj_size = sys.getsizeof(obj)
        if obj_size > max_size:
            max_size = obj_size
            max_obj = obj
    return max_size        
   
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
    
