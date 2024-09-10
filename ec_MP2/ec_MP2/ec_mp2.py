''' *** Main script for the MP2 correlation energy density generator class ***

General formula is taken from:
S. Vuckovic, T. J. P. Irons, A. Savin, A. M. Teale, and P. Gori-Giorgi, “Ex-
change-Correlation Functionals via Local Interpolation along the Adiabatic
Connection”, Journal of Chemical Theory and Computation 12, 2598-2610 (2016).

The following python functionalities evaluate the MP2 correlation energy density and the corresponding 
MP2 correlation energy.

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
V_ijabp - orbital tensor integrals     
'''

import numpy as np 
import jax.numpy as jnp
import time
import os
import sys
import opt_einsum as oe
import functools
import gc
import threading
lock = threading.Lock()

from pyscf import df, gto, ao2mo, lib
from pyscf.dft import numint
from pyscf.data import elements
from concurrent.futures import ThreadPoolExecutor, as_completed

### MP2 correlation energy density for a closed shell system ###
'''       
For closed shell systems, the MP2 correlation energy density reads:
    
0.5*w'_0(r_p) = e_c^{MP2}(r_p)
              = 1/(rho(r_p))*sum_{ijab}V_{ijabp}(0.5*T_{ijba}-T_{ijab})
                + V_{ijbap}(0.5*T_{ijab}-T_{ijba}),
           
where T_ijab is the partial MP2 doubles amplitude,

T_{ijab} = (<ij|ab>)/(eps_a+eps_b-eps_i-eps_j),

and V_{ijabp} is the orbital tensor integral,

V_{ijabp} = phi_i(r_p)phi_a(r_p)*int (phi_j(r')*phi_b(r'))/(r_p-r')dr'
  
Expansion to atomic orbitals yields a tensor multiplication notation per grid point:

V_{ijabp} = phi_i(r_p)phi_a(r_p)*sum_{mn}[C_{mj}*C_{nb} *int (chi_m(r')*chi_n(r'))/(r_p-r')dr']
          = phi_i(r_p)phi_a(r_p)*sum_{mn}[C_{mj}*C_{nb}*A_{mnp}]
'''
class ec_mp2_cs:
    ''' Evaluation of the MP2 correlation energy density for a closed shell system. 
        Density fitting, batchwise parallelization, frozen core orbitals or 
        kappa regularization are optional arguments. 
        
        Requires a RKS or RHF Pyscf class object.
        
        Example:

        >>>batch_size=0 
        >>>verbose=False/True 
        >>>DF=True/False 
        >>>optimal_contract=True
        >>>max_num_array=5000
        >>>frozen_core=False
        >>>spinorb=False
        >>>num_core='auto'  
        >>>kappa='inf'
        >>>kwargs = ec_mp2_kwargs(batch_size,DF,verbose,optimal_contract,max_num_array,frozen_core,
                    spinorb,num_core,kappa)
        >>>atom_geom = 'He 0 0 0'
        >>>basis = 'def2-tzvp'
        >>>Abasis = 'def2tzvpri' #aux basis for MP2 correlation energy
        >>>mol = gto.M(atom=atom_geom, basis=basis)
        >>>mf = dft.RKS(mol) 
        >>>mf.xc = 'hf'  
        >>>mf.kernel()
        >>>args = ec_mp2_args(mf,mol,Abasis)
        >>>Ec=ec_mp2_cs(*args, *kwargs)
        >>>print('MP2 correlation energy: %s' % Ec.energy)
        '''

    def __init__(self,dm,mol,Amol,mo_coeff,mo_occ,mo_energies,coords,weights,
                 batch_size=0,DF=True,verbose=False, optimal_contract=False, max_num_array=None,
                 frozen_core=False,spinorb=False,num_core='auto',kappa='inf'):
        '''
        
        *args* 
        dm          : Density matrix from a scf calculation (#basis,#basis)
        mol         : gto molecular structure incorporating the basis set
        Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        mo_coeff    : Coefficient matrix of the atomic orbitals (#basis,#basis)
        mo_occ      : Occupation numbers (#basis,)
        mo_energies : Orbital energies (#basis,)
        coords      : Grid coordinates (#coords,3)
        weights     : Grid weights (#coords,1)
        
        **kwargs
        batch_size = 0           : Grid batch size to run the evaluation on, 0 is for no parallelization
        DF = True                : Density fitting option
        verbose = False          : Additional printings of time and memory statements
        optimal_contract = False : Optimized contraction algorithm for the opt_einsum summation path
        max_num_array = None     : Maximum number of elements in a temporary array for the optimal contraction path, requires a postive integer.
        frozen_core = False      : Frozen core orbital option
        spinorb = False          : Specification of the spin orbital usage of frozen core orbitals, False for R/U is fine
        num_core = 'auto'        : Amount of frozen orbitals, 'auto' selects the core ones.
        kappa = 'inf'            : Laplace transform regularization parameter for the doubles amplitudes, 'inf' for no regularization (original MP2 expression)
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
        self.DF                 = DF
        self.verbose            = verbose
        self.optimal_contract   = optimal_contract
        self.frozen_core        = frozen_core
        self.kappa              = kappa
        
        #Checking input...
        if self.optimal_contract: #Einsum contraction option
            self.max_num_array   = max_num_array
            #Check value of memory limit:
            if isinstance(self.max_num_array,int)==False or self.max_num_array < 0 or self.max_num_array ==0:
                print('Memory limit for optimal contraction has to be a positive integer in MB')
                sys.exit()()
        
        if isinstance(self.batch_size,int)==False or self.batch_size <0:  #Batch size option
            print('batch_size argument hast to be a positive integer or 0 for no parallelization')
            sys.exit()()    
        
        if isinstance(num_core,int)==False or num_core < 0: #Number of core orbitals
            if num_core == 'auto':
                pass
            else:
                print('''Number of core orbitals must be a positive integer or 'auto' for automatic assignement''')
                sys.exit()()
       
        if isinstance(self.kappa, (int,float))==False or self.kappa < 0: #Kappa regularization 
            if self.kappa == 'inf':
                pass
            else:           
                print('''The regularization paramater kappa has to be a positive real number, 0 or 'inf'.''')
                sys.exit()()   
                

        #========================================#
        # Extra Functions for T_ijab and V_ijabp 
        
        #Two-body integrals of occupied and virtual molecular orbitals
        def two_body_integrals(mo_coeff,mo_occ,mol,frozen_core,num_core):
            '''Two-body integral computation of occupied and virtual molecular orbital functions: <ij|ab>.
            
            Input:
            mo_coeff          : Coefficient matrix of the atomic orbitals (#basis,#basis)
            mo_occ            : Molecular orbital occupation numbers (#basis,)
            mol               : gto molecular geometry
            frozen_core       : Frozen core orbital option
            num_core          : Amount of frozen orbitals, 'auto' selects the core ones.
            
            Output: 
            two_integral_eval        : Two integral value depending on the molecular orbital function index
            (#occ_basis,#occ_basis,#virt_basis,#virt_basis)'''    
            
            #Extracting the number of total and occupied orbitals
            Nocc, Nvirt  = orb_occ_virt(mo_occ)
                
            #Check for frozen core orbital option
            if frozen_core:
                #iajb integrals (iofree):
                two_integral_eval_0 = ao2mo.outcore.general_iofree(mol, (mo_coeff[:,num_core:Nocc], mo_coeff[:,Nocc:],
            mo_coeff[:,num_core:Nocc],mo_coeff[:,Nocc:]),compact=False).reshape(Nocc-num_core,Nvirt,Nocc-num_core,Nvirt)
            
            else:
                #iajb integrals (iofree):
                two_integral_eval_0 = ao2mo.outcore.general_iofree(mol, (mo_coeff[:,:Nocc], mo_coeff[:,Nocc:],
            mo_coeff[:,:Nocc],mo_coeff[:,Nocc:]),compact=False).reshape(Nocc,Nvirt,Nocc,Nvirt)
            
            
            #ijab integrals:    
            two_integral_eval = two_integral_eval_0.transpose((0,2,1,3));
                
            return two_integral_eval 

        #Partial MP2 doubles amplitude T_ijab
        def part_mp2_amplitude(mol,mo_coeff, mo_energies,mo_occ,frozen_core,num_core, kappa):
            '''Evaluation of the partial MP2 doubles amplitude:
            
            T_{ijab}=(<ij|ab>)/(eps_a+eps_b-eps_i-eps_j)
            
            Input:
            mol           : gto molecular geometry
            mo_coeff      : Coefficient matrix of the atomic orbitals (#basis,#basis)
            mo_energies   : Orbital energies (#basis,)
            mo_occ        : Occupation numbers (#basis,)
            frozen_core   : Frozen core orbital option
            num_core      : Amount of frozen orbitals, 'auto' selects the core ones.
            kappa = 'inf' : Laplace transform regularization parameter for the doubles amplitudes,
                            'inf' for no regularization (original MP2 expression)
            
            Output:
            T_eval : Evaluated partial MP2 doubles amplitude (#occ_basis,#occ_basis,#virt_basis,#virt_basis)
            '''
            
            #Orbital energies distribution
            orbs_energies_occ  = mo_energies[mo_occ > 0] #Energies of occupied orbitals
            orbs_energies_virt = mo_energies[mo_occ ==0] #Energies of virtual orbitals
            
            #Extracting the number of total and occupied orbitals
            Nocc, Nvirt  = orb_occ_virt(mo_occ)
            UC = Nvirt
            
            #Checking for frozen core orbital option
            if frozen_core:
                OC = Nocc-num_core #Updated number of occupied orbitals
                orbs_energies_occ = orbs_energies_occ[num_core:] #Updated list of orbital energies
                
            else:
                OC = Nocc

            #Denominator of orbital energies:
            Eps=np.zeros([OC,OC,UC,UC])
            
            for i in np.arange(OC):
                for j in np.arange(OC):
                    for a in np.arange(UC):
                        for b in np.arange(UC):
                            Eps[i,j,a,b] = orbs_energies_virt[a] + orbs_energies_virt[b] - orbs_energies_occ[i] - orbs_energies_occ[j]
            
            
            #Evaluate two body integrals
            T_ijab = two_body_integrals(mo_coeff,mo_occ,mol,frozen_core,num_core)
            
            #Final partial MP2 doubles amplitude
            if kappa == 'inf':
                T_eval = T_ijab / Eps
            else:
                T_eval = T_ijab / Eps * ((1-np.exp( - kappa * Eps)) ** 2)        
            
            return T_eval

        #Evaluation of virtual and occupied orbital functions
        def occ_virt_basis(mol, coords, mo_coeff,frozen_core,num_core):
            '''Extracting orbital functions from the atomic orbitals (basis set) and molecular orbital coefficients
            
            Input:
            mol         : gto molecular structure incorporating the basis set
            coords      : Grid coordinates (#coords,3)
            mo_coeff    : Coefficient matrix of the atomic orbitals (#basis,#basis)
            frozen_core : Option to use frozen core orbitals within the evaluation
            num_core    : Number of frozen core orbitals. 
            
            Output:
            mol_orb_occ  : Occupied molecular orbital functions evaluated on the grid (#occ-basis, #coords)
            mol_orb_virt : Virtual molecular orbital functions evaluated on the grid (#virt-basis,#coords)'''
                    
            #Atomic orbitals evaluated on the (batched) grid
            ao_value = numint.eval_ao(mol, coords, deriv=1)              #  (#derivatives, #coords, #basis)
            
            #Molecular orbital evaluation
            mol_orb = np.einsum('ji,pj->ip',mo_coeff,ao_value[0])        #   (#basis, #coords)
            
            #Check for frozen core orbital option
            if frozen_core:
                mol_orb_occ = mol_orb[num_core:Nocc,:]                  #   (#occ_basis-#core_orb, #coords)
            else:
                mol_orb_occ = mol_orb[:Nocc,:]                          #   (#occ_basis, #coords)
                
            #Extracting virtual orbitals
            mol_orb_virt = mol_orb[Nocc:,:]                              #   (#vir-basis, #coords)
         
            return mol_orb_occ, mol_orb_virt
        
        #=======================================#
        
        #Extracting the number of virtual and occupied orbitals
        Nocc, Nvirt  = orb_occ_virt(self.mo_occ)     
         
        #Initialized printing:
        
        print('MP2 correlation energy density modelling for a closed-shell system.')
        
        if self.verbose: #Parameter printing
            print('Evaluation parameters: ')
            print('Batch wise parallelization = ' + str(self.batch_size > 0))
            print('Density fitting = ' + str(self.DF))
            print('Optimized einsum path = ' + str(self.optimal_contract))
            print('Frozen core orbitals = ' + str(self.frozen_core))
            print('Initialising evaluation of the necessary components.')
            print(f'Number of occupied molecular orbitals is {Nocc}.')
            print(f'Number of virtual molecular orbitals is {Nvirt}.')
            self.contract_size      = [] #Preallocating contraction memory usage
               
        #Freezing the core orbitals
        if self.frozen_core:
            if num_core == 'auto': #Automatic frozen core orbitals
                self.num_core = num_core_orb(self.mol,spinorb)
            else: #Manually chosen frozen core orbitals
                self.num_core = num_core
            if self.verbose:
                print()
                print('The number of frozen core orbitals for the evaluation is %s out of %s occupied orbitals' % (self.num_core,Nocc))
        else: #No frozen core orbitals
            self.num_core = 0     
        
        
        #Starting evaluation
        #====================#
        if self.verbose: #Initial printing
            print()
            print('Starting evaluation...')
            print()
            
        start_time = time.time()
        
        # Partial MP2 doubles amplitude T_ijab (#occ_basis,#occ_basis,#virt_basis,#virt_basis)
        T = part_mp2_amplitude(self.mol,self.mo_coeff, self.mo_energies,
                               self.mo_occ,self.frozen_core,self.num_core,self.kappa)
        if self.verbose: #Saving size of T_ijab
            self.T_size = (T.size * T.itemsize) / (1024**3)  #in GB
            print('Memory usage of T_ijab:')
            print(f'{self.T_size:.8f} GB')
            print()
                    
        #Extraction of atomic orbital coefficients 
        if self.frozen_core:
            C_occ  = self.mo_coeff[:,self.num_core:Nocc] #Occupied orbitals (#basis, #occ_basis-#core_orb)
        else:
            C_occ  = self.mo_coeff[:,:Nocc]         #Occupied orbitals (#basis, #occ_basis)                
        
        C_virt = self.mo_coeff[:,Nocc:]             #Virtual orbitals (#basis, #virt-basis)
        
        
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
        if DF:  #Density fitting
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
            
            #Function to evaluate of the correlation energy density array with DF
            def ec_density_eval_DF(T, mol, mo_coeff,C_occ,C_virt, df_coeff, Amol, frozen_core, num_core, coords):
                '''Correlation energy density evaluation from the closed-shell formula 
                using T_ijab and V_ijab in the DF expansion:
                
                e_c(r_p)  = 0.5*w'_0(r_p)*rho(r_p)
                          = e_c^{MP2}(r_p)*rho(r_p)
                          = sum_{ijab}V_{ijab}(r_p)(0.5*T_{ijba}-T_{ijab}) 
                           + sum_{ijab}V_{ijba}(r_p)(0.5*T_{ijab}-T_{ijba}),
                
                Input:
                T            : partial MP2 doubles amplitude (#occ_basis,#occ_basis,#virt_basis,#virt_basis)
                mol          : gto molecular structure incorporating the basis set.
                mo_coeff     : Coefficient matrix of the atomic orbitals (#basis,#basis)
                C_occ        : Atomic orbital coefficients of occupied molecular orbitals (#basis,#occ_basis)
                C_virt       : Atomic orbital coefficients of virtual molecular orbitals (#basis,#virt_basis)
                df_coeff     : coefficient matrix from density fitting (#aux-basis,#basis,#basis)
                Amol         : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
                frozen_core  : Option to use frozen core orbitals within the evaluation
                num_core     : Number of frozen core orbitals. 
                coords       : Given grid coordinates (#coords,3)
            
                Output:
                ec              : Correlation energy density array evaluated on the given (batched) grid'''
               
                #Extraction of molecular orbital functions
                mol_orb_occ, mol_orb_virt = occ_virt_basis(mol, coords, mo_coeff,frozen_core,num_core)
                    
                #Extracting Hartree integral
                I_integral=aux_basis_int(coords, Amol)
                
                #Handling partial MP2 doubles amplitude (a<->b)
                T_transpose = np.transpose(T,(0,1,3,2))
                
                # 1st Intermediate T_eval: 
                '0.5*T_{ijba}-T_{ijab}'
                T_inter_ba = 0.5*T_transpose-T
                
                # 2st Intermediate T_eval: 
                '0.5*T_{ijab}-T_{ijba}'
                T_inter_ab = 0.5*T-T_transpose 
                
                with lock: #Lock tasks in one Thread
                    
                    # Optimized contraction
                    if self.optimal_contract:  
                        #First sum 
                        ec = oe.contract('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,df_coeff,I_integral,T_inter_ba,optimize='auto',memory_limit=self.max_num_array)
                        #Second sum
                        ec += oe.contract('ip,bp,mj,na,tmn,pt,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,df_coeff,I_integral,T_inter_ab,optimize='auto',memory_limit=self.max_num_array)
                    
                    else:
                        #First sum 
                        ec = oe.contract('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,df_coeff,I_integral,T_inter_ba)
                        #Second sum
                        ec += oe.contract('ip,bp,mj,na,tmn,pt,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,df_coeff,I_integral,T_inter_ab)
                        
                    if self.verbose: #Saving size of largest intermediate contraction arrays
                        contract_info = oe.contract_path('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,df_coeff,I_integral,T_inter_ba)
                        self.contract_size.append(contract_info[1].largest_intermediate * 8 / (1024**3)) #in GB    
                        
                return ec      
              
            #Density fitting coefficients for the auxliliary basis set (#aux-basis, #basis, #basis)
            self.df_coeff = obtain_df_coef(self.mol, self.Amol) 
            
            if self.verbose: #Saving size of auxiliary basis coefficients
                self.df_coeff_size = self.df_coeff.size * self.df_coeff.itemsize / (1024**3) #in GB
                print()
                print('Memory usage of the auxiliary basis set coefficients:')
                print(f'{self.df_coeff_size:.8f} GB')
                print()
            
            #Evaluating the correlation energy density on given grid
            if self.batch_size==0: #No batches specified
                if self.verbose:
                    print('Running correlation energy density evaluation on the full grid.')
                self.ec = ec_density_eval_DF(T,self.mol,self.mo_coeff,C_occ,C_virt,self.df_coeff,self.Amol,
                                             self.frozen_core,self.num_core,self.coords_batches)

            else: #Batch-wise parallelization of the evaluation
                if self.verbose:
                    print('Running correlation energy density evaluation on batch wise separated grid.')
                    
                #Prestore arguments:
                partial_ec_density_eval_DF = functools.partial(ec_density_eval_DF, T, self.mol, self.mo_coeff, C_occ, 
                                        C_virt, self.df_coeff, self.Amol, self.frozen_core, self.num_core)
                
                #Parallelization
                with ThreadPoolExecutor(max_workers=self.cpus_per_task) as executor: 
                    
                    #Submitting tasks and saving them at the right index
                    self.ec_per_batch = [None]*len(self.coords_batches) #Preallocate list 
                    
                    #Parallelize with respect to number of CPUs
                    for idx in range(0, len(self.coords_batches), self.cpus_per_task):
                        if self.verbose:
                            print(f'Submission for batches {idx} until {idx+self.cpus_per_task}')
                        coords_batch_par=self.coords_batches[idx:idx+self.cpus_per_task] #Collect batches
                        #Submit execution to available CPUs per batch
                        futures_indx = {executor.submit(partial_ec_density_eval_DF,batch): indx for indx, batch in enumerate(coords_batch_par)}
                        # futures = [executor.submit(partial_ec_density_eval_DF,batch) for batch in coords_batch_par]
                        for future in as_completed(futures_indx):
                            if self.verbose:
                                print(f'Finished batch evaluation {idx+futures_indx[future]}')
                            self.ec_per_batch[futures_indx[future]+idx]=jnp.array(future.result()) #Save results as jnp to ensure immutable
                        #Clear execution and cache
                        futures_indx.clear()   
                        gc.collect()

                #Combination of the resulting correlation energy density batches
                self.ec = jnp.concatenate(self.ec_per_batch, axis=0)
                
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
            def ec_density_eval_noDF(T,mol,mo_coeff,C_occ,C_virt,frozen_core,num_core,coords):
                '''Correlation energy density evaluation from the closed-shell formula 
                using T_ijab and V_ijab:
                
                ec(r_p) := 0.5*w'_0(r_p)*rho(r_p)
                         = e_c^{MP2}(r_p)*rho(r_p)
                         = sum_{ijab}V_{ijab}(r_p)(0.5*T_{ijba}-T_{ijab}) 
                           + sum_{ijab}V_{ijba}(r_p)(0.5*T_{ijab}-T_{ijba}),
                
                Input:
                T           : partial MP2 doubles amplitude (#occ_basis,#occ_basis,#virt_basis,#virt_basis)
                mol         : gto molecular structure incorporating the basis set.           
                mo_coeff    : Coefficient matrix of the atomic orbitals (#basis,#basis)
                C_occ       : Atomic orbital coefficients of occupied molecular orbitals (#basis,#occ_basis)
                C_virt      : Atomic orbital coefficients of virtual molecular orbitals (#basis,#virt_basis)
                frozen_core : Option to use frozen core orbitals within the evaluation
                num_core    : Number of frozen core orbitals. 
                coords      : Given grid coordinates (#coords,3)
                
                Output:               
                ec           : correlation energy density array evaluated with no DF.'''
               
                #Extraction of molecular orbital functions
                mol_orb_occ, mol_orb_virt = occ_virt_basis(mol, coords, mo_coeff,frozen_core,num_core)
                
                #Extracting Hartree integral
                A_integral=tensor_int_fake(coords, mol)
                
                #Handling partial MP2 doubles amplitude (a<->b)
                T_transpose = np.transpose(T,(0,1,3,2))
                
                # 1st Intermediate T_eval: 
                '0.5*T_{ijba}-T_{ijab}'
                T_inter_ba = 0.5*T_transpose-T
                
                # 2st Intermediate T_eval: 
                '0.5*T_{ijab}-T_{ijba}'
                T_inter_ab = 0.5*T-T_transpose 

                with lock: #Lock tasks in one Thread

                    if self.optimal_contract: # Optimized contraction
                        #First sum 
                        ec  = oe.contract('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,A_integral,T_inter_ba,optimize='auto',memory_limit=self.max_num_array)
                        #Second sum 
                        ec += oe.contract('ip,bp,mj,na,mnp,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,A_integral,T_inter_ab,optimize='auto',memory_limit=self.max_num_array)
                    else:
                        #First sum 
                        ec  = oe.contract('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,A_integral,T_inter_ba)
                        #Seond sum 
                        ec += oe.contract('ip,bp,mj,na,mnp,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,A_integral,T_inter_ab)  
                    
                    if self.verbose: #Saving size of largest intermediate contraction arrays
                        contract_info = oe.contract_path('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,A_integral,T_inter_ba)
                        self.contract_size.append(contract_info[1].largest_intermediate * 8 / (1024**3)) #in GB

                return ec
                
            #Evaluating the correlation energy density on given grid
            if self.batch_size==0: #No batches specified
                if self.verbose:
                    print('Running correlation energy density evaluation on the full grid.')
                self.ec = ec_density_eval_noDF(T,self.mol,self.mo_coeff,C_occ,C_virt,self.frozen_core,
                                               self.num_core,self.coords_batches)

            else: #Batch-wise parallelization of the evaluation
                if self.verbose:
                    print('Running correlation energy density evaluation on batch wise separated grid.')
                    
                #Prestore arguments:
                partial_ec_density_eval_noDF = functools.partial(ec_density_eval_noDF, T, self.mol, self.mo_coeff, C_occ, 
                                        C_virt, self.frozen_core, self.num_core)
                
                #Parallelization
                with ThreadPoolExecutor(max_workers=self.cpus_per_task) as executor: 
                    
                    #Submitting tasks and saving them at the right index
                    self.ec_per_batch = [None]*len(self.coords_batches) #Preallocate list 
                    
                    #Parallelize with respect to number of CPUs
                    for idx in range(0, len(self.coords_batches), self.cpus_per_task):
                        if self.verbose:
                            print(f'Submission for batches {idx} until {idx+self.cpus_per_task}')
                        coords_batch_par=self.coords_batches[idx:idx+self.cpus_per_task] #Collect batches
                        #Submit execution to available CPUs per batch
                        futures_indx = {executor.submit(partial_ec_density_eval_noDF,batch): indx for indx, batch in enumerate(coords_batch_par)}
                        # futures = [executor.submit(partial_ec_density_eval_DF,batch) for batch in coords_batch_par]
                        for future in as_completed(futures_indx):
                            if self.verbose:
                                print(f'Finished batch evaluation {idx+futures_indx[future]}')
                            self.ec_per_batch[futures_indx[future]+idx]=jnp.array(future.result()) #Save results as jnp to ensure immutable
                        #Clear execution and cache
                        futures_indx.clear()   
                        gc.collect()

                #Combination of the resulting correlation energy density batches
                self.ec = jnp.concatenate(self.ec_per_batch, axis=0)
        
        end_time = time.time()
        print()
        print('Finished evaluation of the correlation energy density.')
        print('Elapsed total evaluation time: %.2f seconds' % np.abs(start_time-end_time))   
        print()
        if self.verbose: #Final printings
            #Save total memory usage of correlation density array:
            self.ec_size = (self.ec.size * self.ec.itemsize) / (1024**3) #in GB  
            if self.batch_size > 0: #Save batchwise size of the energy density array:     
                self.ec_per_batch_size = get_max_memory_object(self.ec_per_batch) / (1024**3) #in GB 
            else: #For no batches, batchewise size is equal to full array size.
                self.ec_per_batch_size = self.ec_size
            print('Final memory usage:')
            print(f'Largest intermediate of the contraction path: {max(self.contract_size):.8f} GB')
            print(f'Correlation energy density array per batch: {self.ec_per_batch_size:.8f} GB')
            print(f'Full correlation energy density array: {self.ec_size:.8f} GB')
            print('-------------------------------------------------------------')
        
    
    '''MP2 correlation energy value'''
    @property
    def energy(self):
    
        #Evaluate the integral with the density function
        Ec_value = oe.contract('p,p->', self.ec, self.weights)
    
        return Ec_value
        
    '''MP2 correlation energy density evaluated on the grid'''
    @property    
    def array(self):
        '''ec(r)=1/rho(r) * ec^{MP2}(r)'''
        
        #Atomic orbitals evaluated on the grid 
        ao_value = numint.eval_ao(self.mol, self.coords, deriv=1)   
        
        # Evaluate electron density on same grid from atomic orbitals
        rho = numint.eval_rho(self.mol, ao_value[0], self.dm, xctype='LDA')

        ec = self.ec / rho 
        return ec           
    


### Opposite spin based MP2 correlation energy density ###
'''       
The opposite spin based MP2 correlation energy density reads
    
e_c_os^{MP2}(r_p) = - 1/(2*rho(r_p))*sum_{ijab}[V_{ijabp}*T_{ijab} + V_{ijbap}T_{ijba}],
           
where T_ijab is the partial MP2 doubles amplitude,

T_{ijab} = (<ij|ab>)/(eps_a+eps_b-eps_i-eps_j),

and V_{ijabp} is the orbital tensor integral,

V_{ijabp} = phi_i(r_p)phi_a(r_p)*int (phi_j(r')*phi_b(r'))/(r_p-r')dr'
  
Expansion to atomic orbitals yields a tensor multiplication notation per grid point:

V_{ijabp} = phi_i(r_p)phi_a(r_p)*sum_{mn}[C_{mj}*C_{nb} *int (chi_m(r')*chi_n(r'))/(r_p-r')dr']
          = phi_i(r_p)phi_a(r_p)*sum_{mn}[C_{mj}*C_{nb}*A_{mnp}]
'''
class ec_mp2_os:
    ''' Evaluation of the opposite spin based MP2 correlation energy density. 
        Density fitting, batchwise parallelization, frozen core orbitals or 
        kappa regularization are optional arguments. 
        
        Example:

        >>>batch_size=0 
        >>>verbose=False/True 
        >>>DF=True/False 
        >>>optimal_contract=True
        >>>max_num_array=5000
        >>>frozen_core=False
        >>>spinorb=False
        >>>num_core='auto'  
        >>>kappa='inf'
        >>>kwargs = ec_mp2_kwargs(batch_size,DF,verbose,optimal_contract,max_num_array,frozen_core,
                    spinorb,num_core,kappa)
        >>>atom_geom = 'He 0 0 0'
        >>>basis = 'def2-tzvp'
        >>>Abasis = 'def2tzvpri' #aux basis for MP2 correlation energy
        >>>mol = gto.M(atom=atom_geom, basis=basis)
        >>>mf = dft.RKS(mol) 
        >>>mf.xc = 'hf'  
        >>>mf.kernel()
        >>>args = ec_mp2_args(mf,mol,Abasis)
        >>>Ec_os=ec_mp2_os(*args, *kwargs)
        >>>print('os MP2 correlation energy: %s' % Ec_os.energy)
        '''

    def __init__(self,dm,mol,Amol,mo_coeff,mo_occ,mo_energies,coords,weights,
                 batch_size=0,DF=True,verbose=False, optimal_contract=False, max_num_array=None,
                 frozen_core=False,spinorb=False,num_core='auto',kappa='inf'):
        '''
        
        *args* 
        dm          : Density matrix from a scf calculation (#basis,#basis)
        mol         : gto molecular structure incorporating the basis set
        Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        mo_coeff    : Coefficient matrix of the atomic orbitals (#basis,#basis)
        mo_occ      : Occupation numbers (#basis,)
        mo_energies : Orbital energies (#basis,)
        coords      : Grid coordinates (#coords,3)
        weights     : Grid weights (#coords,1)
        
        **kwargs
        batch_size = 0           : Grid batch size to run the evaluation on, 0 is for no parallelization
        DF = True                : Density fitting option
        verbose = False          : Additional printings of time and memory statements
        optimal_contract = False : Optimized contraction algorithm for the opt_einsum summation path
        max_num_array = None     : Maximum number of elements in a temporary array for the optimal contraction path, requires a postive integer.
        frozen_core = False      : Frozen core orbital option
        spinorb = False          : Specification of the spin orbital usage of frozen core orbitals, False for R/U is fine
        num_core = 'auto'        : Amount of frozen orbitals, 'auto' selects the core ones.
        kappa = 'inf'            : Laplace transform regularization parameter for the doubles amplitudes, 'inf' for no regularization (original MP2 expression)
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
        self.DF                 = DF
        self.verbose            = verbose
        self.optimal_contract   = optimal_contract
        self.frozen_core        = frozen_core
        self.kappa              = kappa
        
        #Checking input...
        if self.optimal_contract: #Einsum contraction option
            self.max_num_array   = max_num_array
            #Check value of memory limit:
            if isinstance(self.max_num_array,int)==False or self.max_num_array < 0 or self.max_num_array ==0:
                print('Memory limit for optimal contraction has to be a positive integer in MB')
                sys.exit()()
        
        if isinstance(self.batch_size,int)==False or self.batch_size <0:  #Batch size option
            print('batch_size argument hast to be a positive integer or 0 for no parallelization')
            sys.exit()()    
        
        if isinstance(num_core,int)==False or num_core < 0: #Number of core orbitals
            if num_core == 'auto':
                pass
            else:
                print('''Number of core orbitals must be a positive integer or 'auto' for automatic assignement''')
                sys.exit()()
       
        if isinstance(self.kappa, (int,float))==False or self.kappa < 0: #Kappa regularization 
            if self.kappa == 'inf':
                pass
            else:           
                print('''The regularization paramater kappa has to be a positive real number, 0 or 'inf'.''')
                sys.exit()()  

        #========================================#
        # Extra Functions for T_ijab and V_ijabp 
        
        #Two-body integrals of occupied and virtual molecular orbitals
        def two_body_integrals(mo_coeff,mo_occ,mol,frozen_core,num_core):
            '''Two-body integral computation of occupied and virtual molecular orbital functions: <ij|ab>.
            
            Input:
            mo_coeff          : Coefficient matrix of the atomic orbitals (#basis,#basis)
            mo_occ            : Molecular orbital occupation numbers (#basis,)
            mol               : gto molecular geometry
            frozen_core       : Frozen core orbital option
            num_core          : Amount of frozen orbitals, 'auto' selects the core ones.
            
            Output: 
            two_integral_eval        : Two integral value depending on the molecular orbital function index
            (#occ_basis,#occ_basis,#virt_basis,#virt_basis)'''    
            
            #Extracting the number of total and occupied orbitals
            Nocc, Nvirt  = orb_occ_virt(mo_occ)
                
            #Check for frozen core orbital option
            if frozen_core:
                #iajb integrals (iofree):
                two_integral_eval_0 = ao2mo.outcore.general_iofree(mol, (mo_coeff[:,num_core:Nocc], mo_coeff[:,Nocc:],
            mo_coeff[:,num_core:Nocc],mo_coeff[:,Nocc:]),compact=False).reshape(Nocc-num_core,Nvirt,Nocc-num_core,Nvirt)
            
            else:
                #iajb integrals (iofree):
                two_integral_eval_0 = ao2mo.outcore.general_iofree(mol, (mo_coeff[:,:Nocc], mo_coeff[:,Nocc:],
            mo_coeff[:,:Nocc],mo_coeff[:,Nocc:]),compact=False).reshape(Nocc,Nvirt,Nocc,Nvirt)
            
            
            #ijab integrals:    
            two_integral_eval = two_integral_eval_0.transpose((0,2,1,3));
                
            return two_integral_eval 

        #Partial MP2 doubles amplitude T_ijab
        def part_mp2_amplitude(mol,mo_coeff, mo_energies,mo_occ,frozen_core,num_core, kappa):
            '''Evaluation of the partial MP2 doubles amplitude:
            
            T_{ijab}=(<ij|ab>)/(eps_a+eps_b-eps_i-eps_j)
            
            Input:
            mol           : gto molecular geometry
            mo_coeff      : Coefficient matrix of the atomic orbitals (#basis,#basis)
            mo_energies   : Orbital energies (#basis,)
            mo_occ        : Occupation numbers (#basis,)
            frozen_core   : Frozen core orbital option
            num_core      : Amount of frozen orbitals, 'auto' selects the core ones.
            kappa = 'inf' : Laplace transform regularization parameter for the doubles amplitudes,
                            'inf' for no regularization (original MP2 expression)
            
            Output:
            T_eval : Evaluated partial MP2 doubles amplitude (#occ_basis,#occ_basis,#virt_basis,#virt_basis)
            '''
            
            #Orbital energies distribution
            orbs_energies_occ  = mo_energies[mo_occ > 0] #Energies of occupied orbitals
            orbs_energies_virt = mo_energies[mo_occ ==0] #Energies of virtual orbitals
            
            #Extracting the number of total and occupied orbitals
            Nocc, Nvirt  = orb_occ_virt(mo_occ)
            UC = Nvirt
            
            #Checking for frozen core orbital option
            if frozen_core:
                OC = Nocc-num_core #Updated number of occupied orbitals
                orbs_energies_occ = orbs_energies_occ[num_core:] #Updated list of orbital energies
                
            else:
                OC = Nocc

            #Denominator of orbital energies:
            Eps=np.zeros([OC,OC,UC,UC])
            
            for i in np.arange(OC):
                for j in np.arange(OC):
                    for a in np.arange(UC):
                        for b in np.arange(UC):
                            Eps[i,j,a,b] = orbs_energies_virt[a] + orbs_energies_virt[b] - orbs_energies_occ[i] - orbs_energies_occ[j]
            
            
            #Evaluate two body integrals
            T_ijab = two_body_integrals(mo_coeff,mo_occ,mol,frozen_core,num_core)
            
            #Final partial MP2 doubles amplitude
            if kappa == 'inf':
                T_eval = T_ijab / Eps
            else:
                T_eval = T_ijab / Eps * ((1-np.exp( - kappa * Eps)) ** 2)        
            
            return T_eval

        #Evaluation of virtual and occupied orbital functions
        def occ_virt_basis(mol, coords, mo_coeff,frozen_core,num_core):
            '''Extracting orbital functions from the atomic orbitals (basis set) and molecular orbital coefficients
            
            Input:
            mol         : gto molecular structure incorporating the basis set
            coords      : Grid coordinates (#coords,3)
            mo_coeff    : Coefficient matrix of the atomic orbitals (#basis,#basis)
            frozen_core : Option to use frozen core orbitals within the evaluation
            num_core    : Number of frozen core orbitals. 
            
            Output:
            mol_orb_occ  : Occupied molecular orbital functions evaluated on the grid (#occ-basis, #coords)
            mol_orb_virt : Virtual molecular orbital functions evaluated on the grid (#virt-basis,#coords)'''
                    
            #Atomic orbitals evaluated on the (batched) grid
            ao_value = numint.eval_ao(mol, coords, deriv=1)              #  (#derivatives, #coords, #basis)
            
            #Molecular orbital evaluation
            mol_orb = np.einsum('ji,pj->ip',mo_coeff,ao_value[0])        #   (#basis, #coords)
            
            #Check for frozen core orbital option
            if frozen_core:
                mol_orb_occ = mol_orb[num_core:Nocc,:]                  #   (#occ_basis-#core_orb, #coords)
            else:
                mol_orb_occ = mol_orb[:Nocc,:]                          #   (#occ_basis, #coords)
                
            #Extracting virtual orbitals
            mol_orb_virt = mol_orb[Nocc:,:]                              #   (#vir-basis, #coords)
         
            return mol_orb_occ, mol_orb_virt
        
        #=======================================#
        
        #Extracting the number of virtual and occupied orbitals
        Nocc, Nvirt  = orb_occ_virt(self.mo_occ)     
         
        #Initialized printing:
        
        print('Opposite spin based MP2 correlation energy density modelling.')
        
        if self.verbose: #Parameter printing
            print('Evaluation parameters: ')
            print('Batch wise parallelization = ' + str(self.batch_size > 0))
            print('Density fitting = ' + str(self.DF))
            print('Optimized einsum path = ' + str(self.optimal_contract))
            print('Frozen core orbitals = ' + str(self.frozen_core))
            print('Initialising evaluation of the necessary components.')
            print(f'Number of occupied molecular orbitals is {Nocc}.')
            print(f'Number of virtual molecular orbitals is {Nvirt}.')
            self.contract_size = [] #Preallocate largest memory usage of contraction
               
        #Freezing the core orbitals
        if self.frozen_core:
            if num_core == 'auto': #Automatic frozen core orbitals
                self.num_core = num_core_orb(self.mol,spinorb)
            else: #Manually chosen frozen core orbitals
                self.num_core = num_core
            if self.verbose:
                print()
                print('The number of frozen core orbitals for the evaluation is %s out of %s occupied orbitals' % (self.num_core,Nocc))
        else: #No frozen core orbitals
            self.num_core = 0     
        
        
        #Starting evaluation
        #====================#
        if self.verbose: #Initial printing
            print()
            print('Starting evaluation...')
            print()
            
        start_time = time.time()
        
        # Partial MP2 doubles amplitude T_ijab (#occ_basis,#occ_basis,#virt_basis,#virt_basis)
        T = part_mp2_amplitude(self.mol,self.mo_coeff, self.mo_energies,
                               self.mo_occ,self.frozen_core,self.num_core,self.kappa)
        if self.verbose: #Saving size of T_ijab
            self.T_size = (T.size * T.itemsize) / (1024**3)  #in GB
            print('Memory usage of T_ijab:')
            print(f'{self.T_size:.8f} GB')
            print()
        
        #Extraction of atomic orbital coefficients 
        if self.frozen_core:
            C_occ  = self.mo_coeff[:,self.num_core:Nocc] #Occupied orbitals (#basis, #occ_basis-#core_orb)
        else:
            C_occ  = self.mo_coeff[:,:Nocc]         #Occupied orbitals (#basis, #occ_basis)                
        
        C_virt = self.mo_coeff[:,Nocc:]             #Virtual orbitals (#basis, #virt-basis)
        
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
        if DF:  #Density fitting
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
                     
            #Function to evaluate of the correlation energy density array with DF
            def ec_density_eval_DF(T,mol,mo_coeff,C_occ,C_virt, df_coeff, Amol, frozen_core, num_core,coords):
                '''Correlation energy density evaluation from the opposite spin closed-shell formula 
                using T_ijab and V_ijab in the DF expansion:
                
                e_c(r_p)  = 0.5*w'_0(r_p)*rho(r_p)
                          = e_c^{MP2}(r_p)*rho(r_p)
                          = -0.5*sum_{ijab}[V_{ijab}(r_p)T_{ijab}+V_{ijba}(r_p)T_{ijba}],
                
                Input:
                T            : partial MP2 doubles amplitude (#occ_basis,#occ_basis,#virt_basis,#virt_basis)
                mol          : gto molecular structure incorporating the basis set.
                mo_coeff     : Coefficient matrix of the atomic orbitals (#basis,#basis)
                C_occ        : Atomic orbital coefficients of occupied molecular orbitals (#basis,#occ_basis)
                C_virt       : Atomic orbital coefficients of virtual molecular orbitals (#basis,#virt_basis)
                df_coeff     : coefficient matrix from density fitting (#aux-basis,#basis,#basis)
                Amol         : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
                frozen_core  : Option to use frozen core orbitals within the evaluation
                num_core     : Number of frozen core orbitals. 
                coords       : Given grid coordinates (#coords,3)
            
                Output:
                ec           : Opposite spin based correlation energy density array evaluated on the given (batched) grid'''
               
                #Extraction of molecular orbital functions
                mol_orb_occ, mol_orb_virt = occ_virt_basis(mol, coords, mo_coeff,frozen_core,num_core)
                    
                #Extracting Hartree integral
                I_integral=aux_basis_int(coords, Amol)
                
                #Handling partial MP2 doubles amplitude (a<->b)
                T_transpose = np.transpose(T,(0,1,3,2))
                
                with lock: #Lock tasks in one Thread
                        
                    # Optimized contraction
                    if self.optimal_contract:  
                        #First sum 
                        ec = oe.contract('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ,mol_orb_virt,
                            C_occ,C_virt,df_coeff,I_integral,T,optimize='auto',memory_limit=self.max_num_array)
                        #Second sum
                        ec += oe.contract('ip,bp,mj,na,tmn,pt,ijab->p',mol_orb_occ,mol_orb_virt,
                            C_occ,C_virt,df_coeff,I_integral,T_transpose,optimize='auto',memory_limit=self.max_num_array)
                    
                    else:
                        #First sum 
                        ec = oe.contract('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,df_coeff,I_integral,T)
                        #Second sum
                        ec += oe.contract('ip,bp,mj,na,tmn,pt,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,df_coeff,I_integral,T_transpose)
                        
                    if self.verbose: #Saving size of largest intermediate contraction arrays
                        contract_info = oe.contract_path('ip,ap,mj,nb,tmn,pt,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,df_coeff,I_integral,T)
                        self.contract_size.append(contract_info[1].largest_intermediate * 8 / (1024**3)) #in GB    
                        
                return -0.5 * ec      
              
            #Density fitting coefficients for the auxliliary basis set (#aux-basis, #basis, #basis)
            self.df_coeff = obtain_df_coef(self.mol, self.Amol) 
            
            if self.verbose: #Saving size of auxiliary basis coefficients
                self.df_coeff_size = self.df_coeff.size * self.df_coeff.itemsize / (1024**3) #in GB
                print()
                print('Memory usage of the auxiliary basis set coefficients:')
                print(f'{self.df_coeff_size:.8f} GB')
                print()
            
            #Evaluating the correlation energy density on given grid
            if self.batch_size==0: #No batches specified
                if self.verbose:
                    print('Running correlation energy density evaluation on the full grid.')
                self.ec = ec_density_eval_DF(T,self.mol,self.mo_coeff,C_occ,C_virt,self.df_coeff,self.Amol,
                                             self.frozen_core,self.num_core,self.coords_batches)

            else: #Batch-wise parallelization of the evaluation
                if self.verbose:
                    print('Running correlation energy density evaluation on batch wise separated grid.')
                    
                #Prestore arguments:
                partial_ec_density_eval_DF = functools.partial(ec_density_eval_DF, T, self.mol, self.mo_coeff, C_occ, 
                                        C_virt, self.df_coeff, self.Amol, self.frozen_core, self.num_core)
                
                #Parallelization
                with ThreadPoolExecutor(max_workers=self.cpus_per_task) as executor: 
                    
                    #Submitting tasks and saving them at the right index
                    self.ec_per_batch = [None]*len(self.coords_batches) #Preallocate list 
                    
                    #Parallelize with respect to number of CPUs
                    for idx in range(0, len(self.coords_batches), self.cpus_per_task):
                        if self.verbose:
                            print(f'Submission for batches {idx} until {idx+self.cpus_per_task}')
                        coords_batch_par=self.coords_batches[idx:idx+self.cpus_per_task] #Collect batches
                        #Submit execution to available CPUs per batch
                        futures_indx = {executor.submit(partial_ec_density_eval_DF,batch): indx for indx, batch in enumerate(coords_batch_par)}
                        # futures = [executor.submit(partial_ec_density_eval_DF,batch) for batch in coords_batch_par]
                        for future in as_completed(futures_indx):
                            if self.verbose:
                                print(f'Finished batch evaluation {idx+futures_indx[future]}')
                            self.ec_per_batch[futures_indx[future]+idx]=jnp.array(future.result()) #Save results as jnp to ensure immutable
                        #Clear execution and cache
                        futures_indx.clear()   
                        gc.collect()

                #Combination of the resulting correlation energy density batches
                self.ec = jnp.concatenate(self.ec_per_batch, axis=0)

                
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
            def ec_density_eval_noDF(T,mol,mo_coeff,C_occ,C_virt,frozen_core,num_core,coords):
                '''Opposite spin based correlation energy density evaluation from the closed-shell formula 
                using T_ijab and V_ijab:
                
                ec(r_p) := 0.5*w'_0(r_p)*rho(r_p)
                         = e_c^{MP2}(r_p)*rho(r_p)
                         = -0.5*sum_{ijab}[V_{ijab}(r_p)T_{ijab}+V_{ijba}T_{ijba}],
                
                Input:
                T           : Partial MP2 doubles amplitude (#occ_basis,#occ_basis,#virt_basis,#virt_basis)
                mol         : gto molecular structure incorporating the basis set.           
                mo_coeff    : Coefficient matrix of the atomic orbitals (#basis,#basis)
                C_occ       : Atomic orbital coefficients of occupied molecular orbitals (#basis,#occ_basis)
                C_virt      : Atomic orbital coefficients of virtual molecular orbitals (#basis,#virt_basis)
                frozen_core : Option to use frozen core orbitals within the evaluation
                num_core    : Number of frozen core orbitals
                coords      : Given grid coordinates (#coords,3)
                
                Output:               
                ec          : Opposite spin based correlation energy density array evaluated with no DF.'''
               
                #Extraction of molecular orbital functions
                mol_orb_occ, mol_orb_virt = occ_virt_basis(mol, coords, mo_coeff,frozen_core,num_core)
                
                #Extracting Hartree integral
                A_integral=tensor_int_fake(coords, mol)
                
                #Handling partial MP2 doubles amplitude (a<->b)
                T_transpose = np.transpose(T,(0,1,3,2))

                with lock: #Lock tasks in one Thread
                        
                    if self.optimal_contract: # Optimized contraction
                        #First sum 
                        ec  = oe.contract('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,A_integral,T,optimize='auto',memory_limit=self.max_num_array)
                        #Second sum 
                        ec += oe.contract('ip,bp,mj,na,mnp,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,A_integral,T_transpose,optimize='auto',memory_limit=self.max_num_array)
                    else:
                        #First sum 
                        ec  = oe.contract('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,A_integral,T)
                        #Seond sum 
                        ec += oe.contract('ip,bp,mj,na,mnp,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,A_integral,T_transpose)  
                    
                    if self.verbose: #Saving size of largest intermediate contraction arrays
                        contract_info = oe.contract_path('ip,ap,mj,nb,mnp,ijab->p',mol_orb_occ,mol_orb_virt,
                                        C_occ,C_virt,A_integral,T)
                        self.contract_size.append(contract_info[1].largest_intermediate * 8 / (1024**3)) #in GB
                    
                return -0.5 * ec
                
            #Evaluating the correlation energy density on given grid
            if self.batch_size==0: #No batches specified
                if self.verbose:
                    print('Running correlation energy density evaluation on the full grid.')
                self.ec = ec_density_eval_noDF(T,self.mol,self.mo_coeff,C_occ,C_virt,self.frozen_core,
                                               self.num_core,self.coords_batches)

            else: #Batch-wise parallelization of the evaluation
                if self.verbose:
                    print('Running correlation energy density evaluation on batch wise separated grid.')
                    
                #Prestore arguments:
                partial_ec_density_eval_noDF = functools.partial(ec_density_eval_noDF, T, self.mol, self.mo_coeff, C_occ, 
                                        C_virt, self.frozen_core, self.num_core)
                
                #Parallelization
                with ThreadPoolExecutor(max_workers=self.cpus_per_task) as executor: 
                    
                    #Submitting tasks and saving them at the right index
                    self.ec_per_batch = [None]*len(self.coords_batches) #Preallocate list 
                    
                    #Parallelize with respect to number of CPUs
                    for idx in range(0, len(self.coords_batches), self.cpus_per_task):
                        if self.verbose:
                            print(f'Submission for batches {idx} until {idx+self.cpus_per_task}')
                        coords_batch_par=self.coords_batches[idx:idx+self.cpus_per_task] #Collect batches
                        #Submit execution to available CPUs per batch
                        futures_indx = {executor.submit(partial_ec_density_eval_noDF,batch): indx for indx, batch in enumerate(coords_batch_par)}
                        # futures = [executor.submit(partial_ec_density_eval_DF,batch) for batch in coords_batch_par]
                        for future in as_completed(futures_indx):
                            if self.verbose:
                                print(f'Finished batch evaluation {idx+futures_indx[future]}')
                            self.ec_per_batch[futures_indx[future]+idx]=jnp.array(future.result()) #Save results as jnp to ensure immutable
                        #Clear execution and cache
                        futures_indx.clear()   
                        gc.collect()
                
                #Combination of the resulting correlation energy density batches
                self.ec = jnp.concatenate(self.ec_per_batch, axis=0)
        
        end_time = time.time()
        print()
        print('Finished evaluation of the opposite spin based correlation energy density.')
        print('Elapsed total evaluation time: %.2f seconds' % np.abs(start_time-end_time))   
        print()
        if self.verbose: #Final printings
            #Save total memory usage of correlation density array:
            self.ec_size = (self.ec.size * self.ec.itemsize) / (1024**3) #in GB  
            if self.batch_size > 0: #Save batchwise size of the energy density array:     
                self.ec_per_batch_size = get_max_memory_object(self.ec_per_batch) / (1024**3) #in GB 
            else: #For no batches, batchewise size is equal to full array size.
                self.ec_per_batch_size = self.ec_size
            print('Final Memory usage:')
            print(f'Largest intermediate of the contraction path:{max(self.contract_size):.8f} GB')
            print(f'Correlation energy density array per batch: {self.ec_per_batch_size:.8f} GB')
            print(f'Full correlation energy density array: {self.ec_size:.8f} GB')
            print('-------------------------------------------------------------')
        
      
    '''MP2 correlation energy value'''
    @property
    def energy(self):
    
        #Evaluate the integral with the density function
        Ec_value = oe.contract('p,p->', self.ec, self.weights)
    
        return Ec_value
        
    '''MP2 correlation energy density evaluated on the grid'''
    @property    
    def array(self):
        '''ec(r)=1/rho(r) * ec^{MP2}(r)'''
        
        #Atomic orbitals evaluated on the grid 
        ao_value = numint.eval_ao(self.mol, self.coords, deriv=1)   
        
        # Evaluate electron density on same grid from atomic orbitals
        rho = numint.eval_rho(self.mol, ao_value[0], self.dm, xctype='LDA')

        ec = self.ec / rho 
        return ec           
    


#============================================#
#       Additional python functions          #
#============================================#

#Occupied and Virtual orbitals functions 
def orb_occ_virt(mo_occ):
    '''Extracting the number of occupied and virtual molecular orbitals from the occupation numbers.
    
    Input: 
    mo_occ : Molecular orbital occupation numbers (#basis,)
    
    Output:
    Nocc  : Number of occupied molecular orbitals
    Nvirt : Number of virtual molecular orbitals
    
    Example:
    >>>atom_geom = 'Ar 0 0 0'
    >>>basis = 'def2-svp'
    >>>Abasis = 'def2tzvpri' #aux basis for MP2 correlation energy
    >>>mol = gto.M(atom=atom_geom, basis=basis)
    >>>mf = dft.RKS(mol) 
    >>>mf.xc = 'hf'  
    >>>mf.kernel()
    >>>mo_occ=mf.mo_occ
    >>>Nocc, Nvirt = orb_occ_virt(mo_occ) 
    >>>print('%s occupied and %s virtual orbitals' % (Nocc,Nvirt))
    '''
    
    #Extracting the number of total and occupied orbitals
    Nocc  = mo_occ[mo_occ > 0].shape[0]
    Nvirt = mo_occ[mo_occ == 0].shape[0]
    
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
    

