''' *** Main script for the exact exchange python function ***

Formalism taken from:
S. Vuckovic, T. J. P. Irons, A. Savin, A. M. Teale, and P. Gori-Giorgi, “Ex-
change-Correlation Functionals via Local Interpolation along the Adiabatic
Connection”, Journal of Chemical Theory and Computation 12, 2598-2610 (2016).
            
This scripts contains the python functionalities to obtain the exact exchange energy density
as an array and the corresponding exchange energy.                     

The evaluation requires a resulting scf class object and a given grid in xyz coordinates. 

Notation:
phi     - molecular orbitals
chi     - atomic orbitals

ij      - occupied molecular orbitals
ab      - virtual molecular orbitals
klmn    - atomic orbitals
p       - grid points
'''

import os
import sys
import numpy as np 
import opt_einsum as oe
import time
import jax.numpy as jnp

from pyscf import df, gto, lib
from pyscf.dft import numint
from concurrent.futures import ThreadPoolExecutor


### Exact exchange energy density ###
'''
In the exact case, the energy density of the exchange energy density reads

e_x(r) = -1/(4rho(r))*sum_{ij}[phi_j(r)*phi_i(r)*int(phi_i(r1)*phi_j(r1))/(|r-r1|)dr1)]
          
          
Expansion to atomic orbitals yields a tensor multiplication notation per grid point:

e_x(r_p) = e_xp
         = -1/(4rho(r))*sum_{klmn}[D_{km}*D_{ln}*chi_{kp}*chi_{lp}*int(chi_m*chi_n)/(|r-r1|)dr1)]
         = -1/(4rho(r))*sum_{klmn}[D_{km}*D_{ln}*chi_{kp}*chi_{lp}*A_{mnp}]
'''

class ex_eval:
    ''' Evalation of the exact exchange based energy density on a given grid of coords.
        Density fitting or batchwise parallelization are optional arguments.
    
        Example:
        
        >>>DF=True/False #Density fitting choice 
        >>>verbose=False
        >>>batch_size=5000 #Batch-wise parallelization 
        >>>kwargs = ex.ex_kwargs(batch_size,DF,verbose) #Extracting optional arguments
        >>>atom_geom = 'He 0 0 0'
        >>>basis = 'def2-tzvp'
        >>>Abasis = 'def2universaljkfit' #aux basis for exchange
        >>>mol = gto.M(atom=atom_geom)
        >>>grid_level = 3 
        >>>mol.basis = basis
        >>>mf = dft.RKS(mol)
        >>>mf.xc = 'pbe,pbe'
        >>>mf.grids.level = grid_level  # Grid level
        >>>mf.kernel()
        >>>args = ex.ex_args(mf,mol,Abasis,grids=grid_level) #Extracting input arguments
        >>>Ex=ex_eval(*args,*kwargs)
        >>>Ex_ref=ex_ref(mf)
        >>>print('Exact exchange based energy: %s' % Ex.energy)
        >>>print('Reference exchange energy: %s' % Ex_ref)
        >>>print('Absolute difference: %s' % np.abs(Ex.energy-Ex_ref))
        '''
    
    def __init__(self, dm, mol, Amol, coords, weights, batch_size=0, DF=False, verbose=False):
        '''
        *args* 
        dm      : Density matrix from a scf calculation (#basis,#basis)
        mol     : gto molecular structure incorporating the basis set.
        Amol    : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
        
        coords  : Grid coordinates (#coords,3)
        weights : Grid weights (#coords,3)
        
        
        **kwargs
        batch_size = 0    : Batch size to run the evaluation on, 0 is for no parallelization
        DF = True         : Density fitting option
        verbose = False   : Additional printings of time and memory statements
        
        '''
        
        #Saving arguments of the class
        self.dm                 = dm
        self.mol                = mol
        self.Amol               = Amol
        self.coords             = coords
        self.weights            = weights
        self.batch_size         = batch_size
        self.DF                 = DF
        self.verbose            = verbose

        #Check input:
        if isinstance(self.batch_size,int)==False or self.batch_size <0:  #Batch size option
            print('batch_size argument hast to be a positive integer or 0 for no parallelization')
            quit()    
            
        #Initialized printing:
        print('Exact exchange modelling.')
        time.sleep(0.5)
        if self.verbose:
            print('Evaluation parameters: ')
            print('Batch wise parallelization = ' + str(self.batch_size > 0))
            print('Density fitting = ' + str(self.DF))
            
        #Starting evaluation
        #====================#
        start_time = time.time()          
        
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
                print()
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
            '''Further expansion with a density fitted auxiliary basis set:

            A_{mnp}  = sum_{t}Q_{tmn}int psi_t(r')/(r_p-r')dr'
                     = sum_{t}Q_{tmn}I_tp   
                     
            chi_{kp}*chi_{lp} = Q_{skl} * psi_{sp} '''
            
            #Aux basis electrostatic integrals
            def aux_basis_int(coords, Amol):  
                ''' Tensor integral evaluation of the auxiliary basis set:
                I_t(r)=int psi_t(r1)/(|r1-r|)dr1
                evaluated using delta distributions with the Hartree potential
                
                Input: 
                coords  : grid coordinates (N,3),
                Amol    : gto molecular geometry with aux basis
            
                Output:
                integralvalue : Integral value for every aux basis at every grid point (#coords,#aux-basis)
                '''

                # Creating fake dirac delta charges for the hartree potential evaluation on the grid points
                fakemol = gto.fakemol_for_charges(coords, expnt=1e+16)
                mol1 = fakemol + Amol
                
                I = mol1.intor('int2c2e', shls_slice=(0,fakemol.nbas,fakemol.nbas,mol1.nbas))
                
                if self.verbose:
                    #Saving size of components
                    self.I_integral_size = I.size * I.itemsize / (1024**3) #in GB
                
                return I       
            
            #Function to evaluate of the correlation energy density array with DF
            def ex_density_eval_DF(coords, dm, df_coeff, Amol):
                '''Exchange energy density evaluation on a given grid
                
                -4* e_xp * rho_p = sum_{klmn}[D_{km}*D_{ln}*chi_{kp}*chi_{lp}*A_{mn}]
                                 = sum_{st}[sum_{klmn}[D_{km}D_{ln}Q_{skl}Q_{tmn}]psi_{sp}I_{tp}]
                
                Input:
                coords      : Given grid coordinates (#coords,3)
                dm          : Density matrix from a scf calculation (#basis,#basis) 
                df_coeff    : coefficient matrix from density fitting (#aux-basis,#basis,#basis)
                Amol        : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
       
                Output:
                ex          : Exchange energy density array evaluated on the given (batched) grid'''
                     
                #Auxiliary atomic orbitals evaluated on the grid       
                aux_ao_value = Amol.eval_gto("GTOval", coords)      
                     
                #Tensor product integrals (N, #aux-basis)
                I_integral=aux_basis_int(coords, Amol)
                
                #Obtaining exchange energy density
                self.ex = oe.contract('km,ln,skl,tmn,ps,pt->p',dm,dm,df_coeff,df_coeff,
                                      aux_ao_value,I_integral)
                
                return self.ex
            
            
            #Density fitting coefficients for the auxliliary basis set (#aux-basis, #basis, #basis)
            self.df_coeff = obtain_df_coef(self.mol, self.Amol) 
            
            if self.verbose: #Saving size of auxiliary basis coefficients
                self.df_coeff_size = self.df_coeff.size * self.df_coeff.itemsize / (1024**3) #in GB
            
            #Evaluating the exchange energy density on given grid
            if self.batch_size==0: #No batches specified
                if self.verbose:
                    print('Running exchange energy density evaluation on the full grid.')
                self.ex = ex_density_eval_DF(self.coords, self.dm, self.df_coeff, self.Amol)
                
            else: #Batch-wise parallelization of the evaluation
                if self.verbose:
                    print('Running exchange energy density evaluation on batch wise separated grid.')
                with ThreadPoolExecutor(max_workers=self.cpus_per_task) as executor:
                    self.ex_per_batch = list(executor.map(lambda coords: ex_density_eval_DF(coords, self.dm,
                                            self.df_coeff, self.Amol), self.coords_batches))  
                    
                #Combination of the resulting correlation energy density batches
                self.ex = jnp.concatenate(self.ex_per_batch, axis=0)
                       
        else:   #Without density fitting
            '''
            Expansion to atomic orbitals yields the tensor multiplication from above
            
            -4*e_x(r_p)*rho(r_p) = sum_{klmn}[D_{km}*D_{ln}*chi_{kp}*chi_{lp}*A_{mnp}]
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
            
            #Function to evaluate of the exchange energy density array without DF
            def ex_density_eval_noDF(coords, mol, dm):
                '''Exchange energy density evaluation on a given grid
                
                -4*e_x(r_p)*rho(r_p) = sum_{klmn}[D_{km}*D_{ln}*chi_{kp}*chi_{lp}*A_{mnp}]
                
                Input:
                coords      : Given grid coordinates (#coords,3)
                mol         : PySCF Mol, with aux. basis, e.g., Amol = df.addons.make_auxmol(mol, Abasis)
                dm          : Density matrix from a scf calculation (#basis,#basis)
                
                Output:
                ex          : Exchange energy density array evaluated on the given (batched) grid
                '''
        
                #Atomic orbitals evaluated on the grid  
                ao_value = numint.eval_ao(mol, coords, deriv=0)
                
                #Tensor integral 
                A = tensor_int_fake(coords, mol)  
                
                #Obtaining exchange energy density
                self.ex = oe.contract('km,ln,pk,pl,mnp->p',dm,dm,ao_value,ao_value,A)
                
                return self.ex
                 
            #Evaluating the exchange energy density on given grid
            if self.batch_size==0: #No batches specified
                if self.verbose:
                    print('Running exchange energy density evaluation on the full grid.')
                self.ex = ex_density_eval_noDF(self.coords, self.mol, self.dm)
                
            else: #Batch-wise parallelization of the evaluation
                if self.verbose:
                    print('Running exchange energy density evaluation on batch wise separated grid.')
                with ThreadPoolExecutor(max_workers=self.cpus_per_task) as executor:
                    self.ex_per_batch = list(executor.map(lambda coords: ex_density_eval_noDF(coords,
                                        self.mol, self.dm), self.coords_batches))  
                    
                #Combination of the resulting correlation energy density batches
                self.ex = jnp.concatenate(self.ex_per_batch, axis=0)

        end_time = time.time()
        print()
        print('Finished evaluation of the exchange energy density.')
        print('Elapsed total evaluation time: %.2f seconds' % np.abs(start_time-end_time))   
        
        if self.verbose: #Final printings
            #Save total memory usage of correlation density array:
            self.ex_size = (self.ex.size * self.ex.itemsize) / (1024**3) #in GB  
            if self.batch_size > 0: #Save batchwise size of the energy density array:     
                self.ex_per_batch_size = get_max_memory_object(self.ex_per_batch) / (1024**3) #in GB 
            else: #For no batches, batchewise size is equal to full array size.
                self.ex_per_batch_size = self.ex_size
            print()
            print('Memory usage:')
            print(f'Exchange energy density array per batch: {self.ex_per_batch_size:.8f} GB')
            print(f'Full exchange energy density array: {self.ex_size:.8f} GB')
            if DF:
                print(f'Auxiliary basis set coefficients: {self.df_coeff_size:.8f} GB')
    
    
    '''Exact exchange modelled energy value'''
    @property
    def energy(self):
        
        #Evaluate the integral 
        Ex_value = oe.contract('p,p->', -0.25 * self.ex, self.weights)
    
        return Ex_value
    
    '''Exact exchange modelled energy density evaluated on the grid'''
    @property
    def array(self):
        '''e_x(r) = -1/(4*rho(r)) * self.ex'''
        
        #Atomic orbitals evaluated on the grid  
        ao_value = numint.eval_ao(self.mol, self.coords, deriv=1) 
        
        # Evaluate electron density on same grid from atomic orbitals
        rho = numint.eval_rho(self.mol, ao_value[0], self.dm, xctype='LDA')
        
        ex = -0.25 * self.ex / rho 
        
        return ex





#============================================#
#       Additional python functions          #
#============================================#

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
          
