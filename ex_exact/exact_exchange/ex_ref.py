'''Functionalities to evaluate the reference exchange energy from a scf calculation'''

import opt_einsum as oe
import numpy as np 

#Exact exchange energy evaluation
def ex_ref_eval(mf):
    '''Exact evaluation of the exchange energy directly from the density matrix.
    
    Input: 
    mf  : Kernel of a RKS/RHF-Calculation
    
    Output:
    Ex  : Exchange energy value '''
    
    dm = mf.make_rdm1()                     #Density matrix (#basis,#basis)
    
    Ex =-1/4*oe.contract('ij,ij->',dm,mf.get_k())
    
    return Ex

#Exchange energy density for an example case
def ex_2elec(coords,weights):
    ''' Exchange energy density function for the example case of 2 electrons
    -> Spherically symmetric and spin unpolarized:
    
    ex(r)=-1/4*pi^(-(3/2))*int(exp(-r1^2)/(r-r1))dr1
    
    Input: 
    coords  : grid coordinates (#coords,3)
    weights : grid weights     (#coords,3)

    Output: 
    ex      : Array of function values on the grid (#coords,)'''
        
    const = -1/4*np.pi**(-3/2) #Constant
    N =len(coords) #N = number of grid points
    ex = np.zeros(N) #Allocate ex
    
    for n in np.arange(N):
        r = coords[n] #grid point    
        dist = r-coords #distance function in the denominator
        rdist = np.linalg.norm(dist, axis=1) #Norm of the distance function
        rexp = np.exp(-np.linalg.norm(coords, axis=1)**2) #exp(-r²)

        #Employ a zero mask to avoid division by zero
        zeromask = rdist > 10**(-14)
        intvalue = rexp[zeromask] / rdist[zeromask] #Evaluation of the integrand
        integralvalue = np.einsum('i,i->', intvalue,weights[zeromask]) #Evaluation of the integral
        ex[n] = 2*const *integralvalue  #Finaly evaluation of the density function on the grid
    return ex

#Exchange energy evaluation from the example case
def ex_2elec_eval(coords, weights, rho):
    ''' Exchange energy evaluation for 2 electrons:
        E_x=\int w_0(r)*rho(r)
        
        Input: 
        coords  : Grid coordinates
        weights : Grid weights
        rho     : Density function (without derivatives) array on the same grid
        
        Output:
        Ex : Exchange energy value'''
        
    ex = ex_2elec(coords,weights)  #Extract the exact exchange density function from above
    Ex = np.einsum('i,i,i->',ex,rho, weights)  #Evaluate the integral with the density function

    return Ex

