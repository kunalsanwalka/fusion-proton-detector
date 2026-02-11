#---------------------------------------------------------------------------
# egedal_f_obj.py
#
# Contains:
# egedal_f (class):
# A class used for the computation of the distribution function appearing
# in Egedal et al., Nucl. Fusion 62 (2022) 126053. In addition to computing
# the distribution function some useful moments are also computed.
#
# Depedencies:
# numpy
# scipy
# legendre
# 
# Author:
# Sam Frank
# 
# Copyright (c) 2025 Realta Fusion Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# --------------------------------------------------------------------------

import scipy as sp
import numpy as np
from legendre import *

m_p = 1.6726e-27 #[kg]
m_e = 9.1093e-31 #[kg]
eCharge = 1.60217e-19 #[C]
eps0 = 8.8542e-12 #[F/m]

class egedal_f():
    """This is a class representation of the Egedal distribution function 
       appearing in Eqn. (14) of Egedal et al., Nucl. Fusion 62 (2022) 
       126053. In addition to computing the distribution function it can 
       do a few useful integrals for moments of the distribution function 
       versus b that appear in Frank et al. PoP (2025) Submitted.

    :param R_m: The magnetic mirror ratio R_m = B_m/B_0, where B_0 is the
        magnetic field at the midplane and B_m is the magnetic field at
        the magnetic mirror throat. Optional, to avoid unnecessary 
        computation, but must be set eventually.
    :type R_m: float, optional
    :param E_NBI: The NBI energy in units [keV]. Optional, with a default
        value of 25.0, equivalent to the NBI energy used in WHAM.
    :type E_NBI: float, optional
    :param theta_NBI: The NBI angle in units [rad] referenced to its angle 
        in velocity-space at the magnetic midplane. Optional with a default 
        value of pi/4.0.
    :type theta_NBI: float, optional
    :param T_e: The electron temperature in units [keV]. Optional with a 
        default value of 1.0.
    :type T_e: float, optional
    :param mu_i: The mass of the ions in units [amu]. Optional with a 
        default value of 2.0.
    :type mu_i: float, optional
    :param Z_eff: The effective charge of the ions. Optional with a default
        value of 1.0.
    :type Z_eff: float, optional
    :param N_j: The number of terms used in the summation solution for f. 
        Optional with a default value of 6.
    :type N_j: int, optional
    :param lookup: NOT YET IMPLEMENTED. Lookup table name. If set to a value
        other than None, the code will look for a hdf5 lookup table with this
        name. 
    :type lookup: str, optional
    """

    def __init__(self,R_m=None,E_NBI=25.0,theta_NBI=np.pi/4.0,\
                 T_e=1.0,mu_i=2.0,Z_eff=1.0,N_j=6, \
                 lookup=None):
        
        self.N_j = N_j
        if lookup != None:
            self.lookup = lookup
            # not yet implemented throw error
            raise ValueError("Lookup table function not yet implemented")
        if R_m != None:
            self.R_m = R_m
        self.E_NBI = E_NBI
        self.theta_NBI = theta_NBI
        self.T_e = T_e
        self.m_i = mu_i*m_p
        self.Z_eff = Z_eff

    # input parameters with setters
    @property 
    def R_m(self):
        return self._R_m

    @R_m.setter
    def R_m(self, val):
        self._R_m = val
        if hasattr(self, "lookup"):
            self._CallLookupTable()
        else:
            self._SetConstants()
            
    @property
    def N_j(self):
        return self._N_j

    @N_j.setter
    def N_j(self,val):
        self._N_j = val
        self._a = np.zeros(self.N_j)
        self._b = np.zeros(self.N_j)
        self._l = np.zeros(self.N_j)
        self._lamb = np.zeros(self.N_j)
        self._alpha = np.zeros(self.N_j)
        if hasattr(self,"_R_m"):
            if hasattr(self, "lookup"):
                self._CallLookupTable()
            else:
                self._SetConstants()

    # internal constants that I don't want users
    # to be allowed to change but users may want
    # to look at
    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def l(self):
        return self._l

    @property
    def alpha(self):
        return self._alpha

    @property
    def lamb(self):
        return self._lamb

    # internal functions not intended to be called by the user
    # during normal code execution
    def _CallLookupTable(self):
        # Call lookup table routine here and pull a,b,l,etc from table
        # for a given value of R_m. Not yet implemented for lack of
        # time.
        pass

    def _M_lamjRoots(self,l,x):
        # Solves system of equations (6) in Egedal (2022) and computes
        # M_lambda
        
        # find legendre functions for j and j+1
        pl0 = legP(l,0.0)
        ql0 = legQ(l,0.0)  
        dpl0 = dlegP0(l)
        dql0 = dlegQ0(l)
    
        # calculate a & b by substitution
        b = 1.0/(ql0-pl0*(dql0/dpl0))
        a = -dql0/dpl0 * b
        M = a*legP(l,x) + b*legQ(l,x)
        return M

    def _abalpha(self,l):
        # Computes normalization constants a, b, and alpha_j found in
        # Egedal (2022)
        
        # find legendre functions for j and j+1
        pl0 = legP(l,0.0)
        ql0 = legQ(l,0.0)  
        dpl0 = dlegP0(l)
        dql0 = dlegQ0(l)

        # calculate a,b, & alpha
        b = 1.0/(ql0-pl0*(dql0/dpl0))
        a = -dql0/dpl0 * b
        f = lambda x: self._M_lamjRoots(l,x)\
            *np.conj(self._M_lamjRoots(l,x))
        alpha = sp.integrate.quad(f,0.0,self.xi_TP,limit=500)[0]
        return a,b,alpha

    def _SetConstants(self,l_max=100.01,Npts=2000):
        # Finds constants a, b, l, lambda, and alpha described in Sec. 2
        # of Egedal (2022) for a given value of R_m. This is called
        # whenever the classes R_m is set to reset these constants.
        # There are adjustable parameters l_max and Npts related to
        # the computation of these contants, but they should not need
        # to be adjusted under normal execution

        # set passing trapped pitch angle boundary
        R_m = self.R_m
        xi_TP = np.sqrt(1.0-1.0/R_m)
        self.xi_TP = xi_TP

        # find roots of M_lambdaj which satisfy eigenvalue problem
        l_arr = np.linspace(1e-6,l_max,Npts)
        j = 0
        signLast = -2
        lLast = 1e-6
        for l in l_arr:
            M = self._M_lamjRoots(l,xi_TP)
            sign = np.sign(M)
            # if sign of M switched, there is a root. zero in on it with
            # root finder for high-accuracy solution
            if signLast!=sign and signLast!=-2:
                lRoot = sp.optimize.brentq( \
                                    lambda x: self._M_lamjRoots(x,xi_TP) \
                                            ,lLast,l)
                # set constants at the root
                a,b,alpha = self._abalpha(lRoot)
                self._a[j] = a
                self._b[j] = b
                self._l[j] = lRoot
                self._lamb[j]  = lRoot*(lRoot+1.0)
                self._alpha[j] = alpha        
                j += 1
                if j == self.N_j:
                    return
            signLast = sign
            lLast = l

    # external functions to be called by the user
    def M(self,x,j):
        """ The value of sum of Legendre functions function M found in 
            Egedal (2022) section 2.
        :param x: the argument of function M
        :type x: float
        :param j: The integer number of the term in the series expansion
        :type j: int
        :return: The value of M_j(x)
        :rtype: float
        """ 
        l = self.l[j]
        return self.a[j]*legP(l,x)+self.b[j]*legQ(l,x)
    
    def S(self,j):
        """ The value of NBI source term S_j*4*pi*alpha_j for an 
            NBI source S approximated as a delta function. I've tested 
            this against a finite theta NBI source and found only very 
            minor differences in solution over all relevant plasma 
            parameters. Note: I have not normalized to finite NBI power 
            here. This can be done pretty easily by integrating the second
            moment of the source term in Egedal Eqn. (10) and multiplying 
            by the desired NBI power in Watts divided by the result of the 
            integration.
        :param j: the integer number of the term in the series expansion
        :type j: int
        :return: The value of S_j*4*pi*alpha_j (normalization by 
        :rtype: float
        """
        return self.M(np.cos(self.theta_NBI),j)
        
    def f(self,v,xi):
        """ The value of the distribution function f at some value
            of pitch angle xi and velocity v. Note: I have not normalized
            f by tau_s here as this code was intended for use in an 
            equilibrium reconstruction package, and the normalization was
            applied in the equilibrium reconstruction code empirically. 
            If you want to do this here, you'll need to do it yourself.
        :param v: The velocity in units [m/s] at which the distribution
           function should be calculated (must satisfy v <= v_NBI).
        :type v: float
        :param xi: The pitch angle v_par/v at which the distribution function
           should be calculated.
        :type xi: float
        :return: The distribution function f (not presently normalized)
        :rtype: float
        """
        if self.R_m is None:
            raise ValueError("You must set R_m before calling f()")

        # pull constants which were precomputed from the object
        m_i = self.m_i
        lamb = self.lamb
        alpha = self.alpha
        xi[xi>self.xi_TP] = self.xi_TP

        # compute physical prefactors
        # take loglambi/loglambe ~ 1
        v_c = ((3*np.sqrt(np.pi)*m_e*self.Z_eff)/(4.0*m_i))**(1/3)\
            *np.sqrt(2.0e3*eCharge*self.T_e/m_e)
        v0 = np.sqrt(2.0e3*eCharge*self.E_NBI/m_i)
        if np.any(v > v0):
            raise ValueError("v specified is greater than NBI velocity")
        v03 = v0**3
        v3 = v**3
        v_c3 = v_c**3
        u = ((v03 + v_c3)*(v3/v03)/(v3 + v_c3))**(self.Z_eff/6)

        # preallocate distribution function then compute
        # series solution found in Eqn. (14) of Egedal
        dist = np.zeros_like(v)
        fac = 1.0/(v3+v_c3)
        for j in range(self.N_j):
            S = self.S(j)/(4*np.pi*alpha[j])
            M = self.M(xi,j)
            dist += S*M*u**(lamb[j])
        dist *= fac 
        dist[dist<0] = 0 # physically f>0. any values<0 from error in sum.
                         # remove as numerical kludge
        return dist            

    def n(self,b,Nv=200,Nxi=200):
        """ The value of n at some point in b=B/B_0. This routine uses
        a simple sampled based integration of 0th moment to obtain. There
        :Param b: The velocity in units [m/s] at which the distribution
           function should be calculated (must satisfy v <= v_NBI).
        :type b: float
        :Param Nv: The number of samples in v used in the integration
        :type Nv: int, optional
        :Param Nxi: The number of samples in xi used in the integration
        :type Nxi: int, optional
        :return: The density n (not presently norm. fix dist. norm.)
        :rtype: float
        """
        #compute bounds on integration arrays
        xi_TPLoc = np.sqrt(1.0-b/self.R_m)
        v0 = np.sqrt(2.0e3*eCharge*self.E_NBI/self.m_i)

        #set up integration arrays
        xi_arr = np.linspace(0,xi_TPLoc,Nxi)
        v_arr = np.linspace(0,v0,Nv)
        Xi,V = np.meshgrid(xi_arr,v_arr)

        #set up integration kernel
        xi0 = lambda xi: np.sqrt((xi*xi - (1.0-b))/b)
        Xi0 = xi0(Xi)         
        nKern = self.f(V,Xi0)*V**2

        #integrate with sample based integration
        n = 4.0*np.pi\
            *sp.integrate.simpson(sp.integrate.simpson(nKern,xi_arr,axis=0),v_arr)

        return n
        
    def p(self,b,Nv=200,Nxi=200):
        """ The value of p at some point in b=B/B_0. This routine uses
        a simple sampled based integration of 0th moment to obtain. There
        :Param b: The velocity in units [m/s] at which the distribution
           function should be calculated (must satisfy v <= v_NBI).
        :type b: float
        :Param Nv: The number of samples in v used in the integration
        :type Nv: int, optional
        :Param Nxi: The number of samples in xi used in the integration
        :type Nxi: int, optional
        :return: The pressures p_prp and p_par (not presently norm. 
           fix dist. norm.)
        :rtype: list
        """
        #compute bounds on integration arrays
        xi_TPLoc = np.sqrt(1.0-b/self.R_m)
        v0 = np.sqrt(2.0e3*eCharge*self.E_NBI/self.m_i)

        #set up integration arrays
        xi_arr = np.linspace(0,xi_TPLoc,Nxi)
        v_arr = np.linspace(0,v0,Nv)
        Xi,V = np.meshgrid(xi_arr,v_arr)

        #set up integration kernels
        xi0 = lambda xi: np.sqrt((xi*xi - (1.0-b))/b)
        Xi0 = xi0(Xi)         
        fpiece = self.f(V,Xi0)*V**4
        Xi2 = Xi*Xi
        p_parKern = fpiece*Xi2
        p_prpKern = fpiece*(1-Xi2)

        #integrate with sampled based simpson integration (more consistent than quadrature for high N_j)
        p_par = 4.0*np.pi*self.m_i\
            *sp.integrate.simpson(sp.integrate.simpson(p_parKern,xi_arr,axis=0),v_arr)
        p_prp = 2.0*np.pi*self.m_i\
            *sp.integrate.simpson(sp.integrate.simpson(p_prpKern,xi_arr,axis=0),v_arr)
        
        return p_par, p_prp
