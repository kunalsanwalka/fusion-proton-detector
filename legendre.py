#---------------------------------------------------------------------------
# legendre.py
#
# Contains:
# legP (function)
# legQ (function)
# dlegP0 (function)
# dlegQ0 (function)
#
# Depedencies:
# numpy
# scipy
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

def legP(l,z):
    """ Calculates the legendre function of the first kind P_l^0(z) this
    replaces the scipy function which only allows integer l
        :Param l: The subscript index of the Legendre function
        :type l: float
        :Param z: The argument of the Legendre function
        :type z: float
        :return: The first Legendre function order l at z
        :rtype: float
        """
    a = -l
    b = l+1
    c = 1.0
    x = 0.5*(1.0-z)
    return sp.special.hyp2f1(a,b,c,x)    

def legQ(l,z,m=0):
    """ Calculates the legendre function of the second kind P_l^0(z) this
    replaces the scipy function which only allows integer l
        :Param l: The subscript index of the Legendre function
        :type l: float
        :Param z: The argument of the Legendre function
        :type z: float
        :return: The second Legendre function order l at z
        :rtype: float
        """
    z2 = z*z
    gamr = sp.special.gamma(0.5*l + 1)/sp.special.gamma(0.5*(l+1))
    h1 = np.cos(0.5*np.pi*l)*sp.special.hyp2f1(0.5*(1-l),0.5*l+1,1.5,z2)
    h2 = np.sin(0.5*np.pi*l)*sp.special.hyp2f1(-0.5*l,0.5*(l+1),0.5,z2)
    return np.sqrt(np.pi)*(z*h1*gamr - h2/(2*gamr))    

# find first derivative of legendre functions at x=0.0
# see 8.6.3 and 8.6.4 of abramowitz and stegun
def dlegP0(l):
    """ Derivative of  the legendre function of the first kind 
    dP_l^0(0)/dz. This replaces the scipy function which only allows 
    integer l
        :Param l: The subscript index of the Legendre function
        :type l: float
        :return: The derivative of the first legendre function order l at
            z=0
        :rtype: float
        """
    return 2.0/np.sqrt(np.pi)*np.sin(0.5*np.pi*l)\
        *sp.special.gamma(0.5*l+1)/sp.special.gamma(0.5*(l+1.0))
    
def dlegQ0(l):
    """ Derivative of  the legendre function of the second kind 
    dQ_l^0(0)/dz. This replaces the scipy function which only allows 
    integer l
        :Param l: The subscript index of the Legendre function
        :type l: float
        :return: The derivative of the second legendre function order l at
            z=0
        :rtype: float
    """
    return np.sqrt(np.pi)*np.cos(0.5*np.pi*l)\
        *sp.special.gamma(0.5*l+1)/sp.special.gamma(0.5*(l+1.0))
