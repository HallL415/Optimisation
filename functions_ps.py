#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:47:44 2022

@author: luba
"""


import numpy as np
from scipy import integrate


def DiagM(g, wx, gx, gc, det, omp):
    """Diagonalise the JC Liouvillian"""
    ###############################

    delta = det
    wc = wx+delta
    omx = (wx-1j*gx)
    omc = (wc-1j*gc)

    ###############################

    def delt(n):
        return np.sqrt(((omx-omc)/2)**2+n*g**2)
    def l(n):
        return (omx+(2*n-1)*omc)/2-delt(n)
    def m(n):
        return (omx+(2*n-1)*omc)/2+delt(n)
    def gam(n):
        return np.sqrt((delt(n)-(omx-omc)/2)**2+n*g**2)
    def a(n):
        return (delt(n)-(omx-omc)/2)/gam(n)
    def b(n):
        return np.sqrt(n)*g/gam(n)

    w1 = l(1)
    w2 = m(1)

    ww = np.array([w1, w2])

    Y = np.matrix([[a(1), b(1)],
                    [-b(1), a(1)]])

    Yt=np.transpose(Y)

    A=np.matrix([[ww[0], 0],
                 [0, ww[1]]])


    return A, ww, Y, Yt

######################################################################

def Sint(w, T, J0, w0):
    """Integrand for Huang Rhys"""
    kb=1#8.617333*10**(-5)
    return J0*w*np.exp(-w**2/w0**2)/np.tanh(w/(2*kb*T))

#@njit
def SHR(T, J0, w0):
    """Huang Rhys factor"""
    def re_fun(w, T, J0, w0):
        return np.real(Sint(w, T, J0, w0))
    def im_fun(w, T, J0, w0):
        return np.imag(Sint(w, T, J0, w0))
    #limits
    w1 = 0
    w2 = np.inf
    re_int = integrate.quad(re_fun, w1, w2, args=(T, J0, w0), epsabs=1e-15, epsrel=1e-15)
    im_int = integrate.quad(im_fun, w1, w2, args=(T, J0, w0), epsabs=1e-15, epsrel=1e-15)
    return re_int[0] + 1j*im_int[0]

##########################################################



def Oint(w, J0, w0):
    """Integrand for polaron shift"""
    return -J0 * w**2 * np.exp(-w ** 2 / w0 ** 2)

#@njit
def OMP(J0, w0):
    """Polaron Shift"""
    #return compint2(Oint, 1e-10, 0.1, 10000, J0, w0)
    def re_fun(w):
        return np.real(Oint(w, J0, w0))

    def im_fun(w):
        return np.imag(Oint(w, J0, w0))
    w1 = 0
    w2 = np.inf
    re_int = integrate.quad(re_fun, w1, w2, epsabs=1e-8, epsrel=1e-8)
    im_int = integrate.quad(im_fun, w1, w2, epsabs=1e-8, epsrel=1e-8)
    return re_int[0] + 1j * im_int[0]


def wN(w, T):
    """Boson distribution fn * w"""
    kb=1#8.617333*10**(-5)
    return w/(np.exp(w/(kb*T))-1)




def integrand(w, t, T, J0, w0):
    """Integrand for Kbb2"""
    I0= J0*(wN(w, T)*np.exp(1j*w*t)+(wN(w, T)+w)*np.exp(-1j*w*t))*np.exp(-w**2/w0**2)
    return I0


def Kbb2(t, T, J0, w0):
    """The cumulant"""
    w1 = 0
    w2 = np.inf
    def re_fun(w, t, T, J0, w0):
        return np.real(integrand(w, t, T, J0, w0))
    def im_fun(w, t, T, J0, w0):
        return np.imag(integrand(w, t, T, J0, w0))
    re_int = integrate.quad(re_fun, w1, w2, args=(t, T, J0, w0), limit=1000)
    im_int = integrate.quad(im_fun, w1, w2, args=(t, T, J0, w0), limit=1000)
    return re_int[0] + 1j*im_int[0]



def integrand2(w, t, T, J0, w0):
    """Integrand for Kbb2"""
    N=((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)
    # N=((1/(np.exp(w/T)))*(np.exp(1j*w*t)) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)) )
    I0= J0*N*np.exp(-w**2/w0**2)*w 
    return I0

def Kbb22(t, T, J0, w0):
    """The cumulant"""
    w1 = 0
    w2 = np.inf
    def re_fun(w, t, T, J0, w0):
        return np.real(integrand2(w, t, T, J0, w0))
    def im_fun(w, t, T, J0, w0):
        return np.imag(integrand2(w, t, T, J0, w0))
    re_int = integrate.quad(re_fun, w1, w2, args=(t, T, J0, w0), limit=1000)
    im_int = integrate.quad(im_fun, w1, w2, args=(t, T, J0, w0), limit=1000)
    return re_int[0] + 1j*im_int[0]

def integrand222(t,r0,Vs,J0,w0,T):
    
    return lambda theta,w,t,r0,Vs,J0,w0,T: np.real( J0*((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)*np.exp(-w**2/w0**2)*w *(1-np.exp(1j*(w*r0/Vs)*np.cos(theta)))*(1-np.exp(-1j*(w*r0/Vs)*np.cos(theta)))), lambda theta,w,t,r0,Vs,J0,w0,T:np.imag( J0*((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)*np.exp(-w**2/w0**2)*w *(1-np.exp(1j*(w*r0/Vs)*np.cos(theta)))*(1-np.exp(-1j*(w*r0/Vs)*np.cos(theta)))   )

# def integrand222(w,t,r0,Vs,J0,w0,T):
    
#     return J0*((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)*np.exp(-w**2/w0**2)*w 



from scipy.integrate import dblquad, nquad

def Kbb222(t,r0,Vs,J0,w0,T):
    re_int=dblquad(integrand222(t,r0,Vs,J0,w0,T)[0], 0, np.inf, 0, np.pi, args=(t,r0,Vs,J0,w0,T))
    im_int=dblquad(integrand222(t,r0,Vs,J0,w0,T)[1], 0,np.inf,0,np.pi,args=(t,r0,Vs,J0,w0,T))
    # def re_fun(w,t,r0,Vs,J0,w0,T):
    #     return np.real(integrand222(w,t,r0,Vs,J0,w0,T))
    # def im_fun(w,t,r0,Vs,J0,w0,T):
    #     return np.imag(integrand222(w,t,r0,Vs,J0,w0,T))
    # re_int=integrate.quad(re_fun, 0, np.inf, args=(t,r0,Vs,J0,w0,T))
    # im_int=integrate.quad(im_fun, 0,np.inf,args=(t,r0,Vs,J0,w0,T))
    return re_int[0] + 1j*im_int[0]



