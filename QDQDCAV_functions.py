''' Functions used in QD-QD system with shared phonon environment '''
import numpy as np
from scipy import integrate
from scipy.integrate import dblquad
from scipy import special
import warnings
# sharebath=1
warnings.filterwarnings('ignore')

# def DiagM(g, w_qd1, w_qd2):
#     """Diagonalise the QD-QD Liouvillian"""
#     w1=(w_qd1+w_qd2)/2 - np.sqrt(g**2 +(0.5*(w_qd1-w_qd2))**2)
#     w2=(w_qd1+w_qd2)/2 + np.sqrt(g**2 +(0.5*(w_qd1-w_qd2))**2)
#     Delta_xc=np.sqrt((0.5*(w_qd1-w_qd2))**2+g**2)-0.5*(w_qd1-w_qd2)
#     alpha=Delta_xc/np.sqrt(Delta_xc**2 + g**2)
#     beta=g/np.sqrt(Delta_xc**2 + g**2)
    
#     ws=np.array([w1,w2])
#     Y=np.matrix([[alpha, beta],
#                     [-beta, alpha]])
#     Yt=np.transpose(Y)
#     return ws, Y, Yt



def LFpol(g1, g2,gd, w_qd1, w_qd2, w_c):  #using python package to diagonalise matrix equivalent to analyitcally solving - although has some 1e-19 terms instead of 0
    """Diagonalise the QD-QD Liouvillian"""
    Hxx=np.matrix([[w_qd1, gd, g1],[gd, w_qd2, g2],[g1, g2, w_c]]) 

    return Hxx



def Sint(w,T,j0,w0):
    '''Huang-Rhys integrand for in=im'''
    HRintegrand= j0* w*np.exp(-w**2/w0**2) * (2*(1/(np.exp(w/T)-1))+1)
    return HRintegrand
    
def S_inin(T,j0,w0):
    ''' Computing the Huang-Rhys factor'''
    def re_fun(w, T, j0, w0):
        return np.real(Sint(w, T, j0, w0))
    def im_fun(w, T, j0, w0):
        return np.imag(Sint(w, T, j0, w0))
    w1 = 0 #limits
    w2 = np.inf
    re_int = integrate.quad(re_fun, w1, w2, args=(T, j0, w0))
    im_int = integrate.quad(im_fun, w1, w2, args=(T, j0, w0))
    return re_int[0] + 1j*im_int[0]


def Sint_inim(w,T,j0_1,w0,r0,Vs):
    '''Huang-Rhys integrand for in=/=im'''
    HRintegrand= j0_1*np.exp(-w**2/w0**2) * (2*(1/(np.exp(w/T)-1))+1)* np.sin(w*r0/Vs)
    return HRintegrand

def S_inim(T,j0_1,w0,r0,Vs):
    ''' Computing the Huang-Rhys factor'''
    def re_fun(w, T, j0_1, w0,r0,Vs):
        return np.real(Sint_inim(w,T,j0_1,w0,r0,Vs))
    def im_fun(w, T, j0_1, w0,r0,Vs):
        return np.imag(Sint_inim(w,T,j0_1,w0,r0,Vs))
     #limits
    w1 = 0
    w2 = np.inf
    re_int = integrate.quad(re_fun, w1, w2, args=(T, j0_1, w0, r0, Vs))#, epsabs=1e-10, epsrel=1e-10)
    im_int = integrate.quad(im_fun, w1, w2, args=(T, j0_1, w0, r0, Vs))
    return re_int[0] + 1j*im_int[0]
    

def PolaronShift(j0,w0):
    '''Polaron shift for in=im'''
    return -(j0*np.sqrt(np.pi)/4) * w0**3 

def PolaronShift_inim(j0,w0,r0,l):
    '''Polaron shift for in=/=im'''
    return -(j0*np.sqrt(np.pi)/4) * w0**3 *np.exp(-r0**2/(2*l**2))



def phi_inin_integrand(w,t,j0,w0,T):
    '''Kbb (broadband contribution to the cumulant) for in=im'''
    integrand= j0*w*np.exp(-(w/(w0))**2)*((1/(np.exp(w/T)-1)) *np.exp(complex(0,w*t)) + ((1/(np.exp(w/T)-1))+1)*np.exp(complex(0,-w*t)))
    return integrand
def phi_inin(t,j0,w0,T):
    '''Computing the rapidly decaying part of the cumulant'''
    def re_fun(w, t, j0, w0, T):
        return np.real(phi_inin_integrand(w,t,j0,w0,T))
    def im_fun(w, t, j0, w0, T):
        return np.imag(phi_inin_integrand(w,t,j0,w0,T))
    re_int = integrate.quad(re_fun, 0, np.inf, args=(t, j0, w0, T), limit=1000)
    im_int = integrate.quad(im_fun, 0, np.inf, args=(t, j0, w0, T), limit=1000)
    return re_int[0] + 1j*im_int[0]


def phi_inim_integrand(w,t,j0_1,w0,T,r0,Vs):
    '''Kbb (broadband contribution to the cumulant) for in=/=im'''
    integrand= j0_1*np.exp(-w**2/w0**2) * ((1/(np.exp(w/T)-1)) *np.exp(complex(0,w*t)) + ((1/(np.exp(w/T)-1))+1)*np.exp(complex(0,-w*t)))*np.sin(w*r0/Vs)
    return integrand

def phi_inim(t,j0_1,w0,T,r0,Vs):
    '''Computing the rapidly decaying part of the cumulant'''
    def re_fun(w, t, j0_1, w0, T,r0,Vs):
        return np.real(phi_inim_integrand(w,t,j0_1,w0,T,r0,Vs))
    def im_fun(w, t, j0_1, w0, T,r0,Vs):
        return np.imag(phi_inim_integrand(w,t,j0_1,w0,T,r0,Vs))
    re_int = integrate.quad(re_fun, 0, np.inf, args=(t, j0_1,w0,T,r0,Vs), limit=1000)
    im_int = integrate.quad(im_fun, 0, np.inf, args=(t, j0_1,w0,T,r0,Vs), limit=1000)
    return (re_int[0] + 1j*im_int[0])


def K11_smartie_integrand(w, t,j0,l,lp,Vs,T):
    N=((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)
    eta= special.wofz((w* np.sqrt(l**2 - lp**2))/(np.sqrt(2)*Vs))*np.exp(-(lp**2 *w**2)/(2*Vs**2)) - np.exp(-(l**2 *w**2)/(2*Vs**2))
    return N* eta


def K11_smartie(t,j0,l,lp,Vs,T):
    '''Computing the rapidly decaying part of the cumulant'''
    def re_fun(w, t, j0, l,lp,Vs, T):
        return np.real(K11_smartie_integrand(w, t,j0,l,lp,Vs,T))
    def im_fun(w, t, j0, l,lp,Vs, T):
        return np.imag(K11_smartie_integrand(w, t,j0,l,lp,Vs,T))
    re_int = integrate.quad(re_fun, 0,np.inf, args=(t, j0, l,lp,Vs, T), limit=1000, epsabs=0, epsrel=1e-12)
    im_int = integrate.quad(im_fun, 0, np.inf, args=(t, j0, l,lp,Vs, T), limit=1000,epsabs=0, epsrel=1e-12)
    return ((-1j*j0*Vs*np.sqrt(np.pi)) /(np.sqrt(2)*np.sqrt(l**2 - lp**2))) *(re_int[0] + 1j*im_int[0])#, re_int[1], 1j*im_int[1]



def K12_smartie_integrand(w, t,j0,l,lp,Vs,T,r0):
    N=((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)
    alph= 1j*np.sqrt(l**2 - lp**2)*w/(np.sqrt(2)*Vs)
    bet=w*r0/Vs
    # eta= -special.wofz(1j*alph - bet/(2*alph)) * np.exp(-lp**2 *w**2 / (2*Vs**2)) * np.exp(-1j*bet) + np.exp(-alph**2) * np.exp(-lp**2 * w**2 /(2*Vs**2)) * np.exp(-(bet/(2*alph))**2) + special.wofz(-1j*alph - bet/(2*alph))  * np.exp(-lp**2 *w**2 / (2*Vs**2)) * np.exp(1j*bet) - np.exp(-alph**2) * np.exp(-lp**2 * w**2 /(2*Vs**2)) * np.exp(-(bet/(2*alph))**2)
    eta= -special.wofz(1j*alph - bet/(2*alph)) * np.exp(-lp**2 *w**2 / (2*Vs**2)) * np.exp(-1j*bet) + special.wofz(-1j*alph - bet/(2*alph))  * np.exp(-lp**2 *w**2 / (2*Vs**2)) * np.exp(1j*bet) 

    return ((-1j*j0*Vs*np.sqrt(np.pi) ) /(2**(3/2)*np.sqrt(l**2 - lp**2))) *N* eta


def K12_smartie(t,j0,l,lp,Vs,T,r0):
    '''Computing the rapidly decaying part of the cumulant'''
    def re_fun(w, t, j0, l,lp,Vs, T,r0):
        return np.real(K12_smartie_integrand(w, t,j0,l,lp,Vs,T,r0))
    def im_fun(w, t, j0, l,lp,Vs, T,r0):
        return np.imag(K12_smartie_integrand(w, t,j0,l,lp,Vs,T,r0))
    re_int = integrate.quad(re_fun, 0,np.inf, args=(t, j0, l,lp,Vs, T,r0), limit=1000, epsrel = 1e-13, epsabs = 0)
    im_int = integrate.quad(im_fun, 0, np.inf, args=(t, j0, l,lp,Vs, T,r0), limit=1000, epsrel = 1e-13, epsabs = 0)
    return (re_int[0] + 1j*im_int[0])#, re_int[1] + 1j* im_int[1]










# def K11_smartie_integrand(t,j0,l,lp,Vs,T):
#     return lambda theta,w,t,j0,l,lp,Vs,T: np.real(((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)*np.sin(theta)*w*j0/2 *np.exp(- (l**2 * w**2 *np.sin(theta)**2/(2*Vs**2)))*np.exp(- (lp**2 * w**2 *np.cos(theta)**2/(2*Vs**2)))), lambda theta,w,t,j0,l,lp,Vs,T: np.imag(((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)*np.sin(theta)*w*j0/2 *np.exp(- (l**2 * w**2 *np.sin(theta)**2/(2*Vs**2)))*np.exp(- (lp**2 * w**2 *np.cos(theta)**2/(2*Vs**2))))

# def K11_smartie(t,j0,l,lp,Vs,T):
#     re_int=dblquad(K11_smartie_integrand(t,j0,l,lp,Vs,T)[0], 0, np.inf, 0, np.pi, args=(t,j0,l,lp,Vs,T))
#     im_int=dblquad(K11_smartie_integrand(t,j0,l,lp,Vs,T)[1], 0,np.inf,0,np.pi,args=(t,j0,l,lp,Vs,T))
#     return re_int[0] + 1j*im_int[0]

# def K12_smartie_integrand(t,j0,l,lp,Vs,T, r0):
#     return lambda theta,w,t,j0,l,lp,Vs,T,r0: np.real(((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)*np.sin(theta)*w*j0/2 *np.exp(- (l**2 * w**2 *np.sin(theta)**2/(2*Vs**2)))*np.exp(- (lp**2 * w**2 *np.cos(theta)**2/(2*Vs**2)))* np.exp(1j*w*r0*np.cos(theta) / Vs)), lambda theta,w,t,j0,l,lp,Vs,T,r0: np.imag(((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)*np.sin(theta)*w*j0/2 *np.exp(- (l**2 * w**2 *np.sin(theta)**2/(2*Vs**2)))*np.exp(- (lp**2 * w**2 *np.cos(theta)**2/(2*Vs**2))) * np.exp(1j*w*r0*np.cos(theta) / Vs))

# def K12_smartie(t,j0,l,lp,Vs,T,r0):
#     re_int=dblquad(K12_smartie_integrand(t,j0,l,lp,Vs,T,r0)[0], 0, np.inf, 0, np.pi, args=(t,j0,l,lp,Vs,T,r0))
#     im_int=dblquad(K12_smartie_integrand(t,j0,l,lp,Vs,T,r0)[1], 0,np.inf,0,np.pi,args=(t,j0,l,lp,Vs,T,r0))
#     return re_int[0] + 1j*im_int[0]
