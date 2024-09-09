''' Functions used in QD-QD system with shared phonon environment '''
import numpy as np
from scipy import integrate
from scipy.integrate import dblquad, quad
from scipy import special
import sympy as smp
import warnings
# sharebath=1
warnings.filterwarnings('ignore')

def LFpop(g, om1, om2,gam1,gam2):
    """Return Forster Liouvilian for populations"""
    LF=np.matrix([[0,-2*1j*gam1,0,0,-2*1j*gam2],
                 [0,2*1j*gam1,-g,g,0],
                 [0,-g,om1-om2+1j*gam1+1j*gam2,0,g],
                 [0,g,0,om2-om1+1j*gam1+1j*gam2,-g],
                 [0,0,g,-g,2*1j*gam2]
                ])
    return LF


def LpopAsymptot(R,gam1,gam2,G1,G2):
    """Return asymptotic Liouvilian for populations"""
    L=np.matrix([[0,2*1j*gam1,0,0,2*1j*gam2],
                 [0,-2*1j*(gam1+G2),0,0,2*1j*G1],
                 [0,0,R-1j*(gam1+gam2+G1+G2),0,0],
                 [0,0,0,-R-1j*(gam1+gam2+G1+G2),0],
                 [0,2*1j*G2,0,0,-2*1j*(gam2+G1)]
                ])

    return L

def FreqAsymptot(R,gam1,gam2,G1,G2):
    """Return asymptotic frequencies for populations"""
    ww=np.array([0,-2*1j*(gam1+G1+G2),R-1j*(gam1+gam2+G1+G2),-R-1j*(gam1+gam2+G1+G2),-2*1j*gam2])
    return ww

def LFpol(g, om1, om2,gam1,gam2):
    """Return Forster Liouvilian for polarisation"""
    LF=np.matrix([[1j*gam1+om1,g],
                  [g,1j*gam2+om2]
                ])
    return LF

def LFpol_qdqdcav(g1, g2,gd, w_qd1, w_qd2, w_c):  #using python package to diagonalise matrix equivalent to analyitcally solving - although has some 1e-19 terms instead of 0
    """Diagonalise the QD-QD Liouvillian"""
    Hxx=np.matrix([[w_qd1, gd, g1],[gd, w_qd2, g2],[g1, g2, w_c]]) 
    return Hxx

   
def DiagM(g, w_qd1, w_qd2):
    """Diagonalise the QD-QD Liouvillian"""
    w1=(w_qd1+w_qd2)/2 - np.sqrt(g**2 +(0.5*(w_qd1-w_qd2))**2)
    w2=(w_qd1+w_qd2)/2 + np.sqrt(g**2 +(0.5*(w_qd1-w_qd2))**2)
    Delta_xc=np.sqrt((0.5*(w_qd1-w_qd2))**2+g**2)-0.5*(w_qd1-w_qd2)
    alpha=Delta_xc/np.sqrt(Delta_xc**2 + g**2)
    beta=g/np.sqrt(Delta_xc**2 + g**2)
    
    ws=np.array([w1,w2])
    Y=np.matrix([[alpha, beta],
                    [-beta, alpha]])
    Yt=np.transpose(Y)
    return ws, Y, Yt

def DiagM_qdcav(g, wx, gx, gc, det, omp):
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
    
def Sanalyt(T,j0,j0_1,w0,r0,Vs):
    SS=S_inin(T,j0,w0)-S_inim(T,j0_1,w0,r0,Vs)
    return SS

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




'''K11 works well in terms of these Faddeeva functions'''
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


#by hand integration re-expressed in terms of Faddeeva functions
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




       
       
def K12_smartie_integrandterm1(w, t,j0,l,lp,Vs,T,r0):
    N=((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)
    
    eta=     special.wofz((-w*(l**2 - lp**2) + 1j*r0*Vs )/(np.sqrt(2)*Vs * np.sqrt(l**2 - lp**2)))*np.exp(-(lp**2 *w**2)/(2*Vs**2))*np.exp(-1j*r0*w/Vs)  - 1*np.exp(-(l**2 *w**2)/(2*Vs**2)) *np.exp(+(r0**2)/(2*l**2 - 2*lp**2)) +  special.wofz((-w*(l**2 - lp**2) - 1j*r0*Vs )/(np.sqrt(2)*Vs * np.sqrt(l**2 - lp**2)))*np.exp(-(lp**2 *w**2)/(2*Vs**2))*np.exp(+1j*r0*w/Vs)  - 1*np.exp(-(l**2 *w**2)/(2*Vs**2)) *np.exp(+(r0**2)/(2*l**2 - 2*lp**2)) 
    return ((1/(np.exp(w/T)-1)) *np.exp(complex(0,w*t)) + ((1/(np.exp(w/T)-1))+1)*np.exp(complex(0,-w*t))) *((1j*j0*Vs*np.sqrt(np.pi) ) /(2**(3/2)*np.sqrt(l**2 - lp**2))) * eta


def K12_smartieterm1(t,j0,l,lp,Vs,T,r0):
    '''Computing the rapidly decaying part of the cumulant'''
    def re_fun(w, t, j0, l,lp,Vs, T,r0):
        return np.real(K12_smartie_integrandterm1(w, t,j0,l,lp,Vs,T,r0))
    def im_fun(w, t, j0, l,lp,Vs, T,r0):
        return np.imag(K12_smartie_integrandterm1(w, t,j0,l,lp,Vs,T,r0))
    re_int = integrate.quad(re_fun, 0,np.inf, args=(t, j0, l,lp,Vs, T,r0), limit=1000, epsrel = 1e-13, epsabs = 1e-13)
    im_int = integrate.quad(im_fun, 0, np.inf, args=(t, j0, l,lp,Vs, T,r0), limit=1000, epsrel = 1e-13, epsabs = 1e-13)
    return (re_int[0] + 1j*im_int[0])#, re_int[1] + 1j* im_int[1]




def K12_smartie_integrandtermS(w, t,j0,l,lp,Vs,T,r0):
    N=((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)
    
    eta=     special.wofz((-w*(l**2 - lp**2) + 1j*r0*Vs )/(np.sqrt(2)*Vs * np.sqrt(l**2 - lp**2)))*np.exp(-(lp**2 *w**2)/(2*Vs**2))*np.exp(-1j*r0*w/Vs)  - 1*np.exp(-(l**2 *w**2)/(2*Vs**2)) *np.exp(+(r0**2)/(2*l**2 - 2*lp**2)) +  special.wofz((-w*(l**2 - lp**2) - 1j*r0*Vs )/(np.sqrt(2)*Vs * np.sqrt(l**2 - lp**2)))*np.exp(-(lp**2 *w**2)/(2*Vs**2))*np.exp(+1j*r0*w/Vs)  - 1*np.exp(-(l**2 *w**2)/(2*Vs**2)) *np.exp(+(r0**2)/(2*l**2 - 2*lp**2)) 
    return  (1j*np.pi*j0*Vs /(2**(3/2)*np.sqrt(l**2-lp**2)))*(2*(1/(np.exp(w/T)-1))+1) *  eta


def K12_smartietermS(t,j0,l,lp,Vs,T,r0):
    '''Computing the rapidly decaying part of the cumulant'''
    def re_fun(w, t, j0, l,lp,Vs, T,r0):
        return np.real(K12_smartie_integrandtermS(w, t,j0,l,lp,Vs,T,r0))
    def im_fun(w, t, j0, l,lp,Vs, T,r0):
        return np.imag(K12_smartie_integrandtermS(w, t,j0,l,lp,Vs,T,r0))
    re_int = integrate.quad(re_fun, 0,np.inf, args=(t, j0, l,lp,Vs, T,r0), limit=1000, epsrel = 1e-12, epsabs = 0)
    im_int = integrate.quad(im_fun, 0, np.inf, args=(t, j0, l,lp,Vs, T,r0), limit=1000, epsrel = 1e-12, epsabs = 0)
    return (re_int[0] + 1j*im_int[0])#, re_int[1] + 1j* im_int[1]




def K12_smartie_integrandtermshift(w, t,j0,l,lp,Vs,T,r0):
    N=((1/(np.exp(w/T)-1))*(np.exp(1j*w*t)-1) + ((1/(np.exp(w/T)-1)) + 1)*(np.exp(-1j*w*t)-1) + 1j*w*t)
    
    eta=     special.wofz((-w*(l**2 - lp**2) + 1j*r0*Vs )/(np.sqrt(2)*Vs * np.sqrt(l**2 - lp**2)))*np.exp(-(lp**2 *w**2)/(2*Vs**2))*np.exp(-1j*r0*w/Vs)  - 1*np.exp(-(l**2 *w**2)/(2*Vs**2)) *np.exp(+(r0**2)/(2*l**2 - 2*lp**2)) +  special.wofz((-w*(l**2 - lp**2) - 1j*r0*Vs )/(np.sqrt(2)*Vs * np.sqrt(l**2 - lp**2)))*np.exp(-(lp**2 *w**2)/(2*Vs**2))*np.exp(+1j*r0*w/Vs)  - 1*np.exp(-(l**2 *w**2)/(2*Vs**2)) *np.exp(+(r0**2)/(2*l**2 - 2*lp**2)) 
    return  1j*t*w* ((1j*j0*Vs*np.sqrt(np.pi) ) /(2**(3/2)*np.sqrt(l**2 - lp**2))) * eta


def K12_smartietermshift(t,j0,l,lp,Vs,T,r0):
    '''Computing the rapidly decaying part of the cumulant'''
    def re_fun(w, t, j0, l,lp,Vs, T,r0):
        return np.real(K12_smartie_integrandtermshift(w, t,j0,l,lp,Vs,T,r0))
    def im_fun(w, t, j0, l,lp,Vs, T,r0):
        return np.imag(K12_smartie_integrandtermshift(w, t,j0,l,lp,Vs,T,r0))
    re_int = integrate.quad(re_fun, 0,np.inf, args=(t, j0, l,lp,Vs, T,r0), limit=1000, epsrel = 1e-13, epsabs = 1e-13)
    im_int = integrate.quad(im_fun, 0, np.inf, args=(t, j0, l,lp,Vs, T,r0), limit=1000, epsrel = 1e-13, epsabs = 1e-13)
    return (re_int[0] + 1j*im_int[0])#, re_int[1] + 1j* im_int[1]






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


def Jw(w,j0,w0):
    '''Spectral density'''
    integrand= j0*np.exp(-(w/(w0))**2)*w**3
    return integrand

def Jw2(w,j0,w0,r0,Vs):
    '''Spectral density'''
    integrand= j0*np.exp(-(w/(w0))**2)*w**3*np.sin(w*r0/Vs)*Vs/(w*r0)
    return integrand
#Nazir comparison:

def BNintegrand(w,j0,w0,r0,Vs,T):
    B=(Jw(w,j0,w0)/w**2)*(1-np.sinc(w*r0/Vs))*(1/np.tanh(w/(2*T)))
    return B

def BN(j0,w0,r0,Vs,T):
    def re_fun(w,j0,w0,r0,Vs,T):
        return np.real(BNintegrand(w,j0,w0,r0,Vs,T))
    def im_fun(w,j0,w0,r0,Vs,T):
        return np.imag(BNintegrand(w,j0,w0,r0,Vs,T))
    re_int = integrate.quad(re_fun, 0, np.inf, args=(j0,w0,r0,Vs,T), limit=1000)
    im_int = integrate.quad(im_fun, 0, np.inf, args=(j0,w0,r0,Vs,T), limit=1000)
    return np.exp(-(re_int[0] + 1j*im_int[0]))


def phi_bar_int(w,tau,j0,w0,T,r0,Vs):
    integrand= 2*(Jw(w,j0,w0)/w**2)*(1-np.sinc(w*r0/Vs))*np.cos(w*tau)/np.sinh(w/(2*T))
    return integrand

def phi_bar(tau,j0,w0,T,r0,Vs):
    def re_fun(w, tau,j0, w0, T,r0,Vs):
        return np.real(phi_bar_int(w,tau,j0,w0,T,r0,Vs))
    def im_fun(w, tau,j0, w0, T,r0,Vs):
        return np.imag(phi_bar_int(w,tau,j0,w0,T,r0,Vs))
    re_int = integrate.quad(re_fun, 0, np.inf, args=(tau,j0,w0,T,r0,Vs), limit=1000)
    im_int = integrate.quad(im_fun, 0, np.inf, args=(tau,j0,w0,T,r0,Vs), limit=1000)
    return (re_int[0] + 1j*im_int[0])

def gxNint(w,tau,j0,w0,T,r0,Vs):
    integrand=np.exp(1j*w*tau)*(BN(j0,w0,r0,Vs,T)**2/2)*(np.exp(phi_bar(tau,j0,w0,T,r0,Vs))+np.exp(-phi_bar(tau,j0,w0,T,r0,Vs))-2)
    return integrand

def gxN(w,j0,w0,T,r0,Vs):
    def re_fun(w, tau,j0, w0, T,r0,Vs):
        return np.real(gxNint(w,tau,j0,w0,T,r0,Vs))
    def im_fun(w, tau,j0, w0, T,r0,Vs):
        return np.imag(gxNint(w,tau,j0,w0,T,r0,Vs))
    re_int = integrate.quad(re_fun, -np.inf, np.inf, args=(w,j0,w0,T,r0,Vs), limit=1000)
    im_int = integrate.quad(im_fun, -np.inf, np.inf, args=(w,j0,w0,T,r0,Vs), limit=1000)
    return np.exp(w/(2*T))*(re_int[0] + 1j*im_int[0])

def gyNint(w,tau,j0,w0,T,r0,Vs):
    integrand=np.exp(1j*w*tau)*(BN(j0,w0,r0,Vs,T)**2/2)*(np.exp(phi_bar(tau,j0,w0,T,r0,Vs))-np.exp(-phi_bar(tau,j0,w0,T,r0,Vs)))
    return integrand

def gyN(w,j0,w0,T,r0,Vs):
    def re_fun(w, tau,j0, w0, T,r0,Vs):
        return np.real(gyNint(w,tau,j0,w0,T,r0,Vs))
    def im_fun(w, tau,j0, w0, T,r0,Vs):
        return np.imag(gyNint(w,tau,j0,w0,T,r0,Vs))
    re_int = integrate.quad(re_fun, -np.inf, np.inf, args=(w,j0,w0,T,r0,Vs), limit=1000)
    im_int = integrate.quad(im_fun, -np.inf, np.inf, args=(w,j0,w0,T,r0,Vs), limit=1000)
    return np.exp(w/(2*T))*(re_int[0] + 1j*im_int[0])

def SyNint(w,tau,j0,w0,T,r0,Vs):
    #check this:
    integrand=np.exp(1j*w*tau)*(BN(j0,w0,r0,Vs,T)**2/2)*(np.exp(phi_bar(tau,j0,w0,T,r0,Vs))-np.exp(-phi_bar(tau,j0,w0,T,r0,Vs)))
    return integrand

def SyN(w,j0,w0,T,r0,Vs):
    def re_fun(w, tau,j0, w0, T,r0,Vs):
        return np.real(SyNint(w,tau,j0,w0,T,r0,Vs))
    def im_fun(w, tau,j0, w0, T,r0,Vs):
        return np.imag(SyNint(w,tau,j0,w0,T,r0,Vs))
    re_int = integrate.quad(re_fun, 0, np.inf, args=(w,j0,w0,T,r0,Vs), limit=1000)
    im_int = integrate.quad(im_fun, 0, np.inf, args=(w,j0,w0,T,r0,Vs), limit=1000)
    return (re_int[0] + 1j*im_int[0]).imag

#Forster coupling:

import math
def erfcc_exp(x):
    #t=1.0/(1.0+0.5*x);
    #ans=t*np.exp(-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))));
    ans=math.erfc(x)*np.exp(x**2)
    return ans;

def erfc(x):
    f= lambda t: np.exp(t**2)
    ans = integrate.quad(f, x, np.inf)
    return ans*2/np.sqrt(np.pi)

def forster_potential(z,L_e,L_h,Epsilon,d_cv):
    '''Forster potential from Egor's code'''
    #HP  =  0.658212197   # //[meV*ps] hbar
    #KB  =  0.0861712     # //[meV/K] Boltzmann
    RY   =   13605.11835 # //[meV] Rydberg
    A_B   =  0.052917706 # // [nm] Bohr radius
    PI = 3.141592653589793 #pi

    E_e2=2.0*RY*A_B
    #print(E_e2)

    L_h=L_e

    b=4/(1/L_e**2+1/L_h**2)
    b=np.sqrt(b)
    rho=2.0*L_e*L_h/(L_e**2+L_h**2)

    def v_f(x):
        return(-x+(x*x+0.5)*np.sqrt(PI)*erfcc_exp(x) )
    f=E_e2*2*d_cv**2*rho**2*v_f(z/b)/(Epsilon*b**3) 
    return f

def forster_integrand_1(z,zp,l,eps,dcv,d):
    '''Forster interaction microscopic calc for distance dependance'''
    le=l; lh=l
    #V0z=1/(np.pi*le*lh)/(np.pi*le*lh)
    V0z=(np.sqrt(1/(np.sqrt(np.pi)*l)))**4 #corrected
    #print(V0z2/V0z)#=pi l^2
    #print(np.pi*l**2)
    #g= 217.1531272304639 mu eV
    #g= 2462.7660758574984 mu eV #new
    exponent=-0.5*(z/le)*(z/le)-0.5*(z/lh)*(z/lh)-0.5*((zp-d)/le)*((zp-d)/le)-0.5*((zp-d)/lh)*((zp-d)/lh)
    psi=np.exp(exponent)
    integrand= V0z*psi*forster_potential(zp-z,l,l,eps,dcv)
    return integrand


def forster(l,eps,dcv,d,lim1,lim2, lim11, lim22):
    '''Forster interaction microscopic calc for distance dependance'''
    f= lambda z,zp: forster_integrand_1(z,zp,l,eps,dcv,d)
    re_int = integrate.dblquad(f, lim1, lim2, lim11, lim22)
    return re_int[0]



def FGR_qdcav_spherical(g,j0_FGR,l,Vs,T):
    hbar= 0.6582
    N= (1/(np.exp(2*g/T)-1))  
    gamma_ph= g**3 * j0_FGR * np.exp(-(2*g**2 * l**2)/(Vs**2))
    Gamma_1= N * gamma_ph
    Gamma_2 =(N+1) * gamma_ph
    return 1e3*hbar*Gamma_1, 1e3*hbar*Gamma_2
    
def FGR_qdcav_spherical_det(g,detuning,j0_FGR,l,Vs,T):
    hbar= 0.6582
    R= np.sqrt((detuning**2 + 4*g**2)) 
    # print(hbar*1e3*R)
    N=1/(np.exp(R/T)-1)
    Dplus=(1/np.sqrt(2)) * np.sqrt(1 + (detuning/R))
    Dneg=(1/np.sqrt(2)) * np.sqrt(1 - (detuning/R))
    Gamma_0= Dplus**2 * Dneg**2 * R**3 *j0_FGR/2 * np.exp(-(l**2 * R**2) / (2*Vs**2))
    Gamma_1= N*Gamma_0
    Gamma_2= (N+1)*Gamma_0
    return 1e3*hbar*Gamma_1, 1e3*hbar*Gamma_2

def analytics_bareg(j0,j0_FGR,l,Vs,w0,T,r0, detuning,g,Om1,Om2,gamma1,gamma2):  #FGR QD-QD isotropic dots
    hbar= 0.6582
    R= np.sqrt((detuning**2 + 4*g**2)) 
    # print(hbar*1e3*R)
    N=1/(np.exp(R/T)-1)
    Dplus=(1/np.sqrt(2)) * np.sqrt(1 + (detuning/R))
    Dneg=(1/np.sqrt(2)) * np.sqrt(1 - (detuning/R))
    Gamma_0= Dplus**2 * Dneg**2 * R**3 *j0_FGR * np.exp(-(l**2 * R**2) / (2*Vs**2))
    
    Gamma_ph=Gamma_0*(1- (Vs*np.sin(R*r0/Vs)/(R*r0)))
    Gamma_plus=(N+1)*Gamma_ph
    Gamma_neg=N*Gamma_ph
    # print(1e3*hbar*Gamma_plus)
    return 1e3*hbar*Gamma_plus,1e3*hbar*Gamma_neg

def analytics_modified(j0,j0_FGR,l,Vs,w0,T,r0, detuning,g,Om1,Om2,gamma1,gamma2):  #FGR QD-QD anisotropic dots modified through polaron shift + huang-rhys
    Sinin= S_inin(T,j0,w0)      
    j0_1=j0*Vs/r0 
    Sinim= S_inim(T, j0_1, w0, r0, Vs)
    DeltaS=np.exp(-(Sinin-Sinim))
  
    
    hbar= 0.6582
    R= np.sqrt((detuning**2 + 4*(g*DeltaS)**2)) 
    # print(hbar*1e3*R)
    N=1/(np.exp(R/T)-1)
    Dplus=(1/np.sqrt(2)) * np.sqrt(1 + (detuning/R))
    Dneg=(1/np.sqrt(2)) * np.sqrt(1 - (detuning/R))
    Gamma_0= Dplus**2 * Dneg**2 * R**3 *j0_FGR * np.exp(-(l**2 * R**2) / (2*Vs**2))
    # Gamma_0= Dplus**2 * Dneg**2 * R**3 *j0_FGR * np.exp(-(l**2 * R**2) / (Vs**2))

    Gamma_ph=Gamma_0*(1- (Vs*np.sin(R*r0/Vs)/(R*r0)))
    Gamma_plus=(N+1)*Gamma_ph
    Gamma_neg=N*Gamma_ph
    # print(1e3*hbar*Gamma_plus)
    return 1e3*hbar*Gamma_plus,1e3*hbar*Gamma_neg


def QDQD_analytics_smartie(gd,detuning,l,lp,Vs,r0,T,DvDc):  #FGR QD-QD anisotropic dots
     hbar= 0.6582
     R=np.sqrt((detuning)**2 + 4*(gd)**2)
     Dp=np.sqrt(1/2) * np.sqrt(1+(detuning/ R))
     Dm=np.sqrt(1/2) * np.sqrt(1-(detuning / R))
     alph=R* 1j*np.sqrt(l**2-lp**2)/(np.sqrt(2)*Vs)
     bet=R*r0/Vs
     BoseDist= (1/(np.exp(R/T)-1))  
    
     eta_1= special.wofz(-1j*alph)*np.exp(-(lp**2 *R**2)/(2*Vs**2)) - np.exp(-(l**2 *R**2)/(2*Vs**2))
     eta_2= -special.wofz(1j*alph - bet/(2*alph)) * np.exp(-lp**2 *R**2 / (2*Vs**2)) * np.exp(-1j*bet) + special.wofz(-1j*alph - bet/(2*alph))  * np.exp(-lp**2 *R**2 / (2*Vs**2)) * np.exp(1j*bet) 
     
     j0=(DvDc**2 * (1e3)**2 * (3e8)**2 * (1e9)**2 * (1e-12)**2) /  (4*np.sqrt(np.pi) * 5.65 * (1e-3) * (3e8)**2 * (6.24e21) *(1e-7)**3 * 4.6**5)
     j0=j0/hbar  
     gam=(R**3 * Dp**2 * Dm**2 * (j0)/alph ) * (eta_1 - eta_2/2)
     
     gam1=(BoseDist+1)*gam
     gam2=(BoseDist)*gam
     return hbar*1e3*gam1,hbar*1e3*gam2



def FGR_spherical(j0_FGR,l,Vs,T,g1,gd,w_qd1,w_c,R0s): #QD-QD-CAV spherical FGR
    # hbar= 0.6582
    # H0= smp.Matrix([[w_qd1, gd, g1 ], [gd , w_qd1, g1],[g1, g1, w_c]])
    # P, D = H0.diagonalize()   #P*D*P**-1
    
    
    # # var('V1 V2')
    # # Vmtrx=Matrix([[V1,0,0],[0,V2,0],[0,0,0]])
    # # Vtransform=P**-1 * Vmtrx * P
    # # print(Vtransform)
    
    # #extracting the eigenvectors that diagonalise H0 as a list. 
    # sym_eignvects = []
    # for tup in H0.eigenvects():
    #     for v in tup[2]:
    #         sym_eignvects.append(list(v))
    
    # coefficients=np.sort(np.array([abs(sym_eignvects[0][0] *sym_eignvects[1][0]) , abs(sym_eignvects[0][0] * sym_eignvects[2][0]), abs(sym_eignvects[1][0] * sym_eignvects[2][0])])) #the corresponding components of eigenvectors - like alpha*beta but 3x3 and numerically diagonalised
    
    # constsmall=float(np.array([coefficients[2]**2])[0] ) #corresponding to smallest transition   #like (alpha*beta)**2 for QDQD zero detuning case. and (1/2root2)**2 in zero det QDQDCAV system
    # constbig=float(np.array([coefficients[1]**2])[0])  #corresponding to 2nd largest transition  - 1/(4) **2 in zero det
    # constbiggest=float(np.array([coefficients[0]**2])[0]) #corresponding to largest transition - 1/(4) **2 in zero det case
    
    
    # eigenvals = H0.eigenvals()
    # numerical_eigenvals = [val.evalf() for val in eigenvals.keys()]
    # numerical_eigenvals=np.array([numerical_eigenvals], dtype=np.float64)
    # energylvls=np.sort(numerical_eigenvals)[0]
    # wsmall=energylvls[1]-energylvls[0]
    # wbig=energylvls[2]-energylvls[1]
    # wbiggest=energylvls[2]-energylvls[0]
    # # wsmall=float(np.array([abs(list(H0.eigenvals().keys())[1])])[0])  #smallest transition between states 1 and 3
    # # wbig=float(np.array([abs(list(H0.eigenvals().keys())[2])])[0])   #second largest transition in terms of energy, from middle state to 2nd state (equidistant for zero det)
    # # wbiggest=float(abs(wbig + wsmall))  #energy difference from lowest energy state (3) to highest (2) ( 1 is in middle)
    hbar= 0.6582
    if gd==g1:
        gd=g1-1e-7
    H0= smp.Matrix([[w_qd1, gd, g1 ], [gd , w_qd1, g1],[g1, g1, w_c]])
    P, D = H0.diagonalize()   #P*D*P**-1
    P_inv=P**-1
    
    rows_P = [P.row(i) for i in range(P.rows)]
    rows_P_inv = [P_inv.row(i) for i in range(P_inv.rows)]
    
    # V1, V2 = symbols('V1 V2')
    # # Define matrix V
    # V = Matrix([[V1, 0, 0],
    #         [0, V2, 0],
    #         [0, 0, 0]])
    
    # result1 = simplify(P1.inv() * V * P1)
    # # Print the results for the original matrix M1
    # print("Transformation for original matrix M1:")
    # print(result1)
    
    
    
    # smp.pprint(D)
    # smp.pprint(P**-1)
    # smp.pprint(P)
    # smp.pprint(P*P**-1)
    # smp.pprint(P*D*P**-1)
    
    #extracting the eigenvectors that diagonalise H0 as a list. 
    # sym_eignvects = []
    # for tup in H0.eigenvects(): 
    #     for v in tup[2]:
    #         sym_eignvects.append(list(v))
    
    # coefficients=np.sort(np.array([abs(sym_eignvects[0][0] *sym_eignvects[1][0]) , abs(sym_eignvects[0][0] * sym_eignvects[2][0]), abs(sym_eignvects[1][0] * sym_eignvects[2][0])])) #the corresponding components of eigenvectors - like alpha*beta but 3x3 and numerically diagonalised
    coefficients1=np.array([rows_P_inv[0][0]*rows_P[0][1], rows_P_inv[0][0]*rows_P[0][2], rows_P_inv[1][0]*rows_P[0][2]]) # if you expand out S^T V S, these are the coefficients on the relevant elements (offdiagonal)
    # smp.pprint(coefficients)
    # constsmall=float(np.array([coefficients[1]**2])[0] ) #corresponding to smallest transition   #like (alpha*beta)**2 for QDQD zero detuning case. and (1/2root2)**2 in zero det QDQDCAV system
    # constbig=float(np.array([coefficients[2]**2])[0])  #corresponding to 2nd largest transition  - 1/(4) **2 in zero det
    # constbiggest=float(np.array([coefficients[0]**2])[0]) #corresponding to largest transition - 1/(4) **2 in zero det case
    
    eigenvals_dict = H0.eigenvals()
    all_eigenvals = []
    for eigenval, multiplicity in eigenvals_dict.items():
        all_eigenvals.extend([eigenval] * multiplicity)
    
    # Convert all eigenvalues to numerical values
    numerical_eigenvals = [val.evalf() for val in all_eigenvals]
    numerical_eigenvals=np.array([numerical_eigenvals], dtype=np.float64)
    # energylvls=np.sort(numerical_eigenvals)[0]
    # wsmall=energylvls[1]-energylvls[0]
    # wbig=energylvls[2]-energylvls[1]
    # wbiggest=energylvls[2]-energylvls[0] 
    
    w_diag12=np.abs(numerical_eigenvals[0][1]-numerical_eigenvals[0][0])
    w_diag13=np.abs(numerical_eigenvals[0][2]-numerical_eigenvals[0][0])
    w_diag23=np.abs(numerical_eigenvals[0][2]-numerical_eigenvals[0][1])

     
    def Gamma_Ph(const,R0s,w,PorM):
        return const*j0_FGR*w**3 * np.exp(-(l**2 * w**2)/(2*Vs**2)) * (1 +PorM* (Vs* np.sin(w*R0s/Vs))/(w*R0s) )
    
    def Gammas(const,R0s,w,PorM):
        N=1/(np.exp(w/T)-1)
        return hbar* N * Gamma_Ph(const,R0s,w,PorM),hbar* (N+1) * Gamma_Ph(const,R0s,w,PorM)
    # def GammaDown(const,R0s,w,PorM):
    #     N=1/(np.exp(w/T)-1)
    #     return hbar* (N+1) * Gamma_Ph(const,R0s,w,PorM)
         
    # R0s=np.linspace(0.05,1000,1000)
    # print(1e3*hbar*Gammas(coefficients1[0]**2,R0s,w_diag12,-1)[1])
    # dephasing1=GammaUp(constbig,R0s,wbig,-1) + GammaDown(constsmall,R0s,wsmall,-1)
    # dephasing2=GammaDown(constbig,R0s,wbig,-1) + GammaDown(constbiggest,R0s,wbiggest,+1)
    # dephasing3=GammaUp(constsmall,R0s,wsmall,-1) + GammaUp(constbiggest,R0s,wbiggest,+1)
    dephasing1=Gammas(coefficients1[0]**2,R0s,w_diag12,-1)[0] + Gammas(coefficients1[1]**2,R0s,w_diag13,-1)[1]  # red
    dephasing2=Gammas(coefficients1[0]**2,R0s,w_diag12,-1)[1] + Gammas(coefficients1[2]**2,R0s,w_diag23,+1)[1] #blue
    dephasing3=Gammas(coefficients1[1]**2,R0s,w_diag13,-1)[0] + Gammas(coefficients1[2]**2,R0s,w_diag23,+1)[0] #  green 
    
    return 1e3*dephasing1, 1e3*dephasing2, 1e3*dephasing3





def FGR_smartie(j0_FGR,l,lp,Vs,T,g1,gd,w_qd1,w_c,R0s):  #FGR QD-QD-CAV anisotropic
    hbar= 0.6582
    if gd==g1:
        gd=g1-1e-7
    H0= smp.Matrix([[w_qd1, gd, g1 ], [gd , w_qd1, g1],[g1, g1, w_c]])
    P, D = H0.diagonalize()   #P*D*P**-1
    P_inv=P**-1
    
    rows_P = [P.row(i) for i in range(P.rows)]
    rows_P_inv = [P_inv.row(i) for i in range(P_inv.rows)]
    
    # V1, V2 = symbols('V1 V2')
    # # Define matrix V
    # V = Matrix([[V1, 0, 0],
    #         [0, V2, 0],
    #         [0, 0, 0]])
    
    # result1 = simplify(P1.inv() * V * P1)
    # # Print the results for the original matrix M1
    # print("Transformation for original matrix M1:")
    # print(result1)
    
    
    
    # smp.pprint(D)
    # smp.pprint(P**-1)
    # smp.pprint(P)
    # smp.pprint(P*P**-1)
    # smp.pprint(P*D*P**-1)
    
    #extracting the eigenvectors that diagonalise H0 as a list. 
    # sym_eignvects = []
    # for tup in H0.eigenvects(): 
    #     for v in tup[2]:
    #         sym_eignvects.append(list(v))
    
    # coefficients=np.sort(np.array([abs(sym_eignvects[0][0] *sym_eignvects[1][0]) , abs(sym_eignvects[0][0] * sym_eignvects[2][0]), abs(sym_eignvects[1][0] * sym_eignvects[2][0])])) #the corresponding components of eigenvectors - like alpha*beta but 3x3 and numerically diagonalised
    coefficients1=np.array([rows_P_inv[0][0]*rows_P[0][1], rows_P_inv[0][0]*rows_P[0][2], rows_P_inv[1][0]*rows_P[0][2]]) # if you expand out S^T V S, these are the coefficients on the relevant elements (offdiagonal)
    # smp.pprint(coefficients)
    # constsmall=float(np.array([coefficients[1]**2])[0] ) #corresponding to smallest transition   #like (alpha*beta)**2 for QDQD zero detuning case. and (1/2root2)**2 in zero det QDQDCAV system
    # constbig=float(np.array([coefficients[2]**2])[0])  #corresponding to 2nd largest transition  - 1/(4) **2 in zero det
    # constbiggest=float(np.array([coefficients[0]**2])[0]) #corresponding to largest transition - 1/(4) **2 in zero det case
    
    eigenvals_dict = H0.eigenvals()
    all_eigenvals = []
    for eigenval, multiplicity in eigenvals_dict.items():
        all_eigenvals.extend([eigenval] * multiplicity)
    
    # Convert all eigenvalues to numerical values
    numerical_eigenvals = [val.evalf() for val in all_eigenvals]
    numerical_eigenvals=np.array([numerical_eigenvals], dtype=np.float64)
    # energylvls=np.sort(numerical_eigenvals)[0]
    # wsmall=energylvls[1]-energylvls[0]
    # wbig=energylvls[2]-energylvls[1]
    # wbiggest=energylvls[2]-energylvls[0] 
    
    w_diag12=np.abs(numerical_eigenvals[0][1]-numerical_eigenvals[0][0])
    w_diag13=np.abs(numerical_eigenvals[0][2]-numerical_eigenvals[0][0])
    w_diag23=np.abs(numerical_eigenvals[0][2]-numerical_eigenvals[0][1])

    
    
    # wsmall=float(np.array([abs(list(H0.eigenvals().keys())[1])])[0])  #smallest transition between states 1 and 3
    # wbig=float(np.array([abs(list(H0.eigenvals().keys())[2])])[0])   #second largest transition in terms of energy, from middle state to 2nd state (equidistant for zero det)
    # wbiggest=float(abs(wbig + wsmall))  #energy difference from lowest energy state (3) to highest (2) ( 1 is in middle)
      
    def Gamma_Ph2(theta,R0s,w,PorM):
        integrand= np.sin(theta)* j0_FGR/2 *w**3 * np.exp(-(l**2 * w**2 * np.sin(theta)**2)/(2*Vs**2))* np.exp(-(lp**2 * w**2 * np.cos(theta)**2)/(2*Vs**2))  * (1 +PorM* (np.cos(w*R0s*np.cos(theta)/Vs)) )
        return integrand


 
    def Gammas(const,R0s,w,PorM): #up transition is [0] and down transition is [1], constants and delta w  determined by specific transitions. see diagram
        if w !=0:
            N=1/(np.exp(w/T)-1)
        else:
            N=0
        dephasing=[]
        for r0 in R0s:
            dephasing.append(quad(Gamma_Ph2, 0,np.pi,args=(r0,w,PorM))[0])
        dephasing=np.array([dephasing],dtype=np.float64)[0]
        return hbar* N * const*dephasing, hbar* (N+1) * const*dephasing
    
    

    # dephasing1=Gammas(constbig,R0s,wbig,-1)[0] + Gammas(constsmall,R0s,wsmall,-1)[1]  # red
    # dephasing2=Gammas(constbig,R0s,wbig,-1)[1] + Gammas(constbiggest,R0s,wbiggest,+1)[1] #blue
    # dephasing3=Gammas(constsmall,R0s,wsmall,-1)[0] + Gammas(constbiggest,R0s,wbiggest,+1)[0] #  green 
    
    dephasing1=Gammas(coefficients1[0]**2,R0s,w_diag12,-1)[0] + Gammas(coefficients1[1]**2,R0s,w_diag13,-1)[1]  # red
    dephasing2=Gammas(coefficients1[0]**2,R0s,w_diag12,-1)[1] + Gammas(coefficients1[2]**2,R0s,w_diag23,+1)[1] #blue
    dephasing3=Gammas(coefficients1[1]**2,R0s,w_diag13,-1)[0] + Gammas(coefficients1[2]**2,R0s,w_diag23,+1)[0] #  green 

    if gd > g1:
        dephasing1=Gammas(coefficients1[0]**2,R0s,w_diag12,-1)[0] + Gammas(coefficients1[1]**2,R0s,w_diag13,-1)[0]  # red
        dephasing2=Gammas(coefficients1[0]**2,R0s,w_diag12,-1)[1] + Gammas(coefficients1[2]**2,R0s,w_diag23,+1)[1] #blue
        dephasing3=Gammas(coefficients1[1]**2,R0s,w_diag13,-1)[1] + Gammas(coefficients1[2]**2,R0s,w_diag23,+1)[0] #  green 

    return 1e3*dephasing1, 1e3*dephasing2, 1e3*dephasing3
