''' Parameters used in QD-QD system with shared phonon environment '''

import numpy as np
import os
import sys
import getopt
from Functions import PolaronShift_inim, PolaronShift, S_inin, S_inim, forster
path=os.getcwd()

''' Set parameters here, all parameters in ps units'''
#Choose system
cavity=1
no_of_QDs=1
#choose number of neighbours
L=12
#Calculate linear polarization 'LP' or Population dynamics 'NQD'
correlator='LP'  #LP'#'NQD'
#choose whether to use a shared or independent bath (1 for shared, 0 for independent baths):
sharebath=1
#propagation end time
tf=100

dotshape='spherical'  #'spherical' or 'smartie'
#choose excitation and measuring channels {'1','2','C'}. Dot 1, 2 or cavity mode.
ec='1'
mc='1'
#temperature, K
T=50
#exciton-exciton coupling strength in micro eV (when r0=sep0):
g = 0
# exciton-cavity coupling strength
gc= 2000
#detuning in micro eV

detuning=0 # setting it to omp later
# exciton 1-cavity and exciton-2 cavity coupling strengths
g1=gc
g2=gc
#distance between QDs in nm
r0=0.01
#Dc-Dv in eV (up to ~14 is sensible)
DvDc=7.5
l=3.3# exciton confinement lengths
lp=3.3

threshold_factor=1e-10
threshold_str = str(threshold_factor)
### The following two parameters are used to tune the convergence. 
### This must be done by trial and error for a specific parameter regime.
#set multiple of tauib in definition of t0 (factortau=2 in general)
factortau=1.2
factort0=1


if cavity==1 and no_of_QDs==2:
    #exciton-cavity coupling strength in micro eV:
    g1 = gc
    g2 = gc
    gd=g
if cavity==1 and no_of_QDs==1:
    gd=gc
    g1=0
    g2=0
    r0=0
if correlator=='NQD':
    d=5
    gd=0
if correlator=='LP':
    d=cavity+no_of_QDs
  
#QD size in nm, l is in plane length, lp is perpendicular length. lp=l gives spherical result

if dotshape=='spherical':
    lp=l
lbar = (np.sqrt((l**2 + lp**2)/2))


dcv=0.6
incr=0.05
n=0  
#Foerster coupling
eps=12.53#*8.33227308696581#*(np.pi*l*2)
fact=5
if correlator=='NQD':
    g=forster(l,eps,dcv,r0,-2*fact,10*fact,-4*fact,8*fact)
    r0p=r0+incr
    gp=forster(l,eps,dcv,r0p,-2*fact,10*fact,-4*fact,8*fact)


#finishing time as a multiple of tauib:
pfin=300
#phenomenological decay rate of each dot in micro eV
gamma1=0
gamma2=0
gammac=0
#show graphs
shgr=1


# name of data folder to use (this folder is assumed to be in the same directory as all relevant .py files)
dat_fold='data'

#adjusting params from terminal:
opts, args = getopt.getopt(sys.argv[1:], 'e:b:l:d:g:r:t:c:s:x:i:h:paf', [
                                                    'T=',
                                                    'DvDc='                    
                                                    'L=',
                                                    'l=',
                                                    'g=',
                                                    'r0=',
                                                    't0=',
                                                    'correlator=',
                                                    'shgr=',
                                                    'detuning=',
                                                    'Gamma=',
                                                    'gamma=',
                                                    'propagate',
                                                    'analytics',
                                                    'fit'])
print(opts,args)
for opt, arg in opts:
    if opt in ('-e', '--T'):
        T = int(arg)
    elif opt in ('-b', '--DvDc'):
        DvDc = float(arg)
    elif opt in ('-l', '--L'):
        L = int(arg)
    elif opt in ('-d', '--l'):
        l = float(arg)
    elif opt in ('-g', '--g'):
        g0 = float(arg)
    elif opt in ('-r', '--r0'):
        r0 = float(arg)
    elif opt in ('-t', '--t0'):
        t0 = float(arg)
    elif opt in ('-c', '--correlator'):
        correlator = arg
    elif opt in ('-s', '--shgr'):
        shgr = int(arg)
    elif opt in ('-x', '--detuning'):
        detuning = float(arg)
    elif opt in ('-i', '--Gamma'):
        Gamma = float(arg)
    elif opt in ('-h', '--gamma'):
        gamma = float(arg)
    else:
        pass
###############################################################################


# t0_data={0.0: 0.1, 1.0: 0.1, 2.0: 0.2, 3.0: 0.4, 4.0: 0.6, 5.0: 0.9, 6.0: 1.1, 7.0: 1.3, 8.0: 1.6, 9.0: 1.75, 10.0: 2.0, 11.0: 2.2, 12.0: 2.4, 13.0: 2.6, 14.0: 2.8, 15.0: 3.0, 16.0: 3.2, 17.0: 3.4, 18.0: 3.6, 19.0: 3.8, 20.0: 4.0, 21.0: 4.2, 22.0: 4.4, 23.0: 4.6, 24.0: 4.8, 25.0: 5.0, 26.0: 5.2, 27.0: 5.4, 28.0: 5.6, 29.0: 5.8, 30.0: 6.0, 31.0: 6.2, 32.0: 6.4, 33.0: 6.6, 34.0: 6.8, 35.0: 7.0, 36.0: 7.2, 37.0: 7.4, 38.0: 7.6, 39.0: 7.8, 40.0: 8.0, 41.0: 8.2, 42.0: 8.4, 43.0: 8.6, 44.0: 8.8, 45.0: 9.0, 46.0: 9.2, 47.0: 9.4, 48.0: 9.6, 49.0: 9.8, 50.0: 10.0, 51.0: 10.2, 52.0: 10.4, 53.0: 10.6, 54.0: 10.8, 55.0: 11.0, 56.0: 11.2, 57.0: 11.4, 58.0: 11.6, 59.0: 11.8, 60.0: 12.0, 61.0: 12.2, 62.0: 12.4, 63.0: 12.6, 64.0: 12.8, 65.0: 13.0, 66.0: 13.2, 67.0: 13.4, 68.0: 13.6, 69.0: 13.8, 70.0: 14.0, 71.0: 14.2, 72.0: 14.4, 73.0: 14.6, 74.0: 14.8, 75.0: 15.0, 76.0: 15.2, 77.0: 15.4, 78.0: 15.6, 79.0: 15.8, 80.0: 16.0, 81.0: 16.2, 82.0: 16.4, 83.0: 16.6, 84.0: 16.8, 85.0: 17.0, 86.0: 17.2, 87.0: 17.4, 88.0: 17.6, 89.0: 17.8, 90.0: 18.0, 91.0: 18.2, 92.0: 18.4, 93.0: 18.6, 94.0: 18.8, 95.0: 19.0, 96.0: 19.2, 97.0: 19.4, 98.0: 19.6, 99.0: 19.8, 100.0: 20.0, 101.0: 20.2, 102.0: 20.4, 103.0: 20.6, 104.0: 20.8, 105.0: 21.0, 106.0: 21.2, 107.0: 21.4, 108.0: 21.6, 109.0: 21.8, 110.0: 22.0, 111.0: 22.2, 112.0: 22.4, 113.0: 22.6, 114.0: 22.8, 115.0: 23.0, 116.0: 23.2, 117.0: 23.4, 118.0: 23.6, 119.0: 23.8, 120.0: 24.0, 121.0: 24.2, 122.0: 24.4, 123.0: 24.6, 124.0: 24.8, 125.0: 25.0, 126.0: 25.2}
# t0=t0_data[np.round(r0)]


#adjust t0



#fundamental constants:
hbar= 0.6582# [meV ps]
kb=8.617e-2 # [meV ps]
vc=3e8 #[m s^-1]
#QD material parameters:
t00=1e-12 
Vs= 4.6e3*t00 * 1e9 #[km/s]
dens=5.65 #[g/cm^3]

w0=(np.sqrt(2)*Vs/l) #[ps]

T=T*kb/hbar #[ps]
j0=(DvDc**2  * (1e3)**2 * (vc)**2 * (1e9)**2 * (1e-12)**2) / ( (2*np.pi)**2 * dens * (1e-3) * (vc)**2 * (6.24e21) *(1e-7)**3 * Vs**5)
j0=j0/hbar #j0 in ps^2
#constant terms in K_{in,i(n+/- 1)} in [meV ps^3] units (similar to j0 except v_s^4 instead of v_s^5 and an r0 factor)
if no_of_QDs==2:
    j0_1=(DvDc**2 * (1e3)**2 * (vc)**2 * (1e9)**2 * (1e-12)**2) / ( (2*np.pi)**2 * dens * (1e-3) * (vc)**2 * (6.24e21) *(1e-7)**3 * Vs**4 *r0)
    j0_1=j0_1/hbar
j0_FGR=((DvDc**2 * (1e3)**2 * (3e8)**2 * (1e9)**2 * (1e-12)**2) /  (2*np.pi * 5.65 * (1e-3) * (3e8)**2 * (6.24e21) *(1e-7)**3 * 4.6**5) )/hbar
 

omp=PolaronShift(j0,w0)
detuning=omp*1e3*hbar #!!!
#convert all parameters ready to be used in calculations
gamma1=gamma1*1e-3/hbar
gamma2=gamma2*1e-3/hbar
gammac=gammac*1e-3/hbar
if correlator=='LP':
    g=g*1e-3 /hbar
g1=g1*1e-3 /hbar
g2=g2*1e-3 /hbar
if cavity==1:
    gd=gd*1e-3/hbar
detuning=detuning*1e-3 /hbar
det=detuning  #!!
#real parts, QD energies:
if cavity==0:
    Om1=0
    Om2=detuning
if cavity==1:
    Om1=0
    Om2=0
    OmC= detuning
#constant terms in K_{in,in} in [meV ps^3] units (same as j0 earlier)


ompinim=PolaronShift_inim(j0,w0,r0,l)

SHR=S_inin(T,j0,w0)
if no_of_QDs==2:
    SHRinim=S_inim(T,j0_1,w0,r0,Vs)
    def Sanalyt(r0):
        SS=S_inin(T,j0,w0)-S_inim(T,j0_1,w0,r0,Vs)
        return SS

w_qd1=(Om1 - 1j*gamma1)
w_qd2=(Om2 - 1j*gamma2)
if cavity==1:
    w_c = OmC -1j*gammac
w1= - np.sqrt(g**2 +(0.5*(w_qd1-w_qd2))**2) + (w_qd1+w_qd2)/2
w2= + np.sqrt(g**2 +(0.5*(w_qd1-w_qd2))**2) + (w_qd1+w_qd2)/2
Delta_xc=np.sqrt((0.5*(w_qd1-w_qd2))**2+g**2)-0.5*(w_qd1-w_qd2)
alpha=Delta_xc/np.sqrt(Delta_xc**2 + g**2)
beta=g/np.sqrt(Delta_xc**2 + g**2)

if dotshape=='spherical':
    tauib=np.sqrt(2)*np.pi*l/Vs
if dotshape=='smartie':
    tauib=np.sqrt(2)*np.pi *lbar/Vs

t0=r0/Vs
# choose correct delay time t0
if sharebath ==1:
    dt1=( t0 + factortau*tauib )/(L+1)
if sharebath ==0:
    dt1=factortau*tauib / (L+1)


if no_of_QDs==1:
    (factortau*tauib )/(L+1)
#length of short-time region where the Trotter step depends on t
offset=0*tauib


if correlator=='LP' and cavity==0:
    d=2
    # QD2 channel:
    Q2 = np.matrix([[0],
                    [1]
                    ])

    o2 = Q2.T

    # QD1 channel:
    Q1 = np.matrix([[1],
                    [0]
                    ])

    o1 = Q1.T

    alpha=np.array([1,0]) #left vector for V1
    mu=np.array([0,1])    #left vector for V2
    beta=np.array([0,0])  #right vector for V1
    nu=np.array([0,0])    #right vector for V2

elif correlator=='NQD':
    d=5
    # QD2 channel:
    Q2 = np.matrix([ [0],
                    [0],
                    [0],
                    [0],
                    [1]
                        ])

    o2 = Q2.T

    # QD1 channel:
    Q1 = np.matrix([ [0],
                    [1],
                    [0],
                    [0],
                    [0]
                        ])

    o1 = Q1.T

    alpha=np.array([0,1,1,0,0]) #left vector for V1
    mu=np.array([0,0,0,1,1])    #left vector for V2
    beta=np.array([0,1,0,1,0])  #right vector for V1
    nu=np.array([0,0,1,0,1])    #right vector for V2

    #the following are for the Bloch sphere:
    Ox = np.matrix([ [0],
                    [0],
                    [2],
                    [0],
                    [0]
                        ]).T

    Oy = np.matrix([ [0],
                    [0],
                    [0],
                    [2],
                    [0]
                        ]).T
    
    Oz = np.matrix([ [0],
                    [1],
                    [0],
                    [0],
                    [-1]
                        ]).T

#set defalt label for data and plots (plot titles need to be changed manually, if required)
def label(dotshape=dotshape, no_of_QDs=no_of_QDs, cavity=cavity, L=L,r0=r0, T=T, g=g, g1=g1, g2=g2,gd=gd, detuning=detuning, l=l, lp=lp, DvDc=DvDc, sharebath=sharebath, threshold_str=threshold_str, factortau=factortau, tf=tf):
    #edit this label as you wish
    if cavity==0:
        label2=f"{correlator}_{dotshape}_{int(no_of_QDs)}QDs_{int(cavity)}cavs_T{round(T*hbar/kb)}_g{round(g*hbar*1e3)}_R{np.round(np.float64(r0),3)}_L{L}_l{np.round(np.float64(l),1)}_lp{np.round(np.float64(lp),1)}_EC{ec}_MC{mc}_DvDc{DvDc}_det{np.round(detuning*hbar*1e3,1)}_sharebath{sharebath}_threshold{threshold_str}_factortau{factortau}_tf{tf}"
    if cavity==1 and no_of_QDs==1:
        label2=f"{correlator}_{dotshape}_QDs{int(no_of_QDs)}_cav{int(cavity)}_T{round(T*hbar/kb)}_g{round(gd*hbar*1e3)}_L{L}_l{np.round(np.float64(l),1)}_lp{np.round(np.float64(lp),1)}_EC{ec}_MC{mc}_DvDc{DvDc}_det{np.round(detuning*hbar*1e3,1)}_sharebath{sharebath}_threshold{threshold_str}_factortau{factortau}_tf{tf}"
    if cavity==1 and no_of_QDs==2:
        label2=f"{correlator}_{dotshape}_QDs{int(no_of_QDs)}_cav{int(cavity)}_T{round(T*hbar/kb)}_1g{round(g1*hbar*1e3)}_2g{round(g2*hbar*1e3)}_g{round(gd*hbar*1e3)}_R{np.round(np.float64(r0),3)}_L{L}_l{np.round(np.float64(l),1)}_lp{np.round(np.float64(lp),1)}_EC{ec}_MC{mc}_DvDc{DvDc}_det{np.round(detuning*hbar*1e3,1)}_sharebath{sharebath}_factortau{factortau}_threshold{threshold_str}_tf{tf}"
    return label2


for opt, arg in opts:
    if opt in ('-p', '--propagate'):
        exec(open('Forster_General_L.py').read())
        sys.exit()
    elif opt in ('-a', '--analytics'):
        exec(open('Forster_analytics+FGR.py').read())
        sys.exit()
    elif opt in ('-f', '--fit'):
        exec(open('PY_fit.py').read())
        sys.exit()
    else:
        pass
