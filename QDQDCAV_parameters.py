''' Parameters used in QD-QD system with shared phonon environment '''

import numpy as np
# from QDQDCAV_functions import PolaronShift_inim, PolaronShift
''' Set parameters here, all parameters in ps units'''
#choose number of neighbours
L=4
correlator='LP'
dotshape='smartie' #smartie
#choose whether to use a shared or independent bath (1 for shared, 0 for independent baths):
sharebath=1
#choose excitation and measuring channels {'1','2'}:
ec='1'
mc='1'
#shortt timesteps (increase for accurate BB)
n=0#set to 0 if you wish to calculate only longt dynamics (quicker calculation)
#temperature, K
T=20
#exciton-cavity coupling strength in micro eV:
g1 = 100#
g2 = 100
gd=0
#distance between QDs
r0=5
#phenomenological dampings of dot1, dot2, cavity
gamma1=0#2e-3
gamma2=0#2e-3
gammac=0#30e-3
#detuning between dots and cavity
detuning=0
#finishing time as a multiple of tauib:
pfin=80
#multiprocessing
mpt=0
#folder containing data
dat_fold='data'



###############################################################################

DvDc=6.5
dens=5.65
ps=1e-12
Vs= 4.6e3*ps * 1e9
hbar= 0.6582
kb=8.617e-2
l=7.5
lp=2.5
if dotshape=='spherical':
    lp=l
lbar = (np.sqrt((l**2 + lp**2)/2))
T=T*kb/hbar
w0=(np.sqrt(2)*Vs/l) #w0 in per picoseconds
j0=(DvDc**2 * (1e3)**2 * (3e8)**2 * (1e9)**2 * (1e-12)**2) / ( (2*np.pi)**2 * 5.65 * (1e-3) * (3e8)**2 * (6.24e21) *(1e-7)**3 * 4.6**5)
j0=j0/hbar #j0 in ps^2
#constant terms in K_{in,i(n+/- 1)} in meV ps^3 units (similar to j0 except v_s^4 instead of v_s^5 and an r0 factor)
j0_1=(DvDc**2 * (1e3)**2 * (3e8)**2 * (1e9)**2 * (1e-12)**2) / ( (2*np.pi)**2 * 5.65 * (1e-3) * (3e8)**2 * (6.24e21) *(1e-7)**3 * 4.6**4 *r0)
j0_1=j0_1/hbar

def PolaronShift(j0,w0):
    '''Polaron shift for in=im'''
    return -(j0*np.sqrt(np.pi)/4) * w0**3 

omp=PolaronShift(j0,w0)


g1=g1*1e-3 /hbar
g2=g2*1e-3 /hbar
gd=gd*1e-3/hbar
gamma1=gamma1*1e-3/hbar
gamma2=gamma2*1e-3/hbar
gammac=gammac*1e-3/hbar
detuning=detuning*1e-3 /hbar
# det=detuning  #!!
#real parts, QD energies:
Om1=0
Om2=0
OmC=detuning
w_qd1 = Om1 -1j*gamma1
w_qd2 = Om2 -1j*gamma2
w_c = OmC -1j*gammac

t0_data={0.0: 0.1, 1.0: 0.1, 2.0: 0.2, 3.0: 0.4, 4.0: 0.6, 5.0: 0.9, 6.0: 1.1, 7.0: 1.3, 8.0: 1.6, 9.0: 1.75, 10.0: 2.0, 11.0: 2.2, 12.0: 2.4, 13.0: 2.6, 14.0: 2.8, 15.0: 3.0, 16.0: 3.2, 17.0: 3.4, 18.0: 3.6, 19.0: 3.8, 20.0: 4.0, 21.0: 4.2, 22.0: 4.4, 23.0: 4.6, 24.0: 4.8, 25.0: 5.0, 26.0: 5.2, 27.0: 5.4, 28.0: 5.6, 29.0: 5.8, 30.0: 6.0, 31.0: 6.2, 32.0: 6.4, 33.0: 6.6, 34.0: 6.8, 35.0: 7.0, 36.0: 7.2, 37.0: 7.4, 38.0: 7.6, 39.0: 7.8, 40.0: 8.0, 41.0: 8.2, 42.0: 8.4, 43.0: 8.6, 44.0: 8.8, 45.0: 9.0, 46.0: 9.2, 47.0: 9.4, 48.0: 9.6, 49.0: 9.8, 50.0: 10.0, 51.0: 10.2, 52.0: 10.4, 53.0: 10.6, 54.0: 10.8, 55.0: 11.0, 56.0: 11.2, 57.0: 11.4, 58.0: 11.6, 59.0: 11.8, 60.0: 12.0, 61.0: 12.2, 62.0: 12.4, 63.0: 12.6, 64.0: 12.8, 65.0: 13.0, 66.0: 13.2, 67.0: 13.4, 68.0: 13.6, 69.0: 13.8, 70.0: 14.0, 71.0: 14.2, 72.0: 14.4, 73.0: 14.6, 74.0: 14.8, 75.0: 15.0, 76.0: 15.2, 77.0: 15.4, 78.0: 15.6, 79.0: 15.8, 80.0: 16.0, 81.0: 16.2, 82.0: 16.4, 83.0: 16.6, 84.0: 16.8, 85.0: 17.0, 86.0: 17.2, 87.0: 17.4, 88.0: 17.6, 89.0: 17.8, 90.0: 18.0, 91.0: 18.2, 92.0: 18.4, 93.0: 18.6, 94.0: 18.8, 95.0: 19.0, 96.0: 19.2, 97.0: 19.4, 98.0: 19.6, 99.0: 19.8, 100.0: 20.0, 101.0: 20.2, 102.0: 20.4, 103.0: 20.6, 104.0: 20.8, 105.0: 21.0, 106.0: 21.2, 107.0: 21.4, 108.0: 21.6, 109.0: 21.8, 110.0: 22.0, 111.0: 22.2, 112.0: 22.4, 113.0: 22.6, 114.0: 22.8, 115.0: 23.0, 116.0: 23.2, 117.0: 23.4, 118.0: 23.6, 119.0: 23.8, 120.0: 24.0, 121.0: 24.2, 122.0: 24.4, 123.0: 24.6, 124.0: 24.8, 125.0: 25.0, 126.0: 25.2}
t0=t0_data[np.round(r0)]

if dotshape=='spherical':
    tauib=np.sqrt(2)*np.pi*l/Vs
if dotshape=='smartie':
    tauib=np.sqrt(2)*np.pi *lbar/Vs
#Trotter step
# choose correct delay time t0
if sharebath ==1:
    dt1=( t0 + 1.0*tauib )/(L+1)
if sharebath ==0:
    dt1=2*tauib / (L+1)



#length of short-time region where the Trotter step depends on t
offset=t0+1.0*tauib
#propagation end time
tf=150

d=3
alpha=np.array([1,0]) #left vector for V1
mu=np.array([0,1])    #left vector for V2
beta=np.array([0,0])  #right vector for V1
nu=np.array([0,0])    #right vector for V2

# QD2 channel:
Q2 = np.matrix([[0],
                 [1],
                 [0]
                ])

o2 = np.matrix([[0, 1, 0]])

# QD1 channel:
Q1 = np.matrix([[1],
                 [0],
                 [0]
                ])

o1 = np.matrix([[1, 0, 0]])


def label(dotshape=dotshape, L=L,r0=r0, T=T, g1=g1, g2=g2,gd=gd, detuning=detuning, l=l,lp=lp, DvDc=DvDc, gamma1=gamma1, gamma2=gamma2, gammac=gammac, t0=t0, sharebath=sharebath, n=n):
    #edit this label as you wish
    # label=f"{correlator}_T{round(T*hbar/kb)}_g{round(g*hbar*1e3)}_R{round(r0)}_L{L}_l{np.round(l,1)}_EC{ec}_{DvDc}_det{np.round(detuning*hbar*1e3,1)}_gam{round(gamma1*hbar*1e3)}_gam{round(gamma2*hbar*1e3)}"
    if gd==0:
        label2=f"{correlator}_{dotshape}_T{round(T*hbar/kb)}_1g{round(g1*hbar*1e3)}_2g{round(g2*hbar*1e3)}_R{np.round(np.float64(r0),2)}_L{L}_l{np.round(np.float64(l),1)}_lp{np.round(np.float64(lp),1)}_EC{ec}_DvDc{round(DvDc,1)}_1gamma{round(gamma1*hbar*1e3)}_2gamma{round(gamma2*hbar*1e3)}_cgamma{round(gammac*hbar*1e3)}_det{(np.round(detuning*hbar*1e3,1))}_sharebath{sharebath}_n{n}.npy"
    if gd!=0:
        label2=f"{correlator}_{dotshape}_T{round(T*hbar/kb)}_1g{round(g1*hbar*1e3)}_2g{round(g2*hbar*1e3)}_gd{round(gd*hbar*1e3)}_R{np.round(np.float64(r0),2)}_L{L}_l{np.round(np.float64(l),1)}_lp{np.round(np.float64(lp),1)}_EC{ec}_DvDc{round(DvDc,1)}_1gamma{round(gamma1*hbar*1e3)}_2gamma{round(gamma2*hbar*1e3)}_cgamma{round(gammac*hbar*1e3)}_det{(np.round(detuning*hbar*1e3,1))}_sharebath{sharebath}_n{n}.npy"
    return label2

















