#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:47:42 2022

@author: luba
"""

import numpy as np
import os
import sys
import getopt
from functions_ps import OMP, SHR

import warnings
warnings.filterwarnings("ignore")
path=os.getcwd()


kb=8.617333*1e-5 #in eV
evtps=6.582119e-4 #eV^(-1) to ps
####################################

#write all parameters in units of z
z=1#1.9713636368786647

#works up to L=27 on my pc
L=10
r0=0
#choose channels {'x','c'}:
ec='x'
mc='x'
#shortt timesteps (increase for accurate BB)
n=0
#temperature, K
T=20
#exciton-cavity coupling strength in micro eV:
g = 100#
#finishing time as a multiple of tauib:
pfin=100*2
#show graphs (0/1)?
shgr=1

try:
    opts, args = getopt.getopt(sys.argv[1:], 's:l:e:m:i:n:t:c:p:gfz', [ 'shgr=',
                                                                        'L=',
                                                                        'ec=',
                                                                        'mc=',
                                                                        'de0=',
                                                                        'N=',
                                                                        'T=',
                                                                        'g=',
                                                                        'pfin=',
                                                                        'getdata',
                                                                        'fit',
                                                                        'spectrum' ])
except getopt.GetoptError:
    pass
print(opts)
print(args)
for opt, arg in opts:
    if opt in ('-s', '--shgr'):
        shgr = int(arg)
    elif opt in ('-l', '--L'):
        L = int(arg)
    elif opt in ('-e', '--ec'):
        ec = arg
    elif opt in ('-m', '--mc'):
        mc = arg
    elif opt in ('-n', '--N'):
        n = int(arg)
    elif opt in ('-t', '--T'):
        T = int(arg)
    elif opt in ('-c', '--g'):
        g = int(arg)
    elif opt in ('-p', '--pfin'):
        pfin = int(arg)
    elif opt in ('-d', '--det'):
        det = int(arg)
    else:
        pass


T=kb*T/z/evtps
g=g*1e-6/z/evtps


#difference in defformation potential constants(conduction-valence) [eV]
Dcv=-6.5
#density [g/cm^3]
rm=5.65
rm=(rm*1e-3*1e6)*(1.97327e-7)**3/(1.782662e-36)
#speed of sound [km/s]
vs=4.6
vs=vs*1e3*(6.582119e-16)/(1.97327e-7)
Vs=4.6
#exciton confinement radius [nm]
l=3.3
l=l*1e-9/(1.97327e-7)
J0=Dcv**2/(4*np.pi**2*rm*vs**5)
w0=np.sqrt(2)*vs/l
J0= 51491.1
w0=w0/evtps/z
J0=J0*evtps**2*z**2
#polaron shift
omp=OMP(J0, w0).real
shr=SHR(T, J0, w0).real
#cavity-exciton detuning
det=omp

#in micro eV:
#Re exciton frequency
wx=1.3296*10**6*0
#exciton loss
gx=2*1e-6/evtps/z
#cavity loss
gc=30*1e-6/evtps/z
#effective g
geff = g*np.exp(-shr/2)*1e3*z*evtps

#IB timescale [ps]
tauib=3.25*z
#Trotter step

tdelay=r0/Vs
dt1=tdelay +1.2*tauib/(L+1)

#length of short-time region where the Trotter step depends on t
offset=0*tauib
if offset==0:
    n=L

#propagation end time
tf=150


# cavity chanel:
Q1c = np.matrix([[0],
                 [-1]
                ])

oc = Q1c.T

# exciton channel:
Q1x = np.matrix([[-1],
                 [0]
                ])

ox = Q1x.T


for opt, arg in opts:
    if opt in ('-g', '--getdata'):
        exec(open('LP_xc_past-future.py').read())
        sys.exit()
    if opt in ('-f', '--fit'):
        exec(open('fit.py').read())
        sys.exit()
    if opt in ('-z', '--spectrum'):
        exec(open('transform.py').read())
        sys.exit()
    else:
        pass

