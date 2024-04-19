#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:47:46 2022

@author: Luba, Luke
"""

import matplotlib.pyplot as plt
from time import time
import numpy as np
import parameters_ps as pp
from functions_ps import DiagM, Kbb2, Kbb222, Kbb22
import  os

import warnings
warnings.filterwarnings("ignore")

path=os.getcwd()
os.chdir(path+"/data")
#################### Set Parameters ############################
# os.system('echo "start time:"; date +"%d-%m-%y %T"')
try:
    from subprocess import PIPE, run, check_output
    def out(command):
        result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
        return result.stdout
    output = out('echo ; date +"%d-%m-%y %T"')
    # print("start time:",output)
except:
    pass


d=2
r0=pp.r0
Vs=pp.Vs
z=pp.z
kb=pp.kb
evtps=pp.evtps
shgr=pp.shgr
Q1c=pp.Q1c
Q1x=pp.Q1x
oc=pp.oc
ox=pp.ox
ec=pp.ec
mc=pp.mc
L=pp.L
n=pp.n
offset=pp.offset
tf=pp.tf
tauib=pp.tauib
dt0=pp.dt1
wx=pp.wx
gx=pp.gx
gc=pp.gc
g =pp.g
det=pp.det
J0=pp.J0
w0=pp.w0
T=pp.T
omp=pp.omp
shr=pp.shr
N02=int(tf/dt0)+1


####trying to remove some vals####
epsilon=1e-4



longt=np.linspace(0,tf,num=int(N02),endpoint=True)
shortt=np.linspace(0,offset,num=int(n),endpoint=True)

file1 = open("log.txt","a")
file1.write('\n _______________Starting new run:________________ \n')
file1.write(f'the parameters are: S-T-pts={n}, tf={int(tf/tauib)}tauib, T{round(pp.T*z*evtps/kb)}K, g{round(pp.g*z*evtps*1e6)}microeV, L={L}, {ec}-x/c, \n')
file1.write(f'det={round(pp.det*z*evtps*1e6,2)}microeV, gx={round(pp.gx*z*evtps*1e6)}microeV, gc={round(pp.gc*z*evtps*1e6)}microeV, omp={round(pp.omp*z*evtps*1e6,2)}microeV \n')

try:
    file1.write(output)
except:
    pass

class propagate(object):
    """a set of functions to run numerics"""
    def __init__(self):

        self.dt=None
        self.data=[[],[]]
        self.zerocounter=[]

    def updateK(self, in_K):
        """update the set of cumulants"""
        self.K = in_K



    def Ntensor(self,r1,r2,L,dt):
        """tensor containing correlations in path variables for L nearest timesteps"""
        cumulants=self.K
        a2=self.alpha
        dd=np.full((1, L+1), d)[0]#self.dd
        c1= np.full(dd,1)
        kk=0
        modified=[]
        dim=self.axdim
        for i in range(0,L+1):
            axis = L-i
            dim_arr = dim
            dim_arr=dim_arr[0:axis]+(-1,)+dim_arr[axis+1::]
            if i==0:
                c1 = c1*(a2).reshape(dim_arr)
                a_new = (a2).reshape(dim_arr)
                c1a = c1*a_new
                K=c1a*cumulants[i]
                kk=kk+K
            else:
                a_new = (a2).reshape(dim_arr)
                c1a = c1*a_new
                K=c1a*cumulants[i]
                kk=kk+2*K
                modified.append(kk)
        N=np.exp(kk)
        Nmodified= np.exp(modified)
        return N, Nmodified

    def Ku(self,t):
        """Short time part of the cumulant"""
        Kinf=-1j*omp*t-shr
        return Kbb2(t, T, J0, w0)+Kinf#Kbb222(t,r0,Vs,J0,w0,T)       #

    def Kn(self,cumulants, dt):
        """Finds next cumulant in the set"""
        kk = []
        n = len(cumulants)
        for q in range(1, n):
            kk.append(2*(n+1-q)*cumulants[q])
        K = 0.5*(self.Ku((n+1)*dt)-(n+1)*self.Ku(dt)-sum(kk))
        self.Knn=K

    def cu(self, L, dt):
        """Calculates a set of square cumulants for L neighbours and updates an array"""
        if self.dt != dt:
            K1 = self.Ku(dt)
            cumulants = []
            cumulants.append(K1)
            while len(cumulants) < L+1:
                self.Kn(cumulants, dt)
                cumulants.append(self.Knn)
            self.updateK(in_K=cumulants)
            self.dt=dt



    def PlPast(self):
        """returns numeric data using inverted L approach"""
        if ec == "x":
            EC1 = Q1x
        if ec == "c":
            EC1 = Q1c

        dd = np.full((1, L+1), d)[0]

        self.EC1=EC1
        self.alpha=np.array([1,0])
        self.beta=np.array([0,0])
        self.dd=dd
        self.axdim=tuple(np.full((1, L+1),1).ravel())

        runtime=0
        tref=time()

        dt=dt0
        self.n2 = 1
        D, ww, U, V = DiagM(g, wx, gx, gc, det, omp)
        DD = np.diag(np.exp(-1j*ww*dt))
        M1 = U*DD*V
        self.cu(L, dt)
        cumulants = self.K
        K = cumulants[0]
        F2=np.ones((2,2))
        F2[0,0]=np.exp(K)

        if L==1:
            N=self.Ntensor(0,0,L,dt)[0]
            G = np.multiply(N, M1)

        for nt,t in enumerate(longt):   
            if t==0:
                #G=np.array([])
                Fn=np.asarray(EC1).reshape(d)
                Fs= np.asmatrix(Fn).T   
                # print(Fn)
            elif t==longt[1]:
                #G=(M1[0::,0]).squeeze()
                N=F2
                M=np.asarray(M1)
                G=np.multiply(N,M) 
                Fn=np.einsum('...i, ...->...i', G, Fn)   
                #Fn = np.einsum('i...l, ...l->i...', G, Fn)                                 
                Fs= np.asmatrix(np.sum(Fn.reshape(d**(nt),d),axis=0)).T      
            elif t==longt[2]:   #################################
                ind = np.full((1, nt+1), 1)
                self.axdim=tuple(ind.ravel())
                dd = np.full((1, int(nt+1)), d)[0]
                M = np.full(dd, M1)
                N=self.Ntensor(0,0,nt,dt)[0]
                G = np.multiply(N, M)                                          
                Fn=np.einsum('...i, ...->...i', G, Fn)
                Fs = np.asmatrix(np.sum(Fn.reshape(d**(nt),d),axis=0)).T       
            elif longt[2]<t<longt[L]:   #################################
                ind = np.full((1, nt+1), 1)
                self.axdim=tuple(ind.ravel())
                dd = np.full((1, int(nt+1)), d)[0]
                M = np.full(dd, M1)
                N=self.Ntensor(0,0,nt,dt)[0]
                G = np.multiply(N, M)                                          
                Fn=np.einsum('...i, ...->...i', G, Fn)
                Fs = np.asmatrix(np.sum(Fn.reshape(d**(nt),d),axis=0)).T               
            else:
                ind = np.full((1, L+1), 1)
                self.axdim=tuple(ind.ravel())
                dd = np.full((1, L+1), d)[0]
                M = np.full(dd, M1)
                N=self.Ntensor(0,0,L,dt)[0]
                G = np.multiply(N, M)    
                Fn = np.einsum('l...i, l...->...i', G, Fn)
                Fs = np.asmatrix(np.sum(Fn.reshape(d**(L-1),d),axis=0)).T
            # if 0<t<=longt[L]:
                # print(nt,'N',N.shape,'M',M.shape,'Fn', Fn.shape)
                # print(Fn)

            self.ntot=list(longt).index(t)+1
            PP=Fs

            self.data[1].append(PP)
            self.data[0].append(t)
            runtime=round(time()-tref,1)
            #print('step',self.ntot,'/',len(longt), ' runtime:',runtime,'s')
        # print('g=',round(pp.g*z*evtps*1e6),'L=',L,'with dt=',round(dt0/tauib,2),'tauib -','finished in',runtime,'s')
        return self.data

    def PlFuture(self):
        """returns numeric data for the regular L approach - special case: at least one of reduced basis states does not couple to phonons"""
        if ec == "x":
            EC1 = Q1x
        if ec == "c":
            EC1 = Q1c

        dd = np.full((1, L+1), d)[0]
        self.EC1=EC1
        self.alpha=np.array([1,0])
        self.beta=np.array([0,0])
        self.dd=dd
        self.axdim=tuple(np.full((1, L+1),1).ravel())

        runtime=0
        tref=time()

        ind = np.full((1, L-1), 1)
        ff = tuple(ind.reshape(1, -1)[0])

        dt=dt0
        self.n2 = 1
        D, ww, U, V = DiagM(g, wx, gx, gc, det, omp)
        DD = np.diag(np.exp(-1j*ww*dt))
        M1 = U*DD*V
        print('M1:',M1)
        self.cu(L, dt)
        cumulants = self.K
        print('cumulants:', cumulants)
        K = cumulants[0]
        F2=np.diag(np.exp(np.array([K,0])))
        dd = np.full((1, L+1), d)[0]
        M = np.full(dd, M1)
        N=self.Ntensor(0,0,L,dt)[0]  
        # print('Nshape',np.shape(N))
        G = np.multiply(N, M)                                ######################

        for t in longt:
            if t==0:
                Fn=np.asarray(EC1).reshape(d) 
                
                # print(Fn)
                Fs = np.asmatrix(Fn).T   
                
            elif t==longt[1]:
            
                print(Fn)
                Fn=np.einsum('i...l, ...l->i...', M, Fn)   
                # print(Fn)
                # print('F_2 has shape',Fn.shape,' and is given by \n', Fn )
                Fs = np.asmatrix(np.squeeze(Fn[ff])).T
                Fs=F2*Fs 
            elif t>longt[1]:
                Fn = np.einsum('i...l, ...l->i...', G, Fn)
                # print('the timestep is',t, 's', 'and F_n has shape',Fn.shape,' and is given by \n', Fn )
                Fs = np.asmatrix(np.squeeze(Fn[ff])).T
                Fs=F2*Fs      
                #if t==longt[3]:
                #    print('normal',Fn)
            self.ntot=list(longt).index(t)+1                                 

            self.data[1].append(Fs)
            self.data[0].append(t)
            runtime=round(time()-tref,1)
            ##print('step',self.ntot,'/',len(longt), ' runtime:',runtime,'s')
        print('original forward memory approach', 'g=',round(pp.g*z*evtps*1e6),'L=',L,'with dt=',round(dt0/tauib,2),'tauib -','finished in',runtime,'s')
        return self.data
    
    def PlFuture_opt(self):#trying to reduce some elements to 0
        """returns numeric data for the regular L approach - special case: at least one of reduced basis states does not couple to phonons"""
        if ec == "x":
            EC1 = Q1x
        if ec == "c":
            EC1 = Q1c

        dd = np.full((1, L+1), d)[0]
        self.EC1=EC1
        self.alpha=np.array([1,0])
        self.beta=np.array([0,0])
        self.dd=dd
        self.axdim=tuple(np.full((1, L+1),1).ravel())

        runtime=0
        tref=time()

        ind = np.full((1, L-1), 1)
        ff = tuple(ind.reshape(1, -1)[0])

        dt=dt0
        self.n2 = 1
        D, ww, U, V = DiagM(g, wx, gx, gc, det, omp)
        DD = np.diag(np.exp(-1j*ww*dt))
        M1 = U*DD*V
        self.cu(L, dt)
        cumulants = self.K
        K = cumulants[0]
        F2=np.diag(np.exp(np.array([K,0])))
        dd = np.full((1, L+1), d)[0]
        M = np.full(dd, M1)
        N=self.Ntensor(0,0,L,dt)[0]
        
        G = np.multiply(N, M)                                ######################
        
        ####    SVD test   ######
        # print('G has shape',np.shape(G))
        # print('wih form \n', G)
        square_matrix = G.reshape((-1, int(2**((L+1)/2))))
        # print('square_matrix shape', np.shape(square_matrix))
        U, S, Vt = np.linalg.svd(square_matrix)
        print('S is given by',S)
        S=np.array(S[np.where(S>1e-3)])
        diagonal_matrix = np.diag(S)
        # Ensure that U and Vt have compatible shapes for matrix multiplication
        U = U[:, :len(S)]  # Keep only the first len(S) columns of U
        Vt = Vt[:len(S), :]  # Keep only the first len(S) rows of Vt
        restored_matrix = U @ diagonal_matrix @ Vt
        # Reshape the restored matrix back to the original tensor shape
        G = restored_matrix.reshape(dd)
        # print('G_restored is\n', G)
        
        elementsbefore=2**(L+1)
        elementsafter=len(S) + 2*len(S)*int(2**((L+1)/2)) 
        frac=elementsafter/elementsbefore 
        print('E_after/E_before=', frac)

        
        
        
        
        
        
        for t in longt:
            if t==0:
                Fn=np.asarray(EC1).reshape(d) 
                
                # print(Fn)
                Fs = np.asmatrix(Fn).T   
                
            elif t==longt[1]:
                Fn=np.einsum('i...l, ...l->i...', M, Fn)   
                # print(Fn)
                # print('F_2 has shape',Fn.shape,' and is given by \n', Fn )
                Fs = np.asmatrix(np.squeeze(Fn[ff])).T
                Fs=F2*Fs 
            elif t>longt[1]:
                # print('the timestep is', t)
                Fn = np.einsum('i...l, ...l->i...', G, Fn)
                # if t==longt[155]:
                #     print('the timestep is',t, 's', 'and F_n unmodified has shape',Fn.shape,' and is given by \n', Fn )
                
                ### SVD test #####
                # original_shape=np.shape(Fn)
                # square_matrix = Fn.reshape((-1, int(2**((L)/2))))
                # U, S, Vt = np.linalg.svd(square_matrix)
                # # print('S is given by',S)
                # # print('U is given by',U)
                # # print('Vt is given by',Vt)

                # S=np.array(S[np.where(S>1e-3)])
                # diagonal_matrix = np.diag(S)
                # U = U[:, :len(S)]  # Keep only the first len(S) columns of U
                # Vt = Vt[:len(S), :]  # Keep only the first len(S) rows of Vt
                # print('U shape',np.shape(U))
                # print('Vt shape',np.shape(Vt))

                # restored_matrix = U @ diagonal_matrix @ Vt
                # print('restored matrix shape',np.shape(restored_matrix))

                # Fn = restored_matrix.reshape(original_shape)
                # elementsbefore=2**(L)
                # elementsafter=len(S) + 2*len(S)*int(2**((L)/2)) 
                # frac=elementsafter/elementsbefore 
                # print('E_after/E_before=', frac)
                
                
                
                ###    ##############     #####
                # if t==longt[155]:
                #     print('the timestep is',t, 's', 'and F_n restored has shape',Fn.shape,' and is given by \n', Fn )
                
                # Fn[np.where(np.abs(Fn)<epsilon)]=0
                zerocount=np.count_nonzero(Fn==0)
                # if t==longt[155]:
                #     print('the timestep is',t, 's', 'and F_n has shape',Fn.shape,' and is given by \n', Fn )
                Fs = np.asmatrix(np.squeeze(Fn[ff])).T
                Fs=F2*Fs      
                #if t==longt[3]:
                #    print('normal',Fn)
                self.zerocounter.append(zerocount)
            self.ntot=list(longt).index(t)+1                                 

            self.data[1].append(Fs)
            self.data[0].append(t)
            
            runtime=round(time()-tref,1)
            ##print('step',self.ntot,'/',len(longt), ' runtime:',runtime,'s')
        print('optimisation attempt approach', 'g=',round(pp.g*z*evtps*1e6),'L=',L,'with dt=',round(dt0/tauib,2),'tauib -','finished in',runtime,'s')
        return self.data, self.zerocounter
    
    
    def P_try(self):
        if ec == "x":
            EC1 = Q1x
        if ec == "c":
            EC1 = Q1c

        dd = np.full((1, 10+1), d)[0]
        self.EC1=EC1
        self.alpha=np.array([1,0])
        self.beta=np.array([0,0])
        self.dd=dd
        self.axdim=tuple(np.full((1, 1+1),1).ravel())

        runtime=0
        tref=time()

        ind = np.full((1, 10-1), 1)
        ff = tuple(ind.reshape(1, -1)[0])

        dt=dt0
        self.n2 = 1
        D, ww, U, V = DiagM(g, wx, gx, gc, det, omp)
        DD = np.diag(np.exp(-1j*ww*dt))
        M1 = U*DD*V
        self.cu(L, dt)
        cumulants = self.K 
        M = np.full(dd, M1)
        N=self.Ntensor(0,0,1,dt)[0] 
        # print(M1.shape)
        print('Nshape',N.shape)
        Mtilde = np.multiply(N, M1)   
        Qn=[]
        Qn.append(Mtilde)
        for i in range(2,11):
            Qn.append(np.array([[np.exp(2*cumulants[i]),1],[1,1]]))
        print(Qn[3])
        K = cumulants[0]
        F2=np.diag(np.exp(np.array([K,0])))
        # dd = np.full((1, L+1), d)[0]
       
        
                             
        for t in longt:
            if t==0:
                Fn=np.asarray(EC1).reshape(d) 
                
                # print(Fn)
                Fs = np.asmatrix(Fn).T   
                  
            elif t==longt[1]:
            
                Fn=np.einsum('i...l, ...l->i...', M, Fn)   

                
                # print('F_2 has shape',Fn.shape,' and is given by \n', Fn )
                Fs = np.asmatrix(np.squeeze(Fn[ff])).T
                # print('qdasdad',Fs)
                Fs=F2*Fs 
    
      
    
            elif t>longt[1]:
                
               for i in range(1,11):
                   # Fn=np.matmul(Qn[i-1],Fn)
                   Fn=np.einsum('i...l, ...l->i...',Qn[i-1],Fn)
               Fs = Fn
               Fs=F2*Fs      
            self.ntot=list(longt).index(t)+1                                 
        
            self.data[1].append(Fs)
            self.data[0].append(t)
            
            runtime=round(time()-tref,1)
            ##print('step',self.ntot,'/',len(longt), ' runtime:',runtime,'s')
        print('attempt at something', 'g=',round(pp.g*z*evtps*1e6),'L=',L,'with dt=',round(dt0/tauib,2),'tauib -','finished in',runtime,'s')
        return self.data
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def PlFutureLuke(self):
        """returns numeric data using regular L approach - general case"""
        if ec == "x":
            EC1 = Q1x
        if ec == "c":
            EC1 = Q1c
   
        dd = np.full((1, L+1), d)[0]
        self.EC1=EC1
        self.alpha=np.array([1,0])
        self.beta=np.array([0,0])
        self.dd=dd
        self.axdim=tuple(np.full((1, L+1),1).ravel())
   
        runtime=0
        tref=time()
   
        ind = np.full((1, L-1), 1)
        ff = tuple(ind.reshape(1, -1)[0])
   
        dt=dt0
        self.n2 = 1
        D, ww, U, V = DiagM(g, wx, gx, gc, det, omp)
        DD = np.diag(np.exp(-1j*ww*dt))
        M1 = U*DD*V
        self.cu(L, dt)
        cumulants = self.K
        K = cumulants[0]
        F2=np.diag(np.exp(np.array([K,0])))
        dd = np.full((1, L+1), d)[0]
        M = np.full(dd, M1)
        Nupdated=self.Ntensor(0,0,L,dt)[1]
        G = np.multiply(Nupdated, M)                                ######################
   
        Fn_initial = np.einsum('i...l, ...lr->i...r', M, EC1).squeeze()
        Fn_L1 = np.einsum('i...l, ...lr->i...r', M, EC1).squeeze()
        Fnfull= np.einsum('i...l, ...l->i...', G[L - 1], Fn_initial)
        # print(Fnfull)
        counter = 0

        for t in longt:
            if t == longt[1]:
                Fn_fixed=Fn_initial
            elif longt[1] < t <= longt[L]:
                if t == longt[2]:
                    Fn_fixed = np.einsum('i...l, ...l->i...', G[counter], Fn_initial)
                elif t > longt[2]:
                    Fn_fixed = np.einsum('i...l, ...l->i...', G[counter], Fn_initial)
                    for l in range(counter):
                        Fn_fixed = np.einsum('i...l, ...l->i...', G[counter-(l+1)], Fn_fixed)
                counter = counter + 1
                Fn = np.einsum('i...l, ...l->i...', G[L - 1], Fn_initial)
            if t > longt[L]:
                if L == 1:
                    Fn_L1=np.einsum('i...l, ...l->i...', G[0], Fn_L1)
                    Fn_fixed = Fn_L1
                if L > 1:
                    for i in range(0,L -1):
                        Fn=np.einsum('i...l, ...l->i...', G[L - (i+2)], Fn)
                        Fn_fixed=Fn
                    Fnfull=  np.einsum('i...l, ...l->i...', G[L - 1], Fnfull)
                    Fn=Fnfull
                    
            self.ntot=list(longt).index(t)+1 
        
        
            #Fs = np.asmatrix(np.squeeze(Fn_fixed[ff])).T
            #PP = F2 * Fs
                                            
            if t == 0:
                PP = 1
            if t > 0:
                Fs = np.asmatrix(np.squeeze(Fn_fixed[ff])).T
                PP = F2 * Fs
           
            if t >= pp.offset:
                self.data[1].append(PP)
                self.data[0].append(t)

            # runtime=round(time()-tref,1)
            #print('step',self.ntot,'/',len(longt), ' runtime:',runtime,'s')
        # print('g=',round(pp.g*z*evtps*1e6),'L=',L,'with dt=',round(dt0/tauib,2),'tauib -','finished in',runtime,'s')
        return self.data
    
        

#%% Get Data

runtime=0
tref=time()

# try:
#     data=np.load(path+f"/data/past_T{round(pp.T*z*evtps/kb)}_g{round(g*z*evtps*1e6)}_L{L}_{ec}.npy",allow_pickle=True)
# except:
#     try:
#         top=out('pidof crond; top -p 678 -b -n 1 > top.txt; head -5 top.txt')
#     except:
#         pass
    
#     #past/inverted L data
#     data=propagate().PlPast()
#     np.save(path+f"/data/past_T{round(pp.T*z*evtps/kb)}_g{round(g*z*evtps*1e6)}_L{L}_{ec}.npy",data)

# dat0=data[1]
# if mc=='c':
#     Pn=np.asarray([(oc[0,0:d]*i)[0,0] for i in dat0])
# if mc=='x':
#     Pn=np.asarray([(ox[0,0:d]*i)[0,0] for i in dat0])
# tps=np.asarray(data[0])/z


''' Future data original (one mode uncoupled to phonons)'''
try:
    data=np.load(path+f"/data/future_T{round(pp.T*z*evtps/kb)}_g{round(g*z*evtps*1e6)}_L{L}_{ec}.npy",allow_pickle=True)
except:
    data=propagate().PlFuture()
    variable2 = np.asarray(data, dtype="object")
    np.save(path+f"/data/future_T{round(pp.T*z*evtps/kb)}_g{round(g*z*evtps*1e6)}_L{L}_{ec}.npy",variable2) 
dat0=data[1]

if mc=='c':
    Pnf=np.asarray([(oc[0,0:d]*i)[0,0] for i in dat0])
if mc=='x':
    Pnf=np.asarray([(ox[0,0:d]*i)[0,0] for i in dat0])
tpsf=np.asarray(data[0])/z 
'''
'''


#Future data original (one mode uncoupled to phonons) and setting some elements of Fn to 0
# data,zerocounter=propagate().PlFuture_opt()

# # np.save(path+f"/data/future_T{round(pp.T*z*evtps/kb)}_g{round(g*z*evtps*1e6)}_L{L}_{ec}.npy",data)
# dat0=data[1] 
# if mc=='c':
#     Pnf_opt=np.asarray([(oc[0,0:d]*i)[0,0] for i in dat0])
# if mc=='x':
#     Pnf_opt=np.asarray([(ox[0,0:d]*i)[0,0] for i in dat0])
# tpsf_opt=np.asarray(data[0])/z 
'''

#'''

# data=propagate().P_try()
# # np.save(path+f"/data/future_T{round(pp.T*z*evtps/kb)}_g{round(g*z*evtps*1e6)}_L{L}_{ec}.npy",data)
# dat0=data[1]

# if mc=='c':
#     Pnftry=np.asarray([(oc[0,0:d]*i)[0,0] for i in dat0])
# if mc=='x':
#     Pnftry=np.asarray([(ox[0,0:d]*i)[0,0] for i in dat0])
# tpsftry=np.asarray(data[0])/z 










# #Luke's Future data
# data=propagate().PlFutureLuke()
# np.save(path+f"/data/future_T{round(pp.T*z*evtps/kb)}_g{round(g*z*evtps*1e6)}_L{L}_{ec}.npy",data)
# dat0=data[1]
# if mc=='c':
#     PnfLuke=np.asarray([(oc[0,0:d]*i)[0,0] for i in dat0])
# if mc=='x':
#     PnfLuke=np.asarray([(ox[0,0:d]*i)[0,0] for i in dat0])
# tpsfl=np.asarray(data[0])/z
#'''

'''
#Amy's data
Adat=np.loadtxt(f'Rawdata_T50_g50_gammaC3.000000e-02_L15_PXX_t.txt', skiprows=1)
ap=[]
at=[]
i=0
df=[]
for d in Adat:
    tima=d[0]
    pra=d[1]
    pia=d[2]
    ap.append(abs(pra+pia*1j))
    at.append(abs(tima))#*(6.582*10**-4)
    #df.append(abs(pra+pia*1j-dat[i]))
    i=i+1
'''

runtime=round(time()-tref,1)

#%% Plot

# xm=200
# y1=min(abs(Pn[n:np.where(tps>xm)[0][0]]))*0.9
# y2=1.5*max(abs(Pn))


# fig1 = plt.figure(10, figsize=(6,6),dpi=150)
# bb = fig1.add_subplot(1, 1, 1)
# # bb.plot(tps,abs(Pn),'b',label='LS') #past / inverted
# # bb.set_title(fr'L={L}')
# bb.plot(tpsf,abs(Pnf),'g-',linewidth='1',label=f'L={L}')
# plt.legend(loc='best') 
# # bb.plot(tps,abs(Pn),'b+',labe) #past / inverted
# # bb.plot(tpsf_opt,abs(Pnf_opt),'r--',label=fr'opt') 
# bb.plot(tpsftry,abs(Pnftry),'b--',label=fr'try') 

# bb.plot(tpsfl,abs(PnfLuke),'r--',label='LH fut.') #future / regular
# bb.plot(tpsf,abs(Pnf),'r+') 
#bb.plot(at,ap,'k',dashes=[2,2],label='AM') 
# bb.plot(tpsf,abs(Pnf),'r-',label='fut.') #future / regular

fig1 = plt.figure(10, figsize=(6,6),dpi=150)
# bb = fig1.add_subplot(1, 1, 1)
# bb.plot(tps,abs(Pn),'b',label='LS') #past / inverted
# bb.set_title(fr'L={L}')
plt.plot(tpsf,abs(Pnf),linewidth='1',label=f'L={L}')
plt.legend(loc='best') 
# plt.yscale('log')

# # error=abs(Pnf[3:]-Pnf_opt[3:])
# # bb.plot(tpsf[3:],error)


# bb.set_yscale('log')
# # bb.set_ylim(y1,y2)
# # bb.set_xlim(0,xm)
# bb.set_ylabel(r'$|P_{\mathrm{Lin}}(t)|$')
# bb.set_xlabel(r'$t \mathrm{(ps)}$')
# bb.legend()
# # fig1.savefig(path+f'/plots/LP_T{round(pp.T*z*evtps/kb)}_g{round(pp.g*z*evtps*1e6)}_{ec}-{mc}_L{pp.L}.pdf', bbox_inches='tight',transparent=True)
# if shgr==1:
#     plt.show()
# plt.close(fig1)

# zerocounter=np.array(zerocounter)
# f=zerocounter/(2**(L))

# fig1 = plt.figure(figsize=(6,6),dpi=150)
# bb = fig1.add_subplot(1, 1, 1)
# timesteps=np.arange(2,len(longt))
# bb.plot(timesteps,f,label=fr'$\epsilon$ ={epsilon}')
# bb.set_ylabel('f (zeros/total elements)')
# bb.set_xlabel('timestep, t_n')
# bb.set_ylim(-0.1,1)
# plt.legend()



# ci=0

# styles=['-',':',(0,(4,2,1,2,1,2))]
# colours=['k','tab:blue','k']

# # xmax = params.tf-1
# # ymin = 1e-3 #min(abs(Pn[n:np.where(tm > xm)[0][0]])) * 0.9
# # ymax = 1.0 #max(abs(Pn))
# colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# ci=0
# fig1 = plt.figure(1,figsize=(6,6),dpi=150)
# bb = fig1.add_subplot(1, 1, 1)
# # plt.title(f'{correlator}_{mc}{ec}: {dotshape}, g={round(params.g0)}$(d_0/d)^3$={round(params.g*hbar*1e3)}$\mu$eV,  det={np.round(params.detuning*hbar*1e3,1)}$\mu$eV,  T={round(params.T*hbar/kb)}K \n $d_0$={params.sep0} nm, $d$={params.r0} nm, l={np.round(l,1)}nm, l$_\perp$={np.round(lp,1)}nm, DcDv={params.DvDc}eV') #
# #plt.title(f'{correlator}: P_{mc}{ec}, g={round(params.g*hbar*1e3)}$\mu$eV, $d$={params.r0} nm, DvDc={params.DvDc}eV, det={np.round(params.detuning*1e3,1)}$\mu$eV')#l={np.round(l,1)}nm')
# bb.set_ylabel(r'$|P_{11}(t)|$',fontsize='8')
# bb.set_xlabel(r'time $\mathrm{(ps)}$',fontsize='8')

# for LL in np.array([8,14]):# [10,9,8,7,6]: # 
#     L=LL
#     try:
#         data=np.load(path+f"/data/future_T{round(pp.T*z*evtps/kb)}_g{round(g*z*evtps*1e6)}_L{LL}_{ec}.npy",allow_pickle=True)
#     except:
#         data=propagate().PlFuture()
#         variable2 = np.asarray(data, dtype="object")
#         np.save(path+f"/data/future_T{round(pp.T*z*evtps/kb)}_g{round(g*z*evtps*1e6)}_L{LL}_{ec}.npy",variable2) 
#     dat0=data[1]

#     if mc=='c':
#         Pnf=np.asarray([(oc[0,0:d]*i)[0,0] for i in dat0])
#     if mc=='x':
#         Pnf=np.asarray([(ox[0,0:d]*i)[0,0] for i in dat0])
#     tpsf=np.asarray(data[0])/z 
#     bb.plot(tpsf, np.abs(Pnf),color=colorcycle[ci], linewidth='1', zorder=1+ci, label=f'L={LL}')
#         # plt.figure(2)
#         # print(n)
#         # plt.plot(tm, Pn, linewidth='2', label=f'L={LL}',alpha=1)
#         #plot for the other measurement channel Pomc:
#         #bb.plot(tm, Pomc,color=colorcycle[ci],linestyle=styles[1])#,label=f'2nd QD')
#     ci=ci+1
   
# bb.set_yscale('log')
# leg=bb.legend(loc=4)
# plt.legend(loc=4,ncol=3,columnspacing=0.8)
# plt.legend(bbox_to_anchor=(1.01,1), fancybox=True, framealpha=0.0, loc="upper left")
# leg.get_frame().set_color('none')
# plt.tight_layout()
    # bb.set_ylim(0, 1.222)
# bb.set_xlim(0,30)
# bb.set_xticks([0,5,10,15,20,25])
# bb.set_yticks([np.round(0,0),0.5,1.0])
# plt.xticks(fontsize='8')
# plt.yticks(fontsize='8')
#bb.set_ylim(ymin, ymax)




######################## K function test ####################



# # Rs=np.linspace(0.01,40,5)
# times=np.linspace(0,20,100)
# # L=15
# Rs=([50])
# # Rs=([20,40,60,80,100])
# # starttimes=[]
# for i in range(len(Rs)):
#     r0 = Rs[i]
#     # t0=r0/Vs
#     # dt1=( t0 + 1.2*tauib )/(L+1)
    
#     Ks=[]
#     for t in (times):    
        
#         Ks.append(Kbb222(t,r0,Vs,J0,w0,T))
#     Ks=np.asarray(Ks,dtype=np.complex128)   


    
    
    
#     plt.figure()
#     plt.plot(times,np.real(Ks),label='R = {:.2f} nm'.format(r0))
#     # plt.plot((L+1)*dt1, np.min(np.real(K11s)), 'rx', linewidth=1.5)
#     # plt.title(f' T={round(T*hbar/kb)}K, l={np.round(l,1)}nm, DcDv={DvDc}eV') 
#     plt.ylabel('$K$')
#     plt.xlabel('Time (ps)')
#     plt.legend(loc='best')




# times=np.linspace(0,20,100)
# # L=15
# Rs=([50])
# # Rs=([20,40,60,80,100])
# # starttimes=[]
# for i in range(len(Rs)):
#     r0 = Rs[i]
#     # t0=r0/Vs
#     # dt1=( t0 + 1.2*tauib )/(L+1)
    
#     Ks=[]
#     for t in (times):    
        
#         Ks.append(Kbb22(t, T, J0, w0))
#     Ks=np.asarray(Ks,dtype=np.complex128)   


    
    
    
#     plt.figure()
#     plt.plot(times,np.real(Ks),label='R = {:.2f} nm'.format(r0))
#     # plt.plot((L+1)*dt1, np.min(np.real(K11s)), 'rx', linewidth=1.5)
#     # plt.title(f' T={round(T*hbar/kb)}K, l={np.round(l,1)}nm, DcDv={DvDc}eV') 
#     plt.ylabel('$K$')
#     plt.xlabel('Time (ps)')
#     plt.legend(loc='best')



















