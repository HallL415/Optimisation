
''' Trotter's decomposition method - L neighbour regime used in QD-QD system with shared phonon environment '''
import os
import matplotlib.pyplot as plt
from time import time
from scipy.linalg import expm
import numpy as np
import QDQDCAV_parameters as params
from QDQDCAV_functions import LFpol, S_inin, S_inim, PolaronShift, PolaronShift_inim, phi_inin, phi_inim, K11_smartie,K12_smartie

path=os.getcwd()


#################### Set Parameters ############################
correlator=params.correlator
dotshape=params.dotshape
d = 3
mpt = params.mpt
cores = 6
Q2 = params.Q2
Q1 = params.Q1
o2 = params.o2
o1 = params.o1
ec = params.ec
mc = params.mc
L = params.L
n = params.n
offset = params.offset
tf = params.tf
tauib = params.tauib
dt0 = params.dt1  # long time timestep
hbar = params.hbar
kb = params.kb
w_qd1 = params.w_qd1
detuning=params.detuning
w_qd2 = params.w_qd2
w_c=params.w_c
DvDc=params.DvDc
g1 = params.g1
g2 = params.g2
gd=params.gd
j0 = params.j0
j0_1 = params.j0_1
w0 = params.w0
T = params.T
r0 = params.r0
l = params.l
lp=params.lp
Vs = params.Vs
sharebath=params.sharebath


longt=np.arange(0,tf+1,dt0)
shortt = np.linspace(0, r0/Vs +1.2*tauib, num=int(n), endpoint=True)
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class Propagate(object):
    """a set of functions to run numerics"""

    def __init__(self):

        self.dt = None
        self.data = [[], []]
        self.L = L  # no. of neighbours
        self.K = None  # Kinim for in=\=im
        self.Kinin = None  # Kinin for diagonals or when in=im
        self.dd = np.full((1, self.L + 1), d)[0]  # defining the dimensions of the matrices
        self.flattened = []
        # inim refers to indices

    def updateK(self, in_K, in_Kinin):
        """update the set of cumulants"""
        self.K = in_K
        self.Kinin = in_Kinin

    def Ku(self, t):  # writing Kinin(t) expression and Kinim(t) (in=\=im)
        """Short time part of the cumulant"""
        if dotshape=='spherical':        
            Kinf0 = -1j * PolaronShift(j0, w0) * t - S_inin(T, j0, w0)
            Kinf1 = (-1j * PolaronShift_inim(j0, w0, r0, l) * t - S_inim(T, j0_1, w0, r0, Vs))
            K11s=phi_inin(t, j0, w0, T) + Kinf0
            K12s=sharebath*(phi_inim(t, j0_1, w0, T, r0,Vs) + Kinf1)
        if dotshape=='smartie':
            K11s=K11_smartie(t, j0, l, lp, Vs, T)
            K12s=sharebath*(K12_smartie(t, j0, l, lp, Vs, T, r0))
        return K11s,  K12s  # gives Kinin(t) and Kinim(t) respectively

    def Kn(self, cumulants, cumulantsinin, dt):
        """Finds next cumulant in the set"""
        kk = []
        kkinin = []
        n = len(cumulants)
        for q in range(1, n):
            kk.append(2 * (n + 1 - q) * cumulants[q])
            kkinin.append(2 * (n + 1 - q) * cumulantsinin[q])
        Kinims = 0.5 * (self.Ku((n + 1) * dt)[1] - (n + 1) * self.Ku(dt)[1] - sum(kk))
        Kinins = 0.5 * (self.Ku((n + 1) * dt)[0] - (n + 1) * self.Ku(dt)[0] - sum(kkinin))
        self.Knn = Kinims
        self.Knn_inin = Kinins



    def cu(self, dt):
        """Calculates a set of square cumulants for L neighbours and updates an array"""
        if self.dt != dt:
            K0 = self.Ku(dt)[0]  # K(dt) i.e. K_inin0
            K0_inim = self.Ku(dt)[1]
            cumulants = []
            cumulantsinin = []
            cumulants.append(K0_inim)
            cumulantsinin.append(K0)
            while len(cumulants) < self.L + 1:
                self.Kn(cumulants, cumulantsinin,
                        dt)  # calculate K^(n) for Kinin or Kinim i.e. K_11^(1), K_22(2) or K_12^(1), K_12^(2) etc.
                # for however many L
                cumulants.append(self.Knn)
                cumulantsinin.append(self.Knn_inin)
            self.updateK(in_K=cumulants, in_Kinin=cumulantsinin)
            self.dt = dt


    def Ntensor(self,L):  # filling in the cumulant contributions contained in the exp{} term in G.
        """tensor containing correlations in path variables for L nearest timesteps"""
        cumulants = self.K
        cumulantsinin = self.Kinin  # from updated cumulants method
        dd=np.full((1, L+1), d)[0]
        # dd=np.empty(L+1, dtype=int)
        # dd.fill(d)
        c1 = np.full(dd, 1)
        kk = 0
        # modified= []

        for i in range(0, L + 1):

            if i == 0:
                
                c1a = c1*np.array([[1,1,0],[1,1,0],[1,1,0]])   #correct for K0 placements
                
                K = c1a * cumulantsinin[i]
                kk = kk + K

            if i == 1:
                c1 = np.full(dd, 1)
                c1a = c1 * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0] ])
                c1ainin = c1 * np.array([[1, 0, 0], [0, 1, 0],[0, 0, 0]])
                Kinim = c1a * cumulants[i]
                Kinin = c1ainin * cumulantsinin[i]
                kk = kk + 2 * Kinim + 2 * Kinin
                # modified.append(kk)
            if  i > 1:
                dd1=np.full((1,i+1),d)[0]
                # dd1=np.empty(i+1, dtype=int)
                # dd1.fill(d)
                c1 = np.full(dd1, 1)

                c1a=np.concatenate((c1[0]*np.array([[1,0,0],[1,0,0],[1, 0, 0]]), c1[1]*np.array([[0,1,0],[0,1,0],[0,1,0]]), c1[2]*np.array([[0,0,0],[0,0,0],[0,0,0]]))).reshape(dd1)
                c1a=np.resize(c1a,dd)
                c1a_inim=np.concatenate((c1[0]*np.array([[0,1,0],[0,1,0],[0,1,0]]), c1[1]*np.array([[1,0,0],[1,0,0],[1,0,0]]),c1[2]*np.array([[0,0,0],[0,0,0],[0,0,0]]))).reshape(dd1)
                c1a_inim=np.resize(c1a_inim,dd)

                Kinin = c1a * cumulantsinin[i]
                Kinim = c1a_inim * cumulants[i]
                kk = kk + 2 * Kinin + 2 * Kinim
                # modified.append(kk)

        N = np.exp(kk)
        # Nmodified= np.exp(modified)
        return N

    def Pfwm(self):
        """returns numeric data"""

        if ec == "1":
            EC1 = Q1
        if ec == "2":
            EC1 = Q2

        self.EC1 = EC1
        runtime = 0
        tref = time()
        def Pmap(t):
          """generates short-time data"""
      
          ntot = list(shortt).index(t) + 1
          dt = t / (L + 1)
      
          self.cu(dt)  # calculates all cumulants K0,K1,...,K_L for a given dt(variable t for short times) and stores
          # in an array
          cumulantsinin = self.Kinin  # K stores the cumulants calculated in line above.
          K = cumulantsinin[0]  # defining K as K_0 (first cumulant given by IB model cumulant calculated at dt)
          
          if correlator=='LP':
              LF=LFpol(g1, g2,gd, w_qd1, w_qd2, w_c)
              M1=expm(-1j*LF*dt)
              F2 = np.diag(np.exp(np.array([K, K, 0])))

          else: #fix
              # LF = LFpop(g,  w_qd1.real, w_qd2.real, -params.gamma1/hbar, -params.gamma2/hbar) 
              F2 = np.diag(np.exp(np.array([K,K,K,K,K])))
              # M1=expm(-1j * LF * dt)
          lvals=np.arange(1,L+1) #set of L values we care about
          Glist=[] #A list of Gtensors for different L
          FnList_init=[]
          for l in lvals:
              #print(l)
              N=self.Ntensor(l)
              dd=np.full((1, l + 1), d)[0] 
              M = np.full(dd, M1)
              G = np.multiply(N, M)
              Glist.append(G) 
              Fn_prop=np.einsum('i...l, ...lr->i...r', M, EC1).squeeze() #this is the single Ftensor to be used for further propagation
              FnList_init.append(Fn_prop)
      #     M = np.full(self.dd, M1)
      
      #     Nupdated = self.Ntensor(L)[1]
      
      
      #     G = np.multiply(Nupdated,M) # combines M with the e^(K0+2K1+...2KL) but its a 2x2x2 matrix for L=2
          Fn = np.einsum('i...l, ...lr->i...r', M, EC1).squeeze()  # 2x2 matrix for L=2.   F_iL..i1 = M_i1k => for L=2 -> ([M11,M21],[M11,M21])
      #     # EC1 extracts the correct M matrix elements since it's Mi1K, we want M11 and M21. F^(1)
          for s in range(0, L):
              Fn = np.einsum('...l, ...l->...', Glist[L - (s+1)], Fn)
      #     self.n2 = 0
      
          # Fs = np.asmatrix(np.squeeze(Fn[ff])).T  # creates a 2x1 matrix F2..2j #[ff] takes the situation where all
      #     # axis = 2 except last one, where it can be 1 or 2 (j) # the F_2..2j, part, the j=1 or 2 is taken into
      #     # account later when extracting data.
          Fs = np.asmatrix(Fn).T
          PP = F2 * Fs
          # PP = F2 * Fs  # first term multiplied by F_2..2j, should give ([[e^K_11^(0) F2..2j], [ e^K_22^*F_2...2j]])
      #     # (column vector), later multiply by [1,0] or [0,1] to extract which you want
      
          runtime = round(time() - tref, 1)
          print('step', ntot, '/', n, ', runtime:', runtime, 's')
          return PP
      
      
        for t in shortt:
            PP = Pmap(t)
            self.data[1].append(PP)
            self.data[0].append(t)
      
            
        '''time beyond tau_ib'''
        dt = dt0
        print(dt)
        self.n2 = 1
        # ws, Y, Yt = DiagM(g1, g2, w_qd1, w_qd2, w_c)
        # DD = np.diag(
        #     np.exp(-1j * ws * dt))  # diagonalised part of JC Hamiltonian, mathematically equivalent to H_foerster
        # M1 = Y * DD * Yt
        LF=LFpol(g1, g2,gd, w_qd1, w_qd2, w_c)
        M1=expm(-1j*LF*dt)
        self.cu(dt)
        cumulantsinin = self.Kinin
        K = cumulantsinin[0]
        F2 = np.diag(np.exp(np.array([K, K, 0])))
        
        
        lvals=np.arange(1,L+1) #set of L values we care about
        Glist=[] #A list of Gtensors for different L
        FnList_init=[]
        for l in lvals:
            N=self.Ntensor(l)
            
            dd=np.full((1, l + 1), d)[0] 
            # dd=np.empty(l+1, dtype=int)
            # dd.fill(d)
            M = np.full(dd, M1)
            G = np.multiply(N, M)
            Glist.append(G) 
            Fn_prop=np.einsum('i...l, ...lr->i...r', M, EC1).squeeze() #this is the initial Ftensor to be used for further propagation
            FnList_init.append(Fn_prop) #this is a list of Ftensors to be used for initial propagation (for the first L datapoints)


        for ii,t in enumerate(longt):  
            if t ==longt[0]:
                PP = EC1
            elif t == longt[1]:
                Fn_obs=FnList_init[0]
                # print(Fn_obs.shape)
            elif t<=longt[L]:
                li=np.arange(1,ii-1)
                # print(li)
                Fn_obs = np.einsum('...l, ...l->...', Glist[ii-2], FnList_init[ii-2]) #this is the single Ftensor which will contribute to the observed result
                if t > longt[2]:
                    for s in li:
                        Fn_obs = np.einsum('...l, ...l->...', Glist[ii-s-2], Fn_obs)  
            else:
                for l in lvals:
                    if l==1:
                        Fn_obs = np.einsum('...l, ...l->...', Glist[L-l], Fn_prop)
                        
                    else:
                        Fn_obs = np.einsum('...l, ...l->...', Glist[L-l], Fn_obs)
                    #print(Fn_obs.shape,Glist[L-l].shape)
                Fn_prop = np.einsum('i...l, ...l->i...', Glist[L-1], Fn_prop)  #this line is what keeps stacking full Gs ontop of eachother for long times where there are several L shapes with no cutoffs

            self.ntot = list(longt).index(t) + 1

            if t > 0:
                Fs = np.asmatrix(Fn_obs).T
                PP = F2 * Fs
            if t>r0/Vs +1.2*tauib and n != 0:
                self.data[1].append(PP)
                self.data[0].append(t)
            if n == 0: 
                self.data[1].append(PP)
                self.data[0].append(t)
            runtime = round(time() - tref, 1)
            print('step', self.ntot, '/', len(longt), ', runtime:', runtime, 's')
        print('g1=', round(params.g1 * hbar * 1e3), 'micro eV', 'g2=', round(params.g2 * hbar * 1e3), 'micro eV', 'L=', L, 'with dt=', round(dt0 , 2),
              'ps -', 'l=', round(params.l), 'nm','lp=', round(params.lp), 'nm'
              'finished in', runtime, 's')
        return self.data
    


runtime=0
tref=time()


SMALL_SIZE = 8  
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

try:
    data=np.load(path+f"/data/{correlator}_{dotshape}_T{round(T*hbar/kb)}_1g{round(g1*hbar*1e3)}_2g{round(g2*hbar*1e3)}_R{np.round(np.float64(r0),2)}_L{L}_l{np.round(np.float64(l),1)}_lp{np.round(np.float64(lp),1)}_EC{ec}_DvDc{round(DvDc,1)}_det{(np.round(detuning*hbar*1e3,1))}_sharebath{sharebath}_n{n}.npy",allow_pickle=True)
except:
    data=Propagate().Pfwm()
    data = np.asarray(data, dtype="object")
    np.save(path+f"/data/{correlator}_{dotshape}_T{round(T*hbar/kb)}_1g{round(g1*hbar*1e3)}_2g{round(g2*hbar*1e3)}_R{np.round(np.float64(r0),2)}_L{L}_l{np.round(np.float64(l),1)}_lp{np.round(np.float64(lp),1)}_EC{ec}_DvDc{round(DvDc,1)}_det{(np.round(detuning*hbar*1e3,1))}_sharebath{sharebath}_n{n}.npy",data)

dat0 = data[1]
tm = np.asarray(data[0])
P1 = np.asarray([(o1[0, 0:d] * i)[0, 0] for i in dat0])  # o2/o1 are measuring channel, in 1 or 2.
P2 = np.asarray([(o2[0, 0:d] * i)[0, 0] for i in dat0])  # [0,0] just extracts value of matrix element.

if mc == '2':
    Pn= P2
elif mc == '1':
    Pn = P1



fig1 = plt.figure(10, figsize=(6,6),dpi=150)

plt.plot(tm,abs(Pn),linewidth='1',label=f'L={L}')
plt.legend(loc='best') 
plt.yscale('log')









# L=1
# try:
#     data=np.load(path+f"/{params.dat_fold}/"+params.label(L=1)+".npy",allow_pickle=True)
# except:
#     data=Propagate().Pfwm()
#     np.save(path+f"/{params.dat_fold}/"+params.label(L=1)+".npy",data)

# dat0 = data[1]
# tm = np.asarray(data[0])
# P1 = np.asarray([(o1[0, 0:d] * i)[0, 0] for i in dat0])  # o2/o1 are measuring channel, in 1 or 2.
# P2 = np.asarray([(o2[0, 0:d] * i)[0, 0] for i in dat0])  # [0,0] just extracts value of matrix element.

# if mc == '2':
#     Pn= P2
# elif mc == '1':
#     Pn = P1


# fig1 = plt.figure( figsize=(4.5,3),dpi=200)
# bb = fig1.add_subplot(1, 1, 1)
# # plt.title(f'{correlator}: P_{mc}{ec}, $g_1$={round(params.g1*hbar*1e3)}$\mu$eV, $g_2$={round(params.g2*hbar*1e3)}$\mu$eV,\n $d$={params.r0} nm, det={np.round(params.detuning*hbar*1e3,1)}$\mu$eV')#l={np.round(l,1)}nm')
# bb.set_ylabel(r'$|P_{{11}}(t)|$')
# bb.set_xlabel(r'$t \mathrm{(ps)}$')
# if correlator=='LP':  
#     Pn=abs(Pn)
#     P2=abs(P2)
# bb.plot(tm, (Pn),label=f'NN')
# xm = params.pfin*tauib-1
# y1 = 1e-3 #min(abs(Pn[n:np.where(tm > xm)[0][0]])) * 0.9
# y2 = 1.2 #* max(abs(Pn))
# if correlator=='LP':  
#     bb.set_yscale('log')
#     bb.set_ylim(y1, y2)
# bb.set_xlim(0, xm)
# leg=bb.legend(loc=1)
# plt.legend(bbox_to_anchor=(1.01,1), loc="upper left")
# leg.get_frame().set_color('none')
# plt.tight_layout()
# # plt.show()

# L=16
# try:
#     data=np.load(path+f"/{params.dat_fold}/"+params.label(L=16)+".npy",allow_pickle=True)
# except:
#     data=Propagate().Pfwm()
#     np.save(path+f"/{params.dat_fold}/"+params.label(L=16)+".npy",data)

# dat0 = data[1]
# tm = np.asarray(data[0])
# P1 = np.asarray([(o1[0, 0:d] * i)[0, 0] for i in dat0])  # o2/o1 are measuring channel, in 1 or 2.
# P2 = np.asarray([(o2[0, 0:d] * i)[0, 0] for i in dat0])  # [0,0] just extracts value of matrix element.

# if mc == '2':
#     Pn= P2
# elif mc == '1':
#     Pn = P1


# fig1 = plt.figure( figsize=(4.5,3),dpi=200)
# bb = fig1.add_subplot(1, 1, 1)
# bb.plot(tm, (Pn),label=f'L=16')
# fig1.savefig(path+f'/plots/'+params.label()+'.pdf', bbox_inches='tight',transparent=True)










'''Trying QDCav version'''
# data=Propagate().P2()

# dat0 = data[1]

# P1 = np.asarray([(o1[0, 0:d] * i)[0, 0] for i in dat0])  # o2/o1 are measuring channel, in 1 or 2.
# P2 = np.asarray([(o2[0, 0:d] * i)[0, 0] for i in dat0])  # [0,0] just extracts value of matrix element.

# if mc == '2':
#     Pn= P2
# elif mc == '1':
#     Pn = P1
# tm = np.asarray(data[0])
# xm = params.pfin*tauib-1
# y1 = 1e-3 #min(abs(Pn[n:np.where(tm > xm)[0][0]])) * 0.9
# y2 = 1.2 #* max(abs(Pn))


# fig1 = plt.figure(2, figsize=(4.5,3),dpi=200)
# bb = fig1.add_subplot(1, 1, 1)
# plt.title(f'{correlator}: P2, P_{mc}{ec}, g={round(params.g1*hbar*1e3)}$\mu$eV, $d$={params.r0} nm, DvDc={params.DvDc}eV, det={np.round(params.detuning*hbar*1e3,1)}$\mu$eV')#l={np.round(l,1)}nm')
# bb.set_ylabel(r'$|P_{\mathrm{Lin}}(t)|$')
# bb.set_xlabel(r'$t \mathrm{(ps)}$')
# if correlator=='LP':  
#     Pn=abs(Pn)
#     P2=abs(P2)
# bb.plot(tm, (Pn),label=f'L={L}')
# if correlator=='LP':  
#     bb.set_yscale('log')
#     bb.set_ylim(y1, y2)
# bb.set_xlim(0, xm)
# leg=bb.legend(loc=1)
# plt.legend(bbox_to_anchor=(1.01,1), loc="upper left")
# leg.get_frame().set_color('none')
# plt.tight_layout()
# plt.show()

# For plotting no phonons curve:
# if correlator=='NQD':    
#     def NQD1(t):
#         LF = LFpop(g,  w_qd1.real, w_qd2.real, -params.gamma1/hbar, -params.gamma2/hbar) 
#         M1=expm(-1j * LF * t)
#         #print((M1*Q1))
#         return (o1*M1*Q1)[0,0],(o2*M1*Q1)[0,0]
#     longt=np.arange(0,260,dt0)
#     N1=[]
#     N2=[]
#     for t in longt:
#         N1.append(NQD1(t)[0])
#         N2.append(NQD1(t)[1])
# else:
    
    
'''from here'''
# def NQD1(t):
#     LF = LFpol(g1, g2, w_qd1, w_qd2, w_c)
#     M1=expm(-1j * LF * t)
#     #print((M1*Q1))
#     return (o1*M1*Q1)[0,0],(o2*M1*Q1)[0,0]
# longts=np.arange(0,260,0.1)
# N1=[]
# N2=[]
# for t in longts:
#     N1.append(abs(NQD1(t)[0]))
#     N2.append(abs(NQD1(t)[1]))

# tm = np.asarray(data[0])
# runtime=round(time()-tref,1)



# fig1 = plt.figure( figsize=(3.5,2),dpi=300)
# xm = longt[-1]+1
# y1 = 0.01 #min(abs(Pn[n:np.where(tm > xm)[0][0]])) * 0.9
# y2 = 1.00 #* max(abs(Pn))
# bb = fig1.add_subplot(1, 1, 1)
# plt.title(f'{correlator}: P_{mc}{ec}, {dotshape}, L={L}, $g_1$={round(params.g1*hbar*1e3)}$\mu$eV, $g_2$={round(params.g2*hbar*1e3)}$\mu$eV, $g$={round(params.gd*hbar*1e3)}$\mu$eV, $d$={params.r0} nm \n l={round(params.l,1)}, lp={round(params.lp,1)},  DvDc={round(params.DvDc,1)}eV, det={np.round(params.detuning*hbar*1e3,1)}$\mu$eV')
# bb.set_ylabel(r'$|P_{11}(t)|$')
# bb.set_xlabel(r'time $\mathrm{(ps)}$')
# # if correlator=='NQD':  
# #     #bb.plot(longt, N1,'k',linewidth=1,dashes=[2,2],label=f'no ph.')
# #     bb.set_ylabel(r'N(t)')
# # bb.plot(longts, N1,'k',linewidth=1,dashes=[2,2],label='no ph.')
# for LL in np.array([L]): #2,3,4,5,6,7,8,9,10
#     try:
#         data=np.load(path+f"/{params.dat_fold}/"+params.label(L=LL)+".npy",allow_pickle=True)
#         dat0 = data[1]
#         if mc == '2':
#             Pn = np.asarray([(o2[0, 0:d] * i)[0, 0] for i in dat0])  # [0,0] just extracts value of matrix element.
#         elif mc == '1':
#             Pn = np.asarray([(o1[0, 0:d] * i)[0, 0] for i in dat0])  # o2/o1 are measuring channel, in 1 or 2.
#         tm = np.asarray(data[0])
#         if correlator=='LP':  
#             Pn=abs(Pn)
#             P2=abs(P2)
#         bb.plot(tm, (Pn),linewidth='1', label=f'L={LL}')
        
#     except:
#         pass
    
        
# # if correlator=='LP':  
#     bb.set_yscale('log')
#     bb.set_ylim(y1, y2)
# bb.set_xlim(0, tf)
# # leg=bb.legend(loc=1)
# # plt.legend(bbox_to_anchor=(1.01,1), loc="upper left")
# # leg.get_frame().set_color('none')
# plt.tight_layout()
# plt.show()
# fig1.savefig(path+f'/plots/'+params.label()+'.pdf', bbox_inches='tight',transparent=True)
# fig1.savefig(path+f'/plots/'+params.label()+'.png', bbox_inches='tight',transparent=True)



