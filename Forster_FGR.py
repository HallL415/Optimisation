#For calculatiing FGR decay rates - Gamma1, Gamma2
import matplotlib.pyplot as plt
import numpy as np
import Parameters as params
from Functions import FreqAsymptot, Jw,  LpopAsymptot, Sanalyt,  forster
from scipy.linalg import expm

hbar= params.hbar
kb=params.kb

T=params.T
l=params.l
Vs=params.Vs
#g=params.g
fact=params.fact
def GammaPh(R0,det,T):
    # print(det,T)
    # print(det*1e3*hbar,T*hbar/kb)
    ##################################
    S=Sanalyt(T,params.j0,params.j0_1,params.w0,R0,Vs)
    g=forster(params.l,params.eps,params.dcv,R0,-2*fact,7*fact,-3*fact,6*fact)
    ge=g*np.exp(-S)
    #print(np.exp(-S))
    R=np.sqrt(det**2+4*ge**2)
    omega=abs(det)#+2*params.omp  #params.Om1+params.Om2
    lambdp=(omega+R)/2
    lambdm=(omega-R)/2
    RF=(lambdp-lambdm)
    Dp=(1/np.sqrt(2))*np.sqrt(1+det/RF)
    Dm=(1/np.sqrt(2))*np.sqrt(1-det/RF)
    ##############################

    j0_FGR=(params.DvDc**2 * (1e3)**2 * (params.vc)**2 * (1e9)**2 * (1e-12)**2) /  (np.pi * params.dens * (1e-3) * (params.vc)**2 * (6.24e21) *(1e-7)**3 * params.Vs**5)
    j0_FGR=j0_FGR/hbar   #ps^2 units

    GamPh=4*Dp**2*Dm**2*( (RF/2)**3 * j0_FGR * np.exp(-(0.5*l**2*(RF)**2)/Vs**2) * (1 - (Vs* np.sin((RF)*R0/Vs))/((RF)*R0)) )*hbar

    return  GamPh, RF, Dp, Dm, lambdp, lambdm

def Gamma1_FGR_det(R0,det,T):
    '''FGR decay rate for downward transitions in meV'''
    '''this vanishes at as T -> 0'''
    RF=GammaPh(R0,det,T)[1]
    BoseDist= (1/(np.exp(RF/T)-1)) 
    return  BoseDist*GammaPh(R0,det,T)[0]

def Gamma2_FGR_det(R0,det,T):
    '''FGR decay rate for upward transitions in meV'''
    RF=GammaPh(R0,det,T)[1]
    BoseDist= (1/(np.exp(RF/T)-1)) 
    return  (1+BoseDist)*GammaPh(R0,det,T)[0]


def GammaPhNrm(R0,det,T):
    ##################################
    S=Sanalyt(T,params.j0,params.j0_1,params.w0,R0,Vs)
    g=forster(params.l,params.eps,params.dcv,R0,-2*fact,7*fact,-3*fact,6*fact)
    ge=g#*np.exp(-S/2)
    #print(np.exp(-S))
    R=np.sqrt(det**2+4*ge**2)
    omega=abs(det)#+2*params.omp  #params.Om1+params.Om2
    lambdp=(omega+R)/2
    lambdm=(omega-R)/2
    RF=(lambdp-lambdm)
    Dp=(1/np.sqrt(2))*np.sqrt(1+det/RF)
    Dm=(1/np.sqrt(2))*np.sqrt(1-det/RF)
    ##############################

    j0_FGR=(params.DvDc**2 * (1e3)**2 * (params.vc)**2 * (1e9)**2 * (1e-12)**2) /  (np.pi * params.dens * (1e-3) * (params.vc)**2 * (6.24e21) *(1e-7)**3 * params.Vs**5)
    j0_FGR=j0_FGR/hbar   #ps^2 units

    GamPh=4*Dp**2*Dm**2*( (RF/2)**3 * j0_FGR * np.exp(-(0.5*l**2*(RF)**2)/Vs**2) * (1 - (Vs* np.sin((RF)*R0/Vs))/((RF)*R0)) )*hbar

    return  GamPh, RF, Dp, Dm, lambdp, lambdm

def Gamma1_FGR_det_nrm(R0,det,T):
    '''FGR decay rate for downward transitions in meV'''
    '''this vanishes at as T -> 0'''
    RF=GammaPhNrm(R0,det,T)[1]
    BoseDist= (1/(np.exp(RF/T)-1)) 
    return  BoseDist*GammaPhNrm(R0,det,T)[0]

def Gamma2_FGR_det_nrm(R0,det,T):
    '''FGR decay rate for upward transitions in meV'''
    RF=GammaPhNrm(R0,det,T)[1]
    BoseDist= (1/(np.exp(RF/T)-1)) 
    return  (1+BoseDist)*GammaPhNrm(R0,det,T)[0]
#print(Gamma1_FGR_det(params.r0,params.det,params.T), Gamma2_FGR_det(params.r0,params.det,params.T))

def N21ampli(R0,det):
    G1=Gamma1_FGR_det(R0,det,T)
    G2=Gamma2_FGR_det(R0,det,T)
    Dp=GammaPh(R0,det,T)[2]
    Dm=GammaPh(R0,det,T)[3]
    AA=np.array([0,(Dm**2-Dp**2)*(G2*Dp**2-G1*Dm**2)/(G1+G2),-Dp*Dm*Dp*Dm,-Dp*Dm*Dp*Dm,(G2*Dp**2+G1*Dm**2)/(G1+G2)])
    #print((AA))
    return AA


from Functions import  forster
# def Pan_NQD2(t,R0,det,T):
#     '''Analytics for N_21'''
#     S=Sanalyt(T,params.j0,params.j0_1,params.w0,R0,Vs)
#     g=forster(params.l,params.eps,params.dcv,R0,-2*fact,7*fact,-3*fact,6*fact)
#     ge=g*np.exp(-S)
#     gamma=0.5*(params.gamma1+params.gamma2 )

#     gam1=Gamma1_FGR_det(R0,det,T)/hbar 
#     gam2=Gamma2_FGR_det(R0,det,T)/hbar 

#     dgam=(gam2-gam1)/(gam2+gam1)
#     RF=GammaPh(R0,det,T)[1]
#     C=(
#         (1/2)*(1+dgam*det/RF)*np.exp(-2*gamma*t)
#         -(det/(2*RF))*(dgam+det/RF)*np.exp(-2*(gamma+gam1+gam2)*t)
#         -(2*ge**2/RF**2)*np.exp(-2*(gamma+(gam1+gam2)/2)*t)*np.cos(RF*t)
#       )
#     if params.mc == "1" and params.ec == "1":
#         C=C11
#     if params.mc == "1" and params.ec == "2":
#         C=C12
#     if params.mc == "2" and params.ec == "1":
#         C=C21        
#     if params.mc == "2" and params.ec == "2":
#         C=C22
#     AA=N21ampli(R0,det)
#     ww=FreqAsymptot(RF,params.gamma1*hbar,params.gamma2*hbar,gam1,gam2)
#     return C.real

def Pan_NQD(t,R0,det,T):
    '''Analytics for N_21'''
    S=Sanalyt(T,params.j0,params.j0_1,params.w0,R0,Vs)
    g=forster(params.l,params.eps,params.dcv,R0,-2*fact,7*fact,-3*fact,6*fact)
    ge=g*np.exp(-S)
    gamma=0.5*(params.gamma1+params.gamma2 )

    gam1=Gamma1_FGR_det(R0,det,T)/hbar 
    gam2=Gamma2_FGR_det(R0,det,T)/hbar 

    dgam=(gam2-gam1)/(gam2+gam1)
    RF=GammaPh(R0,det,T)[1]
    C=(
        (1/2)*(1+dgam*det/RF)*np.exp(-2*gamma*t)
        -(det/(2*RF))*(dgam+det/RF)*np.exp(-2*(gamma+gam1+gam2)*t)
        -(2*ge**2/RF**2)*np.exp(-2*(gamma+(gam1+gam2)/2)*t)*np.cos(RF*t)
      )
    AA=N21ampli(R0,det)
    ww=FreqAsymptot(RF,params.gamma1*hbar,params.gamma2*hbar,gam1,gam2)
    return C.real, AA, ww

def Panalyt2(t,RF,dp,dm):
    """ 'Analytics' - general, very slow(for any measurement or excitation channel of Lpop) but numeric approach"""
    #can also add measurement vectors for Bloch sphere
    #GamPh,RF,dp,dm, lambdp, lambdm=GammaPh(params.r0,params.det)

    Q22= np.matrix([ [0],
                    [dp**2],
                    [dp*dm],
                    [dp*dm],
                    [dm**2]
                        ])
    Q11 = np.matrix([ [0],
                    [dm**2],
                    [-dp*dm],
                    [-dp*dm],
                    [dp**2]
                        ])
    Q21= np.matrix([ [0],
                    [dp*dm],
                    [-dp*dp],
                    [dm*dm],
                    [-dm*dp]
                        ])
    Q12 = np.matrix([ [0],
                    [dm*dp],
                    [dm*dm],
                    [-dp*dp],
                    [-dp*dm]
                        ])
    
    Pu= Q12+Q21
    Pv =1j*(Q12-Q21)
    Pw= Q11-Q22


    if params.ec == "1":
        EC1 = Q11
    if params.ec == "2":
        EC1 = Q22
    if params.mc == "1":
        OC1 = Q11.T
    if params.mc == "2":
        OC1 = Q22.T
    G1=Gamma1_FGR_det(params.r0,params.det,params.T)/hbar 
    G2=Gamma2_FGR_det(params.r0,params.det,params.T)/hbar 
    #LL=LpopAsymptot(RF,0,0,gam1,gam2)
    #M1=expm(-1j*LL*t) #this is quite slow
    #Lambdat=Lambdat(RF,0,0,gam1,gam2)
    gam1=0
    gam2=0
    Gt=G1+G2
    U=np.matrix([[1,0,0,0,1],
                 [0,1,0,0,-G1/Gt],
                 [0,0,1,0,0],
                 [0,0,0,1,0],
                 [0,-1,0,0,-G2/Gt],
                ])


    Um1=np.matrix([[1,1,0,0,1],
                 [0,G2/Gt,0,0,-G1/Gt],
                 [0,0,1,0,0],
                 [0,0,0,1,0],
                 [0,-1,0,0,-1],
                ])

    ww=np.array([0,-2*1j*(gam1+G1+G2),RF-1j*(gam1+gam2+G1+G2),-RF-1j*(gam1+gam2+G1+G2),-2*1j*gam2])
    LL=np.matrix([[np.exp(-1j*ww[0]*t),0,0,0,0],
                 [0,np.exp(-1j*ww[1]*t),0,0,0],
                 [0,0,np.exp(-1j*ww[2]*t),0,0],
                 [0,0,0,np.exp(-1j*ww[3]*t),0],
                 [0,0,0,0,np.exp(-1j*ww[4]*t)],
                ])


    M1=U*LL*Um1


    
    C=(OC1*M1*EC1)[0,0]
    x=(Pu.T*M1*EC1)[0,0]
    y=(Pv.T*M1*EC1)[0,0]
    z=(Pw.T*M1*EC1)[0,0]
    return C,x,y,z


def Pan_LP(t,R0,det):
    '''analytics for P_11'''
    Dp=GammaPh(R0,det,T)[2]
    Dm=GammaPh(R0,det,T)[3]
    lambdp=GammaPh(R0,det,T)[4]
    lambdm=GammaPh(R0,det,T)[5]
    gamma=0.5*(params.gamma1+params.gamma2 )
    w1=-lambdp - 1j*gamma -1j* Gamma2_FGR_det(R0,det,T)/hbar
    w2=-lambdm - 1j*gamma -1j* Gamma1_FGR_det(R0,det,T)/hbar
    Pan_11=np.exp(-params.SHR)*(Dp**2 * np.exp(-1j*w1*t) + Dm**2 * np.exp(-1j*w2*t))
    AA=np.array([np.exp(-params.SHR)*Dp**2,np.exp(-params.SHR)*Dm**2])
    ww=np.array([w1,w2])
    return Pan_11, AA, ww


def asymptotics(R0,det):
    '''Asymptotics'''
    #note for NQD the asymptotic result is for N_21
    gam1=Gamma1_FGR_det(R0,det,T)/hbar 
    gam2=Gamma2_FGR_det(R0,det,T)/hbar 
    if params.correlator=='NQD':
        dgam=(gam2-gam1)/(gam2+gam1)
        RF=GammaPh(R0,det,T)[1]
        C=(1/2)*(1+(det/RF)*dgam)
    if params.correlator=='LP':
        Dp=GammaPh(R0,det,T)[2]
        Dm=GammaPh(R0,det,T)[3]
        C=Dp**2+Dm**2
    Tau=1/(gam2+gam1)
    #print(C,Tau)
    return C.real,Tau.real

from Functions import BN, gxN, gyN, SyN, Jw, Jw2
BN=BN(params.j0,params.w0,params.r0,Vs,T)
def Nazir(t,VF):
    VR=BN*VF
    gam1=VF**2*(2*gxN(0,params.j0,params.w0,T,params.r0,Vs)+gyN(2*VR,params.j0,params.w0,T,params.r0,Vs)*(1+2*(1/(np.exp(2*VR/T)-1)))/(1+(1/(np.exp(2*VR/T)-1))))
    gam2=2*VF*gxN(0,params.j0,params.w0,T,params.r0,Vs)
    lambd=2*VF**2*(SyN(2*VR,params.j0,params.w0,T,params.r0,Vs)-SyN(-2*VR,params.j0,params.w0,T,params.r0,Vs))
    xi=np.sqrt(8*VR*(2*VR+lambd)-(gam1-gam2)**2)
    sigz=np.exp((-gam1+gam2)*t/2)*(np.cos(xi*t/2)+((gam2-gam1)/xi)* np.sin(xi*t/2))
    return sigz



######## plot analytics ############
import os
path=os.getcwd()
param_label=f'{params.correlator}{params.ec}{params.mc}_T{round(params.T*hbar/kb)}_g{round(params.g*hbar*1e3)}_R{round(params.r0)}_l{np.round(l,1)}_det{np.round(params.detuning*hbar*1e3,1)}_gam{round(params.gamma1*hbar*1e3)}_gam{round(params.gamma2*hbar*1e3)}'

# #plot the analytic result
# tan=np.arange(0,250,0.5)
# fig1=plt.figure()
# if params.correlator == 'LP':
#     Pan,AA,ww =Pan_LP(tan,params.r0,params.det)
#     plt.ylabel('$|P(t)|$')
#     plt.yscale('log')
# else:
#     Pan,AA,ww =Pan_NQD(tan,params.r0,params.det,params.T)
#     plt.ylabel('$N(t)$')
# plt.plot(tan,np.abs(Pan),label='analytics')
# plt.xlabel('Time (ps)')
# plt.legend(loc=1)
# plt.show()
# plt.close()
# fig1.savefig(path+f'/plots/analyt_'+param_label+'.pdf', bbox_inches='tight',transparent=True)

# #plot Fermi rates vs r0
# R0=np.linspace(0.01,400,300)
# gam1=[1e3*Gamma1_FGR_det(r,params.det) for r in R0]
# gam2=[1e3*Gamma2_FGR_det(r,params.det) for r in R0]
# fig2 = plt.figure(figsize=(4,3),dpi=200)
# bb = fig2.add_subplot(1, 1, 1)
# bb.plot(R0,gam1,'r', linewidth="1", label=r'$\Gamma_\downarrow$')
# bb.plot(R0,gam2,'b',linewidth="0.8", label=r'$\Gamma_\uparrow$')
# plt.show()
# plt.close()
# fig2.savefig(path+f'/plots/FGR_'+param_label+'.pdf', bbox_inches='tight',transparent=True)

# #plot Fermi rates vs T
# R0=np.linspace(0.01,50,100)
# gam1=np.array([Gamma1_FGR_det(params.r0,params.det,T)/hbar  for T in R0])
# gam2=np.array([Gamma2_FGR_det(params.r0,params.det,T)/hbar  for T in R0])
# tau12=(1/(gam1+gam2)).real
# fig2 = plt.figure(figsize=(4,3),dpi=200)
# bb = fig2.add_subplot(1, 1, 1)
# bb.plot(R0,tau12,'g', linewidth="1", label=r'$\Gamma_\downarrow$')
# #bb.plot(R0,gam1,'r', linewidth="1", label=r'$\Gamma_\downarrow$')
# #bb.plot(R0,gam2,'b',linewidth="0.8", label=r'$\Gamma_\uparrow$')
# plt.show()
# plt.close()
# fig2.savefig(path+f'/plots/FGR-tau_'+param_label+'.pdf', bbox_inches='tight',transparent=True)

# # #plopt the spectral density
# ww=np.linspace(0,10,300)
# J1=Jw(ww,params.j0,params.w0)
# J2=Jw2(ww,params.j0,params.w0,params.r0,params.Vs)
# fig2 = plt.figure(figsize=(4,3),dpi=200)
# bb = fig2.add_subplot(1, 1, 1)
# #bb.plot(ww,J1,'r',linewidth="0.8", label=r'$J(w)$')
# bb.plot(ww,J2,'b',linewidth="0.8", label=r'$J(w)$')
# bb.legend(loc=1)
# plt.show()
# plt.close()
# fig2.savefig(path+f'/plots/Jw_'+param_label+'.pdf', bbox_inches='tight',transparent=True)