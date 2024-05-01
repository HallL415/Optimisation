# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:31:43 2024

@author: c1528041
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:09:00 2024

@author: Luke-Arsenis

This version of the code saves certain arrays such one for the permutations and some for the Qs
This makes the code more memory intensive but in turn imrpoves its speed
Permutations are stored in the form of a matrix instead of a list to further improve computation time.
Additionally less memory is required to save them in such as form (roughly 15% of previous space)
"""
import numpy as np
import itertools
import QDQDCAV_parameters as params
from QDQDCAV_functions import LFpol, S_inin, S_inim, PolaronShift, PolaronShift_inim, phi_inin, phi_inim, K11_smartie,K12_smartie
from scipy.linalg import expm
import time
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm 

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
g1 = params.g1
g2 = params.g2
gd=params.gd
j0 = params.j0
j0_1 = params.j0_1
w0 = params.w0
T = params.T
r0 = params.r0
l = params.l
lbar=params.lbar
lp=params.lp
Vs = params.Vs
sharebath=params.sharebath
L = params.L



if dotshape=='spherical':
    tauib=np.sqrt(2)*np.pi*l/Vs
if dotshape=='smartie':
    tauib=np.sqrt(2)*np.pi *lbar/Vs


if L<15: # original needs an enormous amount of ram to run for L > 27
    import QD_QD_CAV as original
    Pnf_orig = original.Pn
    tpsf_orig = original.tm


class Cumulants(object):
    """a set of functions to run numerics"""
    def __init__(self):

        self.dt=None
        self.data=[[],[]]
        self.zerocounter=[]

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



    def cu(self, L, dt):
        """Calculates a set of square cumulants for L neighbours and updates an array"""
        if self.dt != dt:
            K0 = self.Ku(dt)[0]  # K(dt) i.e. K_inin0
            K0_inim = self.Ku(dt)[1]
            cumulants = []
            cumulantsinin = []
            cumulants.append(K0_inim)
            cumulantsinin.append(K0)
            while len(cumulants) < L + 1:
                self.Kn(cumulants, cumulantsinin,
                        dt)  # calculate K^(n) for Kinin or Kinim i.e. K_11^(1), K_22(2) or K_12^(1), K_12^(2) etc.
                # for however many L
                cumulants.append(self.Knn)
                cumulantsinin.append(self.Knn_inin)
            self.updateK(in_K=cumulants, in_Kinin=cumulantsinin)
            self.dt = dt
            cumul=self.K
            cumulinin=self.Kinin
            return cumul, cumulinin
        
        
#%% generalize
def generate_permutations(length):
    for perm in itertools.product([0, 1, 2], repeat=length):
        yield perm

# Usage example
def create_ones_column(n):
    # Create an array of shape (n, 1) filled with ones
    return np.ones((n, 1))
def create_ones_row(n):
    # Create an array of shape (n, 1) filled with ones
    return np.ones((1, n))

def rotate(l, n):
     return l[n:] + l[:n]#


# step_no = 2
def get_nth_permutation():
    start_time = time.time() # time at which function is called
    # permutation_generator = generate_permutations(int(length))  # Ensure length is converted to an integer
    U = create_ones_column(n) + 0j
    
    V = create_ones_row(n) + 0j
    V[0][::3] = M1[0,0]#this layout is assuming the mapping follows 00,01,10,11, i.e. i_2 i_1 so Mi1i0 causes alternating M elements to be placed, rather than half M00 half M10
    V[0][1::3] = M1[1,0]
    V[0][2::3] = M1[2,0]
    ii = 0
    indices = (np.arange(L) + 1).astype('str')[::-1] # first L/2 indices correspond to U and rest to V
    indices2 = indices
    P = []
    S2 = []
    
    permutations = np.ones((n, int(length)), dtype=np.int32)
    generator = generate_permutations(int(length))
    for i in range(n):
        permutations[i, :] = np.array(next(generator))
        
    # print(f"\n  Step0: Storing permutations {np.round((time.time() - start_time), 2)} seconds")
    for step in tqdm(range(step_no),position=0, leave=True):
        
        i_values = {}
        i_values_U = {}
        U0 = U.copy()
        U1 = U.copy()
        U2 = U.copy()

        
        cycle = ii % (L/2)
        if ii == 0:
            indices2 = indices
        else: 
            indices2 = np.roll(indices2, -1)
        
        U_indices = indices2[:int(L/2)]
        V_indices = indices2[int(L/2):]
        i1_pos = np.where(V_indices == '1')[0][0] # location of i1 in permutation order
        

################################################# new addition
        V_perm_order0 = np.ones(int(n))
        V_perm_order1= np.ones(int(n))
        V_perm_order2 = np.ones(int(n))

        value = 0 # start position value for the i1=0 row
        value2 = 3**(ii) # start position value for the i1=1 row
        value3 = 2*3**(ii) # start position value for the i1=2 row

        fill = 0 # position of the perturbation in the new V array (in terms of p-values)
        while fill + int(2 * 3**(ii)) < n: # int(2**(ii)) corresponds to the position of the same perturbation as fill but with a different p-value
            if fill == 0:
                V_perm_order0[fill] = value # effectively saying that at position [fill], the perturbation should be filled in with the perturbation in position [value] of the old array
                V_perm_order0[fill + int(3**(ii))] = value
                V_perm_order0[fill + int(2 * 3**(ii))] = value
                
                V_perm_order1[fill] = value2
                V_perm_order1[fill + int(3**(ii))] = value2
                V_perm_order1[fill + int(2* 3**(ii))] = value2
                
                V_perm_order2[fill] = value3
                V_perm_order2[fill + int(3**(ii))] = value3
                V_perm_order2[fill + int(2* 3**(ii))] = value3
                
                fill += 1
            else:
                
                if fill % (3**(ii)) == 0: # ii gives the number of times the indeces have shifted (happens every time step)
                    fill += 2 * 3**(ii)
                    
                    value += int(2* 3**(ii) + 1)
                    value2 += int(2* 3**(ii) + 1)
                    value3 += int(2* 3**(ii) + 1)
                else:
                    value += 1
                    value2 += 1
                    value3 += 1
                    
                V_perm_order0[fill] = int(value)
                V_perm_order0[fill + int(3**(ii))] = int(value)
                V_perm_order0[fill + int(2 * 3**(ii))] = int(value)
                
                V_perm_order1[fill] = int(value2)
                V_perm_order1[fill + int(3**(ii))] = int(value2)
                V_perm_order1[fill + int(2 * 3**(ii))] = int(value2)
                
                V_perm_order2[fill] = int(value3)
                V_perm_order2[fill + int(3**(ii))] = int(value3)
                V_perm_order2[fill + int(2 * 3**(ii))] = int(value3)
                
                fill += 1
                
         
            
                

        # print(f"\n  Step1: after Arsenis' addition {np.round((time.time() - start_time), 2)} seconds")
 ##########################################################################################################################       
        if cycle == 0:
            indices2 = indices #when new cycle begins reset indices
            if step != 0:
                U0 = V.copy().T
                U1 = V.copy().T
                U2 = V.copy().T
                V = U.T
                
        i_pos = np.ones(len(V_indices))
        i_pos_U = np.ones(len(V_indices))
        
        # print(np.max(U0), np.max(U1), np.max(U2))
        Q0_U=np.ones(n)+0j #We want to generate the appropriate product of Q matrix elements, starting from 1 then multiplying in the loop
        Q1_U=np.ones(n)+0j
        Q2_U=np.ones(n)+0j
        QV = np.ones(n)+0j     
        for j in range(len(V_indices)):
            i_values[f'{V_indices[j]}'] = []
            i_values_U[f'{U_indices[j]}'] = []
            
            i_pos[j] = np.where(V_indices == f'{V_indices[j]}')[0][0]
            i_pos_U[j] = np.where(U_indices == f'{U_indices[j]}')[0][0]
            
            # index = [permutation[j] for permutation in permutations]
            index = list(permutations[:, j])

            
            Q0_U*=Qlist[int(L)-j-2-ii][index,0] #Qlist contains matrices ([[e^2K_r, 1],[ 1, 1]]) and K_r depends on the timestep i_n, so i6i5i4 should use the latter 3 elements of Qlist
            Q1_U*=Qlist[int(L)-j-2-ii][index,1] #see notes why splitting into 0 and 1
            Q2_U*=Qlist[int(L)-j-2-ii][index,2]
            
            if V_indices[j] != '1':
                
                # QV *= np.array([Qlist[int(V_indices[j]) - 2][index[i],0] if permutations[i][i1_pos]==0 else Qlist[int(V_indices[j]) - 2][index[i],1] for i in range(n)])
                
                # instead of the if statement above and the loop use positions through True and False values
                true_false_array_0 = permutations[:, i1_pos]==0 # positions where i1 is zero
                true_false_array_1 = permutations[:, i1_pos]==1 # positions where i1 is one
                true_false_array_2 = permutations[:, i1_pos]==2 # positions where i1 is two
                QV[true_false_array_0] *= Qlist[int(V_indices[j]) - 2][np.array(index)[true_false_array_0],0]
                QV[true_false_array_1] *= Qlist[int(V_indices[j]) - 2][np.array(index)[true_false_array_1],1]
                QV[true_false_array_2] *= Qlist[int(V_indices[j]) - 2][np.array(index)[true_false_array_2],2]

            i_values[f'{V_indices[j]}'] = permutations[:, int(i_pos[j])]
            i_values_U[f'{U_indices[j]}'] = permutations[:, int(i_pos_U[j])]
        
        # print(f"Step2.0: Calculating Qs {np.round((time.time() - start_time), 2)} seconds")
            
        for i in range(U0.shape[-1]):
            U0[:, i] *= Q0_U # multiplying the product of Qs into Us elements appropriately
            U1[:, i] *= Q1_U
            U2[:, i] *= Q2_U

           
            V[i, :] *= QV  ###change
        Q0_U=[]
        Q1_U=[]
        Q2_U=[]

        V_p0 = V.copy()
        V_p1 = V.copy()
        V_p2 = V.copy()

        U = np.hstack((U0,U1,U2))
        U0=[]
        U1=[]
        U2=[]

        # print(f"Step2.1: Applying Qs to Us {np.round((time.time() - start_time), 2)} seconds")
        for i in range(n):
            if i_values['1'][i] == 0:
                V_p0[:, i] *= Qlist[-1][0,0]
                V_p1[:, i] *= Qlist[-1][1,0]
                V_p2[:, i] *= Qlist[-1][2,0]

            if i_values['1'][i] == 1:
                V_p0[:, i] *= Qlist[-1][0,1]
                V_p1[:, i] *= Qlist[-1][1,1]
                V_p2[:, i] *= Qlist[-1][2,1]

            if i_values['1'][i] == 2:
                V_p0[:, i] *= Qlist[-1][0,2]
                V_p1[:, i] *= Qlist[-1][1,2]
                V_p2[:, i] *= Qlist[-1][2,2]

        
        # print(f"\n Step2: Applying Qps to U and V: {np.round((time.time() - start_time), 2)} seconds")
        #reconstruct V to have i1=0 as the first row and i1=1 for the next, p varies across columns
        
        V_i0 = np.reshape(np.ones(sum(len(row) for row in V_p0)), np.shape(V_p0)) + 0j
        V_i1 = np.reshape(np.ones(sum(len(row) for row in V_p0)), np.shape(V_p0)) + 0j
        V_i2 = np.reshape(np.ones(sum(len(row) for row in V_p0)), np.shape(V_p0)) + 0j

        # print(f"Step3: {np.round((time.time() - start_time), 2)} seconds")
        
        
        for i in range(n):
            
            if permutations[i, :][i1_pos] == 0: # effectively if p == 0 for this permutation
                V_i0[:, i] = V_p0[:, int(V_perm_order0[i])]
                V_i1[:, i] = V_p0[:, int(V_perm_order1[i])]
                V_i2[:, i] = V_p0[:, int(V_perm_order2[i])]
            elif permutations[i, :][i1_pos] == 1:
                V_i0[:, i] = V_p1[:, int(V_perm_order0[i])]
                V_i1[:, i] = V_p1[:, int(V_perm_order1[i])]
                V_i2[:, i] = V_p1[:, int(V_perm_order2[i])]
            else:
                V_i0[:, i] = V_p2[:, int(V_perm_order0[i])]
                V_i1[:, i] = V_p2[:, int(V_perm_order1[i])]
                V_i2[:, i] = V_p2[:, int(V_perm_order2[i])]    
        
        # print(f"Step4: {np.round((time.time() - start_time), 2)} seconds")                     
        V = np.vstack((V_i0, V_i1, V_i2))
       
        
        V_i0=[]
        V_i1=[]
        V_i2=[]
        V_p0=[]
        V_p1=[]
        V_p2=[]
        # print(f'for {step+1} the shape of V is:\n', np.shape(V))
        ii += 1
        # print('ii', ii)
        if ii % (L/2) == 0:
            ii = 0
            
        i2_pos_V = np.where(V_indices == '2')[0]
        i2_pos_U = np.where(U_indices == '2')[0]
     

        if len(i2_pos_V) == 0:
            V_column = V[:,-1] # choose the (1, 1) column which is the final one
        else:
            V_val = np.where(np.array(i_values['2']) == 0)[0]
            V_val2 = V_val
            for i_index in V_indices:
                if i_index != '2':
                    V_val =  np.where(np.array(i_values[f'{i_index}'])[V_val2] == 2)[0]
                    V_val2 = V_val2[V_val]
            V_column = V[:, V_val2]        
            
        if len(i2_pos_U) == 0:
            U_row = U[-1, :] # choose the (1, 1) column which is the final one
        else:
            U_val = np.where(np.array(i_values_U['2']) == 0)[0]
            U_val2 = U_val
            for i_index in U_indices:
                if i_index != '2':
                    U_val =  np.where(np.array(i_values_U[f'{i_index}'])[U_val2] == 2)[0]
                    U_val2 = U_val2[U_val]
            U_row = U[U_val2, :]  
        print(f"\n Step3 Finding which indices to extract for calc: {np.round((time.time() - start_time), 2)} seconds")
        
            ############################# Removing only contributions below a threshold of the maximum S value every timestep.
        print('\n V rows before truncation:', V.shape[0])
        A, S, V = np.linalg.svd(V, full_matrices=False)
        threshold = S[0] * threshold_factor
        thresh = np.where(S > threshold)[0]
        S= S[thresh]
        # print('S size:',np.shape(S))
        S2.append(S)
        A= A[:, thresh]
        V = V[thresh, :]    
        S=np.diag(S)
        print('\n V rows after truncation:', V.shape[0])
        U= U @ A @ S
        # extra SVD application on U to try and shorten further if V doesn't truncate much
        if SVD_Both==1 and len(np.diag(S))> 8 :
            print('\n U columns before truncation:', U.shape[1])
            U, S, B = np.linalg.svd(U, full_matrices=False)
            threshold = S[0] * threshold_factor
            thresh = np.where(S > threshold)[0]
            S= S[thresh]
            B=B[thresh,:]
            U=U[:,thresh]
            S=S[thresh]
            S=np.diag(S)
            V=S @ B @ V
            print('\n U columns after truncation:', U.shape[1])

        print('entered SVD part')
        # P3= np.abs(np.exp(cumulants[0])*(np.dot( (U)[-1,:], V[:,V_val2])))
        # print(f"\n Step4 SVD application: {np.round((time.time() - start_time), 2)} seconds")

            
        
        # print(f"Step {step+1}: {np.round((time.time() - start_time), 2)} seconds")
        P.append(np.abs(np.exp(cumulants_inin[0])*(np.dot(U_row,V_column))))
        
    print(f" \n For L = {L} and {step_no} steps, code took {np.round((time.time() - start_time)/60, 2)} minutes to run or {np.round((time.time() - start_time),2)} seconds ")   
    # P_final = np.abs(np.exp(cumulants[0])*(np.dot(U_row,V_column)))

    # Vtilde0_p0,Vtilde0_p1= Vtilde0*Qlist[-1][0,0],Vtilde0*Qlist[-1][1,0],Vtilde1*Qlist[-1][0,0],Vtilde1*Qlist[-1][1,0]
    return  U, V, np.array(P), S2

# L=6

if dotshape=='spherical':
    tauib=np.sqrt(2)*np.pi*l/Vs
if dotshape=='smartie':
    tauib=np.sqrt(2)*np.pi *lbar/Vs
t0=r0/Vs
if sharebath ==1:
    dt=( t0 + 1.2*tauib )/(L+1)
if sharebath ==0:
    dt=1.2*tauib / (L+1)

LF=LFpol(g1, g2,gd, w_qd1, w_qd2, w_c)
M1=expm(-1j*LF*dt)
cumulants,cumulants_inin=Cumulants().cu(L,dt)


Q0=np.array([[M1[0,0]*np.exp(cumulants_inin[0] +2*cumulants_inin[1]), M1[0,1]*np.exp(cumulants_inin[0] +2*cumulants[1]), M1[0,2] ],
             [M1[1,0]*np.exp(cumulants_inin[0] +2*cumulants[1]), M1[1,1]*np.exp(cumulants_inin[0] +2*cumulants_inin[1]), M1[1,2] ],
             [M1[2,0]*np.exp(cumulants[0]), M1[2,1]*np.exp(cumulants[0]), M1[2,2]]])
Qlist=[]
Qlist.append(Q0)
for i in range(int(L-1)):
    Qlist.append(np.array([[np.exp(2*cumulants_inin[i+2]), np.exp(2*cumulants[i+2]), 1 ],
                           [np.exp(2*cumulants[i+2]), np.exp(2*cumulants_inin[i+2]), 1],
                           [1, 1, 1]]))


tfinal=10
step_no = int(tfinal/dt)
# step_no = 1
length = L/2
n=int(3**((L/2)))
threshold_factor=1e-5
SVD_Both=1

U, V, P, S2 = get_nth_permutation()

times = np.array([dt*i for i in range(step_no+2)])
P=np.ravel(P)

###################### Plotting #####################
fig1 = plt.figure( figsize=(6,6),dpi=150)
# bb.set_title(fr'L={L}')
# plt.plot(tpsf_orig, abs(Pnf_orig), 'b', linewidth='1', label=f'original')
plt.plot(tm, abs(Pn), 'b', linewidth='1', label=f'original')
plt.plot(times[2:], P, 'r--', linewidth='1', label=f'SVD, L={L}')

plt.legend(loc='best') 
plt.show()



########### Calculate error ############
def calculate_rmse(y_true, y_pred):
    n = len(y_pred)
    squared_errors = (np.abs(y_true[2:(len(y_pred)+2)]) - y_pred) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse
if L<27:
    error=calculate_rmse(Pnf_orig, P)
    print(f'The RMSE for a threshold value of {threshold_factor} is {error:.2e}')


# for i in range(0,len(S2)):
#     fig1 = plt.figure(11,figsize=(6,6))
#     plt.plot(S2[i], 'b')
#     plt.yscale("log")
#     plt.legend(loc='best') 
# totalS2s=[]
# for i in range(len(S2)):
#     totalS2s.append(len(S2[i]))
# total=np.sum(totalS2s)
# print('total number of S values',total)
