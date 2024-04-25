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
from functions_ps import DiagM, Kbb2
import parameters_ps as pp
import time
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm 

wx=pp.wx
gx=pp.gx
gc=pp.gc
g =pp.g
det=pp.det
omp=pp.omp
shr=pp.shr
J0=pp.J0
w0=pp.w0
T=pp.T
L=pp.L
tauib=3.25

if L<27: # original needs an enormous amount of ram to run for L > 27
    import LP_xc_past_future_original as original
    Pnf_orig = original.Pnf
    tpsf_orig = original.tpsf


class Cumulants(object):
    """a set of functions to run numerics"""
    def __init__(self):

        self.dt=None
        self.data=[[],[]]
        self.zerocounter=[]

    def updateK(self, in_K):
        """update the set of cumulants"""
        self.K = in_K
        return self.K
    def Ku(self,t):
        """Short time part of the cumulant"""
        Kinf=-1j*omp*t-shr
        return Kbb2(t, T, J0, w0)+Kinf

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
            cumul=self.K
            return cumul


#%% generalize
def generate_permutations(length):
    for perm in itertools.product([0, 1], repeat=length):
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
    V[0][::2] = M1[0,0]#this layout is assuming the mapping follows 00,01,10,11, i.e. i_2 i_1 so Mi1i0 causes alternating M elements to be placed, rather than half M00 half M10
    V[0][1::2] = M1[1,0]
    ii = 0
    indices = (np.arange(L) + 1).astype('str')[::-1] # first L/2 indices correspond to U and rest to V
    indices2 = indices
    P = []
    S2 = []
    
    permutations = np.ones((n, int(length)), dtype=np.int32)
    generator = generate_permutations(int(length))
    for i in range(n):
        permutations[i, :] = np.array(next(generator))
        
    print(f"\n  Step0: Storing permutations {np.round((time.time() - start_time), 2)} seconds")
    for step in tqdm(range(step_no),position=0, leave=True):
        
        i_values = {}
        i_values_U = {}
        U0 = U.copy()
        U1 = U.copy()
        
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
        value = 0 # start position value for the i=0 row
        value2 = 2**(ii) # start position value for the i=1 row
        fill = 0 # position of the perturbation in the new V array (in terms of p-values)
        while fill + int(2**(ii)) < n: # int(2**(ii)) corresponds to the position of the same perturbation as fill but with a different p-value
            if fill == 0:
                V_perm_order0[fill] = value # effectively saying that at position [fill], the perturbation should be filled in with the perturbation in position [value] of the old array
                V_perm_order0[fill + int(2**(ii))] = value
                
                V_perm_order1[fill] = value2
                V_perm_order1[fill + int(2**(ii))] = value2
                fill += 1
            else:
                
                if fill % (2**(ii)) == 0:
                    fill += 2**(ii)
                else:
                    pass  
                
                if (fill) % (2**(ii)) == 0: # ii gives the number of times the indeces have shifted (happens every time step)
                    value += int(2**(ii) + 1)
                    value2 += int(2**(ii) + 1)
                else:
                    value += 1
                    value2 += 1
                    
                V_perm_order0[fill] = int(value)
                V_perm_order0[fill + int(2**(ii))] = int(value)
                
                V_perm_order1[fill] = int(value2)
                V_perm_order1[fill + int(2**(ii))] = int(value2)
                fill += 1
                
         
            
                

        print(f"\n  Step1: after Arsenis' addition {np.round((time.time() - start_time), 2)} seconds")
 ##########################################################################################################################       
        if cycle == 0:
            indices2 = indices #when new cycle begins reset indices
            if step != 0:
                U0 = V.copy().T
                U1 = V.copy().T
                V = U.T
                
        i_pos = np.ones(len(V_indices))
        i_pos_U = np.ones(len(V_indices))
        
        
        Q0_U=np.ones(n)+0j #We want to generate the appropriate product of Q matrix elements, starting from 1 then multiplying in the loop
        Q1_U=np.ones(n)+0j
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
            
            if V_indices[j] != '1':
                
                # QV *= np.array([Qlist[int(V_indices[j]) - 2][index[i],0] if permutations[i][i1_pos]==0 else Qlist[int(V_indices[j]) - 2][index[i],1] for i in range(n)])
                
                # instead of the if statement above and the loop use positions through True and False values
                true_false_array = permutations[:, i1_pos]==0 # positions where i1 is zero
                QV[true_false_array] *= Qlist[int(V_indices[j]) - 2][np.array(index)[true_false_array],0]
                QV[np.logical_not(true_false_array)] *= Qlist[int(V_indices[j]) - 2][np.array(index)[np.logical_not(true_false_array)],1] # np.logical_not swaps true and false

            i_values[f'{V_indices[j]}'] = permutations[:, int(i_pos[j])]
            i_values_U[f'{U_indices[j]}'] = permutations[:, int(i_pos_U[j])]
        
        print(f"Step2.0: Calculating Qs {np.round((time.time() - start_time), 2)} seconds")
            
        for i in range(U0.shape[-1]):
            U0[:, i] *= Q0_U # multiplying the product of Qs into Us elements appropriately
            U1[:, i] *= Q1_U
           
            V[i, :] *= QV  ###change
            
        V_p0 = V.copy()
        V_p1 = V.copy()
        
        print(f"Step2.1: Applying Qs to Us {np.round((time.time() - start_time), 2)} seconds")
        for i in range(n):
            if i_values['1'][i] == 0:
                V_p0[:, i] *= Qlist[-1][0,0]
                V_p1[:, i] *= Qlist[-1][1,0]
            else:
                V_p0[:, i] *= Qlist[-1][0,1]
                V_p1[:, i] *= Qlist[-1][1,1]
        
        print(f"\n Step2: Applying Qs to U and V: {np.round((time.time() - start_time), 2)} seconds")
        #reconstruct V to have i1=0 as the first row and i1=1 for the next, p varies across columns
        
        V_i0 = np.reshape(np.ones(sum(len(row) for row in V_p0)), np.shape(V_p0)) + 0j
        V_i1 = np.reshape(np.ones(sum(len(row) for row in V_p0)), np.shape(V_p0)) + 0j
        # print(f"Step3: {np.round((time.time() - start_time), 2)} seconds")
        
        
        for i in range(n):
            
            if permutations[i, :][i1_pos] == 0: # effectively if p == 0 for this permutation
                V_i0[:, i] = V_p0[:, int(V_perm_order0[i])]
                V_i1[:, i] = V_p0[:, int(V_perm_order1[i])]
            else:
                V_i0[:, i] = V_p1[:, int(V_perm_order0[i])]
                V_i1[:, i] = V_p1[:, int(V_perm_order1[i])]
                
        # print(f"Step4: {np.round((time.time() - start_time), 2)} seconds")                     
        V = np.vstack((V_i0, V_i1))
        U = np.hstack((U0,U1))
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
                    V_val =  np.where(np.array(i_values[f'{i_index}'])[V_val2] == 1)[0]
                    V_val2 = V_val2[V_val]
            V_column = V[:, V_val2]        
            
        if len(i2_pos_U) == 0:
            U_row = U[-1, :] # choose the (1, 1) column which is the final one
        else:
            U_val = np.where(np.array(i_values_U['2']) == 0)[0]
            U_val2 = U_val
            for i_index in U_indices:
                if i_index != '2':
                    U_val =  np.where(np.array(i_values_U[f'{i_index}'])[U_val2] == 1)[0]
                    U_val2 = U_val2[U_val]
            U_row = U[U_val2, :]  
        print(f"\n Step3 Finding which indices to extract for calc: {np.round((time.time() - start_time), 2)} seconds")
        
            ############################## Removing only contributions below a threshold of the maximum S value every timestep.
        print('\n V rows before truncation:', V.shape[0])
        A, S, Vt2 = np.linalg.svd(V, full_matrices=False)
        threshold = S[0] * 1e-4
        thresh = np.where(S > threshold)[0]
        S= S[thresh]
        # print('S size:',np.shape(S))
        S2.append(S)
        A= A[:, thresh]
        Vt2 = Vt2[thresh, :]
        S=np.diag(S)
        print('\n V rows after truncation:', Vt2.shape[0])

        # print('S shape:',np.shape(S))
        # print('A shape:',np.shape(A))
        # print(' \n U original shape:',np.shape(U))

        U= U @ A @ S
        V= Vt2  
        # print('entered SVD part')
        P3= np.abs(np.exp(cumulants[0])*(np.dot( (U)[-1,:], V[:,V_val2])))
        print(f"\n Step4 SVD application: {np.round((time.time() - start_time), 2)} seconds")

            
        
        # print(f"Step {step+1}: {np.round((time.time() - start_time), 2)} seconds")
        P.append(np.abs(np.exp(cumulants[0])*(np.dot(U_row,V_column))))
        
    print(f" \n For L = {L} and {step_no} steps, code took {np.round((time.time() - start_time)/60, 2)} minutes to run or {np.round((time.time() - start_time),2)} seconds ")   
    # P_final = np.abs(np.exp(cumulants[0])*(np.dot(U_row,V_column)))

    # Vtilde0_p0,Vtilde0_p1= Vtilde0*Qlist[-1][0,0],Vtilde0*Qlist[-1][1,0],Vtilde1*Qlist[-1][0,0],Vtilde1*Qlist[-1][1,0]
    return  U, V, np.array(P), P3, S2

# L=6

tauib=3.25
dt=1.2*tauib/(L+1)
D, ww, U1, V1 = DiagM(g, wx, gx, gc, det, omp)
DD = np.diag(np.exp(-1j*ww*dt))
M1 = U1*DD*V1

cumulants=Cumulants().cu(L,dt)

Q0=np.array([[M1[0,0]*np.exp(cumulants[0] +2*cumulants[1]), M1[0,1] ],[M1[1,0]*np.exp(cumulants[0]), M1[1,1]]])
Qlist=[]
Qlist.append(Q0)
for i in range(int(L-1)):
    Qlist.append(np.array([[np.exp(2*cumulants[i+2]), 1 ],[1, 1]]))


tfinal=15
step_no = int(tfinal/dt)
# step_no = 1
length = L/2
n=int(2**((L/2)))
U, V, P, P3, S2 = get_nth_permutation()


times = np.array([dt*i for i in range(step_no+2)])


# print(f'The full calculation for L={L} for the given tstep is:\n', np.abs(Pnf[step_no+1]))
# print(f'The calculation for L={L} for the given tstep is:\n', P[-1])
# print(f'The SVD calculation for L={L} for the given tstep is:\n', np.abs(Pnf_orig[step_no+1]))



fig1 = plt.figure(10, figsize=(6,6),dpi=150)
# bb.set_title(fr'L={L}')
# plt.plot(tpsf_orig, abs(Pnf_orig), 'b', linewidth='1', label=f'original, L={L}')
plt.plot(times[2:], P, 'r--', linewidth='1', label=f'SVD, L={L}')

plt.legend(loc='best') 
plt.show()

# for i in range(0,len(S2)):
#     fig1 = plt.figure(11,figsize=(6,6))
#     plt.plot(S2[i], 'b')
#     plt.yscale("log")
#     plt.legend(loc='best') 





