# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:09:00 2024

@author: Luke
"""
import numpy as np
import itertools
from functions_ps import DiagM, Kbb2
import parameters_ps as pp
import LP_xc_past_future_original as original
import time
import matplotlib.pyplot as plt
import sys

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
    
    
    for step in range(step_no):
        
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
        
        # permutations = ['' for i in np.zeros(2**int(length))]
        # permutations0 = ['' for i in np.zeros(2**int(length))]
        # permutations1 = ['' for i in np.zeros(2**int(length))]
        # generator = generate_permutations(int(length))
        # for i in range(n):
        #     permutation = list(next(generator))
        #     # permutation[i1_pos] = 0
        #     # permutations0.append(tuple(permutation)) # list of tuples for the permutations of the i = 0 row
        #     # permutation[i1_pos] = 1
        #     # permutations1.append(tuple(permutation)) # list of tuples for the permutations of the i = 1 row
        #     for ss, sss in enumerate(permutation):
        #         permutations[i] = permutations[i] + f"{sss}"
        #         if ss == i1_pos: # at the position of i1 set 1 or 0 depending on the row that will be constructed
        #             permutations0[i] = permutations0[i] + "0" 
        #             permutations1[i] = permutations1[i] + "1"
                    
        #         else: # the remaining values of the perturbation remain the same (only i is swapped with p)
        #             permutations0[i] = permutations0[i] + f"{sss}"
        #             permutations1[i] = permutations1[i] + f"{sss}"
        
        # V_perm_order0 = [] # order of permutations after swapping i1 and p (effectively a mapping between the (i3, i2, i1) and (i3, i2, p) notation)
        # V_perm_order1 = []
        # for i in range(n):
        #     # order in V matrices for each of the i=0 and i=1 rows eg. the zeroth element of the i=0 V matrix corresponds to Vp1[V_perm_order[0]]
        #     V_perm_order0.append(np.where(np.array(permutations) == permutations0[i])[0][0]) #can include this in above loop so as to not store permutation0 and 1
        #     V_perm_order1.append(np.where(np.array(permutations) == permutations1[i])[0][0])
        
        permutations = ['' for i in np.zeros(2**int(length))]
        V_perm_order0 = []
        V_perm_order1 = []
        generator = generate_permutations(int(length))
        for i in range(n):
            permutation = list(next(generator))
            for ss, sss in enumerate(permutation):
                permutations[i] = permutations[i] + f"{sss}"

        for i in range(n):
            permutations0 = ''
            permutations1 = ''
            permutation = permutations[i]
            for ss, sss in enumerate(permutation):
                if ss == i1_pos: # at the position of i1 set 1 or 0 depending on the row that will be constructed
                    permutations0 = permutations0 + "0" 
                    permutations1 = permutations1 + "1"
                    
                else: # the remaining values of the perturbation remain the same (only i is swapped with p)
                    permutations0 = permutations0 + f"{sss}"
                    permutations1 = permutations1 + f"{sss}"
            # order in V matrices for each of the i=0 and i=1 rows eg. the zeroth element of the i=0 V matrix corresponds to Vp1[V_perm_order[0]]
            V_perm_order0.append(np.where(np.array(permutations) == permutations0)[0][0]) #can include this in above loop so as to not store permutation0 and 1
            V_perm_order1.append(np.where(np.array(permutations) == permutations1)[0][0])
        print(f"Step1: {np.round((time.time() - start_time), 2)} seconds")
        # sys.getsizeof(permutations0) # check size of permutation lists
        '''
        In the above we're creating a list of permutations for each of the i columns. For example for the i = 0 and (i3, i2, p) = (0, 0, 0) element
        one has to simply extract the (i3, i2, i1) = (0, 0, 0) of the Vp0 matrix
        '''
        
        
        generator = generate_permutations(int(length))  # Ensure length is converted to an integer
        if cycle == 0:
            indices2 = indices #when new cycle begins reset indices
            if step != 0:
                U0 = V.copy().T
                U1 = V.copy().T
                V = U.T
        for j in range(len(V_indices)):
            i_values[f'{V_indices[j]}'] = []
            i_values_U[f'{U_indices[j]}'] = []
        for i in range(n): #n is how many permutations we have in total for a given L.
            permutation=next(generator) #generates the possible combinations for a given L, L=4 -> 00,01,10,11, L=4 -> 000,001,010,011,100,101,110,111
            # print(permutation) #checking
            Q0_U=1 #We want to generate the appropriate product of Q matrix elements, starting from 1 then multiplying in the loop
            Q1_U=1
            QV=1

            for j in range(len(V_indices)):                        
                i_pos = np.where(V_indices == f'{V_indices[j]}')[0][0]
                i_values[f'{V_indices[j]}'].append(permutation[i_pos])
                
                i_pos_U = np.where(U_indices == f'{U_indices[j]}')[0][0]
                i_values_U[f'{U_indices[j]}'].append(permutation[i_pos_U])
                
            

            for j in range(int(L/2)): #This is how many elements are in a specific permutation, L=6 -> 000,001,...  L/2 elements
                #looping to select each element of e.g (i6i5i4 -> 001 -> check if it's 0 or 1 at each position)   
                index=permutation[j]
                Q0_U=Q0_U*Qlist[int(L)-j-2-ii][index,0] #Qlist contains matrices ([[e^2K_r, 1],[ 1, 1]]) and K_r depends on the timestep i_n, so i6i5i4 should use the latter 3 elements of Qlist
                Q1_U=Q1_U*Qlist[int(L)-j-2-ii][index,1] #see notes why splitting into 0 and 1
                i1_pos = np.where(V_indices == '1')[0][0]
                if V_indices[j] != '1': #because we multiply less Qs onto V than U (excluding p index)
                    if permutation[i1_pos]==0:
                        QV=QV*Qlist[int(V_indices[j]) - 2][index,0] #same done for V, except V takes Q_iL/2 l ...Qi2l as the Qs to multiply in
                        # print(QV)
                    else:
                        QV=QV*Qlist[int(V_indices[j]) - 2][index,1]
                        # print(QV)
                
            U0[i, :]=Q0_U*U0[i, :] # multiplying the product of Qs into Us elements appropriately
            U1[i, :]=Q1_U*U1[i, :]
           
            V[:, i]=QV*V[:, i]  ###change
            
        V_p0 = V.copy()
        V_p1 = V.copy()
        
        print(f"Step1: {np.round((time.time() - start_time), 2)} seconds")
        for i in range(n):
            if i_values['1'][i] == 0:
                V_p0[:, i] *= Qlist[-1][0,0]
                V_p1[:, i] *= Qlist[-1][1,0]
            else:
                V_p0[:, i] *= Qlist[-1][0,1]
                V_p1[:, i] *= Qlist[-1][1,1]
        
        print(f"Step2: {np.round((time.time() - start_time), 2)} seconds")
        #reconstruct V to have i1=0 as the first row and i1=1 for the next, p varies across columns
        
        V_i0 = np.reshape(np.ones(sum(len(row) for row in V_p0)), np.shape(V_p0)) + 0j
        V_i1 = np.reshape(np.ones(sum(len(row) for row in V_p0)), np.shape(V_p0)) + 0j
        print(f"Step3: {np.round((time.time() - start_time), 2)} seconds")
        
        
        # start_time = time.time()
        generator = generate_permutations(int(length))
        for i in range(n):
            permutation=next(generator)
            # print(permutation)
            
        #     luffy = 0
        #     for i_index in V_indices:
        #         i_pos = np.where(V_indices == i_index)[0][0] #location if i_index in the permutation
        #         if i_index != '1': # if i index is not i1
        #             if luffy == 0: # first itteration (find where desired value of current i index is located in the permutation)
        #                 '''
        #                 We know the permutations that the i=0 (and i=1) row has. Hence we know the values of i3 and i2 (if L = 4).
        #                 Eg when constructing the i = 0 row first element will have i3 = 0 and i2 = 0. We essentially locate the i3=0, i2=0, i1=0 element
        #                 from the original V matrix and assign it to the new matrix
        #                 permutation[i_pos]: locations in the permutations corresponding to the desired i index (single value)
        #                 np.array(i_values[f'{i_index}'] are the values of i index across all permutations
        #                 '''
        #                 val3 = np.where(np.array(i_values[f'{i_index}']) == permutation[i_pos])[0]
        #                 val4 = np.where(np.array(i_values[f'{i_index}']) == permutation[i_pos])[0]
        #                 val3_pos, val4_pos = val3, val4
        #             else: # for following itterations only consider positions 
        #                 val3 = np.where(np.array(i_values[f'{i_index}'])[val3] == permutation[i_pos])[0]
        #                 val4 = np.where(np.array(i_values[f'{i_index}'])[val4] == permutation[i_pos])[0]
        #                 val3_pos, val4_pos = val3_pos[val3], val4_pos[val4]
        #         else:
        #             # Find locations in the permutation where i1 is 0 or 1 (to create the i1 = 0 and i1 = 1 rows)
        #             if luffy == 0:
        #                 val3 = np.where(np.array(i_values[f'{i_index}']) == 0)[0]
        #                 val4 = np.where(np.array(i_values[f'{i_index}']) == 1)[0]
        #                 val3_pos, val4_pos = val3, val4
        #             else:
        #                 val3 = np.where(np.array(i_values[f'{i_index}'])[val3] == 0)[0]
        #                 val4 = np.where(np.array(i_values[f'{i_index}'])[val4] == 1)[0]
        #                 val3_pos, val4_pos = val3_pos[val3], val4_pos[val4]
        #         luffy += 1
        #     val3 = val3_pos[0]
        #     val4 = val4_pos[0]
        #     # print(val3, val4)
                
        # # print(f"Step4: {np.round((time.time() - start_time), 2)} seconds")  
        #     if i_values['1'][i] == 0: #p has the same value as i1 for the given permutation arrangement
        #         V_i0[:, i] = V_p0[:, val3]
        #         V_i1[:, i] = V_p0[:, val4]
        #     else:
        #         V_i0[:, i] = V_p1[:, val3]
        #         V_i1[:, i] = V_p1[:, val4]
                
            if permutation[i1_pos] == 0: # effectively if p == 0 for this permutation
                V_i0[:, i] = V_p0[:, V_perm_order0[i]]
                V_i1[:, i] = V_p0[:, V_perm_order1[i]]
            else:
                V_i0[:, i] = V_p1[:, V_perm_order0[i]]
                V_i1[:, i] = V_p1[:, V_perm_order1[i]]
                
        print(f"Step4: {np.round((time.time() - start_time), 2)} seconds")                     
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
            
        # if (step+1) % 10 ==0:
        if V.shape[0] == 64:
            A, S, Vt2 = np.linalg.svd(V, full_matrices=False)
            k=int(V.shape[0]/2)
            # k = 8
            S2.append(S)
            A = A[:, :k]
            S = np.diag(S[:k])
            Vt2 = Vt2[:k, :]
        
            # print('U original shape:',np.shape(U))
            # print('V original shape:', np.shape(V))
            
            # print('A shape:',np.shape(A))
            # print('S shape:', np.shape(S))
            # print('Vt2 shape:', np.shape(Vt2))
            # Vnew2=np.dot(A, np.dot(S, Vt2))
            # print('Vnew2 shape:', np.shape(Vnew2))
            
            # P3= np.abs(np.exp(cumulants[0])*(np.dot(U[-1, :], Vnew2[:, V_val2])))
            U= U @ A @ S
            V= Vt2  
            print('entered SVD part')
            P3= np.abs(np.exp(cumulants[0])*(np.dot( (U)[-1,:], V[:,V_val2])))
            
        
        print(f"Step {step+1}: {np.round((time.time() - start_time), 2)} seconds")
        P.append(np.abs(np.exp(cumulants[0])*(np.dot(U_row,V_column))))
        
    print(f"For L = {L} and {step_no} steps, code took {np.round((time.time() - start_time)/60, 2)} minutes to run")   
    # P_final = np.abs(np.exp(cumulants[0])*(np.dot(U_row,V_column)))

########## SVD reducing contributions ##############

    #### Thinking perhaps when no. of rows in V reaches 1024 which is then dimension 1024 x 2^(L/2), 
    #   after 10 time steps, we then apply SVD on V, reduce to let's say 8 rows
    #multiply the U of 2^(L/2) x 1024 by  1024 x 8 matrix A, and 8x8 matrix lambda, giving U of dimension 2^(L/2) by 8. 
    #continue the procedure as usual, until dimensions reach 1024 again, reduce back to 8, repeat.
    #Note that using 1024 is effectively waiting for 2^10 elements more, equivalent to L=10, to reach larger L we need to SVD more frequently 
    #at the cost of speed, implemenet this as a parameter.


    # print('Working full calc. with no SVD: \n',P )
    # print('calc attempting reduced SVD:\n',P3 )

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



step_no = 50
length = L/2
n=int(2**((L/2)))
U, V, P, P3, S2 = get_nth_permutation()


times = np.array([dt*i for i in range(step_no+2)])
# times[0] = 
# times[1] =

# P_max=np.abs(np.exp(cumulants[0])*(np.dot(U[-2, : ],V[:,-1])))

# print(f'The full calculation for L={L} for the given tstep is:\n', np.abs(Pnf[step_no+1]))
print(f'The calculation for L={L} for the given tstep is:\n', P[-1])
print(f'The SVD calculation for L={L} for the given tstep is:\n', np.abs(Pnf_orig[step_no+1]))



fig1 = plt.figure(10, figsize=(6,6),dpi=150)
# bb.set_title(fr'L={L}')
plt.plot(tpsf_orig, abs(Pnf_orig), 'b', linewidth='1', label=f'original, L={L}')
plt.plot(times[2:], P, 'r--', linewidth='1', label=f'SVD, L={L}')

plt.legend(loc='best') 


# for i in range(10, 20):
#     fig1 = plt.figure(figsize=(6,6))
#     plt.plot(S2[i], 'b')
#     # plt.yscale("log")
#     plt.legend(loc='best') 



























# By adding (''') commend out all of the below sections
'''
#%%
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

length = L/2
permutation_generator = generate_permutations(int(length))  # Ensure length is converted to an integer
n=int(2**((L/2)))

def get_nth_permutation(generator):
    length = L/2
    permutation_generator = generate_permutations(int(length))  # Ensure length is converted to an integer
    n=int(2**((L/2)))
    #Creating starting arrays U,V for the first step, U is filled with 1s, V contains M_00 or M_10.
    Utilde0 = create_ones_column(n)
    Utilde1 = create_ones_column(n)
    Utilde0=np.array(Utilde0,dtype=np.complex128)
    Utilde1=np.array(Utilde1,dtype=np.complex128)

    Vtilde0 = create_ones_row(n)
    Vtilde0=np.array(Vtilde0,dtype=np.complex128)
    Vtilde0[0][::2] = M1[0,0]#this layout is assuming the mapping follows 00,01,10,11, i.e. i_2 i_1 so Mi1i0 causes alternating M elements to be placed, rather than half M00 half M10
    Vtilde0[0][1::2] = M1[1,0]  
    # Vtilde1 = create_ones_row(n)
    # Vtilde1=np.array(Vtilde1,dtype=np.complex128)
    # Vtilde1[0][::2] = M1[0,0]
    # Vtilde1[0][1::2] = M1[1,0]  #this layout is assuming the mapping follows 00,01,10,11, i.e. i_2 i_1 so Mi1i0 causes alternating M elements to be placed, rather than half M00 half M10
    for i in range(n): #n is how many permutations we have in total for a given L.
        permutation=next(permutation_generator) #generates the possible combinations for a given L, L=4 -> 00,01,10,11, L=4 -> 000,001,010,011,100,101,110,111
        print(permutation) #checking
        # Qstring0=[]
        # Qstring1=[]
        Q0_U=1 #We want to generate the appropriate product of Q matrix elements, starting from 1 then multiplying in the loop
        Q1_U=1
        Q0_V=1
        Q1_V=1
        QV=1
        for j in range(int(L/2)): #This is how many elements are in a specific permutation, L=6 -> 000,001,...  L/2 elements
            #looping to select each element of e.g (i6i5i4 -> 001 -> check if it's 0 or 1 at each position)   
            index=permutation[j]
            # Qstring0.append(Qlist[L-j-1][index,0])
            Q0_U=Q0_U*Qlist[int(L)-j-2][index,0] #Qlist contains matrices ([[e^2K_r, 1],[ 1, 1]]) and K_r depends on the timestep i_n, so i6i5i4 should use the latter 3 elements of Qlist
            Q1_U=Q1_U*Qlist[int(L)-j-2][index,1] #see notes why splitting into 0 and 1
            if j < ( int(L/2)-1 ): #because we multiply less Qs onto V than U (excluding p index)
                if i%2==0:
                    QV=QV*Qlist[int(L/2) - j -2 ][index,0] #same done for V, except V takes Q_iL/2 l ...Qi2l as the Qs to multiply in
                    print(QV)
                if i%2 != 0:
                    QV=QV*Qlist[int(L/2) - j -2 ][index,1]
                    print(QV)
            # Qstring1.append(Qlist[L-j-1][index,1])
        Utilde0[i][0]=Q0_U*Utilde0[i][0] # multiplying the product of Qs into Us elements appropriately
        Utilde1[i][0]=Q1_U*Utilde1[i][0]
        
        Vtilde0[0][i]=QV*Vtilde0[0][i]  ###change
        # Vtilde1[i][0]=Q1_V*V[i][0]
   
    
    V_p0 = Vtilde0.copy()
    V_p1 = Vtilde0.copy()
    even_indices = np.arange(V_p0.shape[1]) % 2 == 0
    V_p0[0,even_indices] *= Qlist[-1][0,0]
    V_p0[0,~even_indices] *= Qlist[-1][0,1]
    print('Vp0:', V_p0)
   
    V_p1[0,even_indices] *= Qlist[-1][1,0]
    V_p1[0,~even_indices] *= Qlist[-1][1,1]
    print('Vp0:', V_p0)
    print('Vp1:', V_p1)

    # Vtilde0_p0,Vtilde0_p1= Vtilde0*Qlist[-1][0,0],Vtilde0*Qlist[-1][1,0],Vtilde1*Qlist[-1][0,0],Vtilde1*Qlist[-1][1,0]
    return Utilde0, Utilde1,V_p0,V_p1

Utilde0,Utilde1,V_p0,V_p1=get_nth_permutation(permutation_generator)

P2nd=np.abs(np.exp(cumulants[0])*(Utilde0[-1]*V_p1[0][0] + Utilde1[-1]* V_p1[0][1]))
print('Pnf', np.abs(Pnf[2]))
Pnf2=np.abs(Pnf)


length = L/2
permutation_generator = generate_permutations(int(length))  # Ensure length is converted to an integer
n=int(2**((L/2)))

def get_nth_permutation(generator):
    length = L/2
    permutation_generator = generate_permutations(int(length))  # Ensure length is converted to an integer
    n=int(2**((L/2)))
    #Creating starting arrays U,V for the first step, U is filled with 1s, V contains M_00 or M_10.
    Utilde0 = create_ones_column(n)
    Utilde1 = create_ones_column(n)
    Utilde0=np.array(Utilde0,dtype=np.complex128)
    Utilde1=np.array(Utilde1,dtype=np.complex128)

    Vtilde0 = create_ones_row(n)
    Vtilde0=np.array(Vtilde0,dtype=np.complex128)
    Vtilde0[0][::2] = M1[0,0]#this layout is assuming the mapping follows 00,01,10,11, i.e. i_2 i_1 so Mi1i0 causes alternating M elements to be placed, rather than half M00 half M10
    Vtilde0[0][1::2] = M1[1,0]
    # Vtilde1 = create_ones_row(n)
    # Vtilde1=np.array(Vtilde1,dtype=np.complex128)
    # Vtilde1[0][::2] = M1[0,0]
    # Vtilde1[0][1::2] = M1[1,0]  #this layout is assuming the mapping follows 00,01,10,11, i.e. i_2 i_1 so Mi1i0 causes alternating M elements to be placed, rather than half M00 half M10
    for i in range(n): #n is how many permutations we have in total for a given L.
        permutation=next(permutation_generator) #generates the possible combinations for a given L, L=4 -> 00,01,10,11, L=4 -> 000,001,010,011,100,101,110,111
        print(permutation) #checking
        # Qstring0=[]
        # Qstring1=[]
        Q0_U=1 #We want to generate the appropriate product of Q matrix elements, starting from 1 then multiplying in the loop
        Q1_U=1
        Q0_V=1
        Q1_V=1
        QV=1
        for j in range(int(L/2)): #This is how many elements are in a specific permutation, L=6 -> 000,001,...  L/2 elements
            #looping to select each element of e.g (i6i5i4 -> 001 -> check if it's 0 or 1 at each position)   
            index=permutation[j]
            # Qstring0.append(Qlist[L-j-1][index,0])
            Q0_U=Q0_U*Qlist[int(L)-j-2][index,0] #Qlist contains matrices ([[e^2K_r, 1],[ 1, 1]]) and K_r depends on the timestep i_n, so i6i5i4 should use the latter 3 elements of Qlist
            Q1_U=Q1_U*Qlist[int(L)-j-2][index,1] #see notes why splitting into 0 and 1
            if j < ( int(L/2)-1 ): #because we multiply less Qs onto V than U (excluding p index)
                if i%2==0:
                    QV=QV*Qlist[int(L/2) - j -2 ][index,0] #same done for V, except V takes Q_iL/2 l ...Qi2l as the Qs to multiply in
                    print(QV)
                if i%2 != 0:
                    QV=QV*Qlist[int(L/2) - j -2 ][index,1]
                    print(QV)
            # Qstring1.append(Qlist[L-j-1][index,1])
        Utilde0[i][0]=Q0_U*Utilde0[i][0] # multiplying the product of Qs into Us elements appropriately
        Utilde1[i][0]=Q1_U*Utilde1[i][0]
        
        Vtilde0[0][i]=QV*Vtilde0[0][i]  ###change
        # Vtilde1[i][0]=Q1_V*V[i][0]
   
    
    
   
    V_p0 = Vtilde0.copy()
    V_p1 = Vtilde0.copy()

    even_indices = np.arange(V_p0.shape[1]) % 2 == 0
    V_p0[0,even_indices] *= Qlist[-1][0,0]
    V_p0[0,~even_indices] *= Qlist[-1][0,1]
    print('Vp0:', V_p0)
   
    V_p1[0,even_indices] *= Qlist[-1][1,0]
    V_p1[0,~even_indices] *= Qlist[-1][1,1]
    print('Vp0:', V_p0)
    print('Vp1:', V_p1)
    #reconstruct V to have i1=0 as the first row and i1=1 for the next, p varies across columns
    V_i0 = np.hstack((V_p0[:, [0, 2]], V_p1[:, [0, 2]]))
    # V_i0 = V_i0.reshape(1, 4)
    # V_i0[:, [1, 2]] = V_i0[:, [2, 1]]
    
    V_i1 = np.hstack((V_p0[:, [1, 3]], V_p1[:, [1, 3]]))
    # V_i1 = V_i1.reshape(1, 4)
    # V_i1[:, [1, 2]] = V_i1[:, [2, 1]]
    V_combined = np.vstack((V_i0, V_i1))
    U_combined= np.hstack((Utilde0,Utilde1))
    
    
    # Vtilde0_p0,Vtilde0_p1= Vtilde0*Qlist[-1][0,0],Vtilde0*Qlist[-1][1,0],Vtilde1*Qlist[-1][0,0],Vtilde1*Qlist[-1][1,0]
    return  U_combined, V_combined

U_combined, V_combined=get_nth_permutation(permutation_generator)

P2nd=np.abs(np.exp(cumulants[0])*(np.dot(U_combined[-1, : ],V_combined[:,2])))
print('Pnf', np.abs(Pnf[2]))
print(np.abs(P2nd))
Pnf2=np.abs(Pnf)



length = L/2
permutation_generator = generate_permutations(int(length))  # Ensure length is converted to an integer
n=int(2**((L/2)))  
U0=U_combined.copy()
U1=U_combined.copy()

V_combined[:, [1, 2]] = V_combined[:, [2, 1]]
V0=V_combined.copy()
V1=V_combined.copy()
def get_nth_permutation(generator):
  
    
    for i in range(n): #n is how many permutations we have in total for a given L.
        permutation=next(permutation_generator) #generates the possible combinations for a given L, L=4 -> 00,01,10,11, L=4 -> 000,001,010,011,100,101,110,111
        # print(permutation) #checking
        
        Q0_U=1 #We want to generate the appropriate product of Q matrix elements, starting from 1 then multiplying in the loop
        Q1_U=1
        QV=1
        for j in range(int(L/2)): #This is how many elements are in a specific permutation, L=6 -> 000,001,...  L/2 elements
            #looping to select each element of e.g (i6i5i4 -> 001 -> check if it's 0 or 1 at each position)   
            index=permutation[j]
            # Qstring0.append(Qlist[L-j-1][index,0])
            Q0_U=Q0_U*Qlist[int(L)-j-3][index,0] #Qlist contains matrices ([[e^2K_r, 1],[ 1, 1]]) and K_r depends on the timestep i_n, so i6i5i4 should use the latter 3 elements of Qlist
            Q1_U=Q1_U*Qlist[int(L)-j-3][index,1] #see notes why splitting into 0 and 1
         

            if j ==1 : #because we multiply less Qs onto V than U (excluding p index)
                if i<2:
                    QV=QV*Qlist[int(L/2)   ][index,0] #same done for V, except V takes Q_iL/2 l ...Qi2l as the Qs to multiply in
                    print('QV even:\n', QV)
                else:
                    QV=QV*Qlist[int(L/2)  ][index,1]
                    print('QV odd:\n',QV)
            # Qstring1.append(Qlist[L-j-1][index,1])
        print('Q0u:\n',Q0_U)
        print('Q1u: \n ',Q1_U)
        U0[i, :]=Q0_U* U0[i, :] # multiplying the product of Qs into Us elements appropriately
        U1[i, :]=Q1_U*U1[i, :]
        U_combined= np.hstack((U0,U1))
       
        

        V0[:,i]=QV*V0[:,i]
        V1[:,i]=QV*V1[:,i]
        
    V0[:, [1, 2]] = V0[:, [2, 1]]
    V1[:, [1, 2]] = V1[:, [2, 1]]
        # V0[:, [1, 2]] = V0[:, [2, 1]]
        # V1[:, [1, 2]] = V1[:, [2, 1]]

    
    even_indices = np.arange(V0.shape[1]) % 2 == 0
    V0[:,even_indices] *= Qlist[-1][0,0]
    V0[:,~even_indices] *= Qlist[-1][0,1]  # V for p=0
    V1[:,even_indices] *= Qlist[-1][1,0]
    V1[:,~even_indices] *= Qlist[-1][1,1] #V for p=1
   
    # print('Vp0:', V_p0)
    row1 = np.hstack((V0[0,(0,2)], V1[0,(0,2)]))
    row2 = np.hstack((V0[1,(0,2)], V1[1,(0,2)]))
    row3 = np.hstack((V0[0,(1,3)], V1[0,(1,3)]))
    row4 = np.hstack((V0[1,(1,3)], V1[1,(1,3)]))
    
    V_combined = np.vstack((row1, row2, row3, row4))
    # V_p1[0,even_indices] *= Qlist[-1][1,0]
    # V_p1[0,~even_indices] *= Qlist[-1][1,1]
    # print('Vp0:', V_p0)
    # print('Vp1:', V_p1)

    # Vtilde0_p0,Vtilde0_p1= Vtilde0*Qlist[-1][0,0],Vtilde0*Qlist[-1][1,0],Vtilde1*Qlist[-1][0,0],Vtilde1*Qlist[-1][1,0]
    return  U_combined, V_combined 

U_combined, V_combined =get_nth_permutation(permutation_generator)

P3rd=np.abs(np.exp(cumulants[0])*(np.dot(U_combined[-2, : ],V_combined[:,-1])))
print(np.abs(Pnf[3]))
print(P3rd)




Vcorrect=V_p0
Vcorrect1=V_p1
#%% generalize

def rotate(l, n):
     return l[n:] + l[:n]#
#%% weird stuff
length = L/2

n=int(2**((L/2)))

step_no = 2
def get_nth_permutation(generator):
    permutation_generator = generate_permutations(int(length))  # Ensure length is converted to an integer
    U = create_ones_column(n) + 0j

    V = create_ones_row(n) + 0j
    V[0][::2] = M1[0,0]#this layout is assuming the mapping follows 00,01,10,11, i.e. i_2 i_1 so Mi1i0 causes alternating M elements to be placed, rather than half M00 half M10
    V[0][1::2] = M1[1,0]
    ii = 0
    indices = (np.arange(L) + 1).astype('str')[::-1] # first L/2 indices correspond to U and rest to V
    indices2 = indices
    permutation_full = []
    for step in range(step_no):
        i_values = {}
        i_values_U = {}
        U0 = U.copy()
        U1 = U.copy()
        
        cycle = ii % (L/2)
        if ii == 0:
            indices2 = indices
        else: 
            indices2 = np.roll(indices2, -1)
        
        generator = generate_permutations(int(length))  # Ensure length is converted to an integer
        U_indices = indices2[:int(L/2)]
        V_indices = indices2[int(L/2):]
        
        if cycle == 0:
            indices2 = indices #when new cycle begins reset indices
            if step != 0:
                U0 = V.copy().T
                U1 = V.copy().T
                V = U.T
        for j in range(len(V_indices)):
            i_values[f'{V_indices[j]}'] = []
            i_values_U[f'{U_indices[j]}'] = []
        for i in range(n): #n is how many permutations we have in total for a given L.
            permutation=next(generator) #generates the possible combinations for a given L, L=4 -> 00,01,10,11, L=4 -> 000,001,010,011,100,101,110,111
            # print(permutation) #checking
            Q0_U=1 #We want to generate the appropriate product of Q matrix elements, starting from 1 then multiplying in the loop
            Q1_U=1
            QV=1
            for j in range(len(V_indices)):                        
                i_pos = np.where(V_indices == f'{V_indices[j]}')[0][0]
                i_values[f'{V_indices[j]}'].append(permutation[i_pos])
                
                i_pos_U = np.where(U_indices == f'{U_indices[j]}')[0][0]
                i_values_U[f'{U_indices[j]}'].append(permutation[i_pos_U])

            for j in range(int(L/2)): #This is how many elements are in a specific permutation, L=6 -> 000,001,...  L/2 elements
                #looping to select each element of e.g (i6i5i4 -> 001 -> check if it's 0 or 1 at each position)   
                index=permutation[j]
                Q0_U=Q0_U*Qlist[int(L)-j-2-ii][index,0] #Qlist contains matrices ([[e^2K_r, 1],[ 1, 1]]) and K_r depends on the timestep i_n, so i6i5i4 should use the latter 3 elements of Qlist
                Q1_U=Q1_U*Qlist[int(L)-j-2-ii][index,1] #see notes why splitting into 0 and 1
                i1_pos = np.where(V_indices == '1')[0][0]
                if V_indices[j] != '1': #because we multiply less Qs onto V than U (excluding p index)
                    if permutation[i1_pos]==0:
                        QV=QV*Qlist[int(V_indices[j]) - 2][index,0] #same done for V, except V takes Q_iL/2 l ...Qi2l as the Qs to multiply in
                        # print(QV)
                    else:
                        QV=QV*Qlist[int(V_indices[j]) - 2][index,1]
                        # print(QV)
                    
            U0[i, :]=Q0_U*U0[i, :] # multiplying the product of Qs into Us elements appropriately
            U1[i, :]=Q1_U*U1[i, :]
           
            V[:, i]=QV*V[:, i]  ###change
            
        V_p0 = V.copy()
        V_p1 = V.copy()

        for i in range(n):
            if i_values['1'][i] == 0:
                V_p0[:, i] *= Qlist[-1][0,0]
                V_p1[:, i] *= Qlist[-1][1,0]
            else:
                V_p0[:, i] *= Qlist[-1][0,1]
                V_p1[:, i] *= Qlist[-1][1,1]
  
        #reconstruct V to have i1=0 as the first row and i1=1 for the next, p varies across columns
        
        V_i0 = np.reshape(np.ones(sum(len(row) for row in V_p0)), np.shape(V_p0)) + 0j
        V_i1 = np.reshape(np.ones(sum(len(row) for row in V_p0)), np.shape(V_p0)) + 0j
        
        generator = generate_permutations(int(length))
        for i in range(n):
            permutation=next(generator)
            # print(permutation)
            
            luffy = 0
            for i_index in V_indices:
                i_pos = np.where(V_indices == i_index)[0][0]
                if i_index != '1':
                    if luffy == 0:
                        val3 = np.where(np.array(i_values[f'{i_index}']) == permutation[i_pos])[0]
                        val4 = np.where(np.array(i_values[f'{i_index}']) == permutation[i_pos])[0]
                        val3_pos, val4_pos = val3, val4
                    else:
                        val3 = np.where(np.array(i_values[f'{i_index}'])[val3] == permutation[i_pos])[0]
                        val4 = np.where(np.array(i_values[f'{i_index}'])[val4] == permutation[i_pos])[0]
                        val3_pos, val4_pos = val3_pos[val3], val4_pos[val4]
                else:
                    if luffy == 0:
                        val3 = np.where(np.array(i_values[f'{i_index}']) == 0)[0]
                        val4 = np.where(np.array(i_values[f'{i_index}']) == 1)[0]
                        val3_pos, val4_pos = val3, val4
                    else:
                        val3 = np.where(np.array(i_values[f'{i_index}'])[val3] == 0)[0]
                        val4 = np.where(np.array(i_values[f'{i_index}'])[val4] == 1)[0]
                        val3_pos, val4_pos = val3_pos[val3], val4_pos[val4]
                luffy += 1
            val3 = val3_pos[0]
            val4 = val4_pos[0]
            # print(val3, val4)
                
            if i_values['1'][i] == 0: #p has the same value as i1 for the given permutation arrangement
                V_i0[:, i] = V_p0[:, val3]
                V_i1[:, i] = V_p0[:, val4]
            else:
                V_i0[:, i] = V_p1[:, val3]
                V_i1[:, i] = V_p1[:, val4]
                
                                
        V = np.vstack((V_i0, V_i1))
        U = np.hstack((U0,U1))

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
                    V_val =  np.where(np.array(i_values[f'{i_index}'])[V_val] == 1)[0]
                    V_val2 = V_val2[V_val]
            V_column = V[:, V_val2]        
            
        if len(i2_pos_U) == 0:
            U_row = U[-1, :] # choose the (1, 1) column which is the final one
        else:
            U_val = np.where(np.array(i_values_U['2']) == 0)[0]
            U_val2 = U_val
            for i_index in U_indices:
                if i_index != '2':
                    U_val =  np.where(np.array(i_values_U[f'{i_index}'])[U_val] == 1)[0]
                    U_val2 = U_val2[U_val]
            U_row = U[U_val2, :]
            
    # print('U_row:', U_row, 'V_column:', V_column)
    P = np.abs(np.exp(cumulants[0])*(np.dot(U_row,V_column)))
    # Vtilde0_p0,Vtilde0_p1= Vtilde0*Qlist[-1][0,0],Vtilde0*Qlist[-1][1,0],Vtilde1*Qlist[-1][0,0],Vtilde1*Qlist[-1][1,0]
    return  U, V, P

step_no = 1
L = 6
length = L/2
n=int(2**((L/2)))
U, V, P = get_nth_permutation(permutation_generator)

# P_max=np.abs(np.exp(cumulants[0])*(np.dot(U[-2, : ],V[:,-1])))

print(np.abs(Pnf[step_no+1]))
print(P)


permutation_generator = generate_permutations(int(length))  # Ensure length is converted to an integer
n=int(2**((L/2)))  
U0=U_combined.copy()
U1=U_combined.copy()

V_combined[:, [1, 2]] = V_combined[:, [2, 1]]
V0=V_combined.copy()
V1=V_combined.copy()
def get_nth_permutation(generator):
  
    
    for i in range(n): #n is how many permutations we have in total for a given L.
        permutation=next(permutation_generator) #generates the possible combinations for a given L, L=4 -> 00,01,10,11, L=4 -> 000,001,010,011,100,101,110,111
        # print(permutation) #checking
        
        Q0_U=1 #We want to generate the appropriate product of Q matrix elements, starting from 1 then multiplying in the loop
        Q1_U=1
        QV=1
        for j in range(int(L/2)): #This is how many elements are in a specific permutation, L=6 -> 000,001,...  L/2 elements
            #looping to select each element of e.g (i6i5i4 -> 001 -> check if it's 0 or 1 at each position)   
            index=permutation[j]
            # Qstring0.append(Qlist[L-j-1][index,0])
            Q0_U=Q0_U*Qlist[int(L)-j-3][index,0] #Qlist contains matrices ([[e^2K_r, 1],[ 1, 1]]) and K_r depends on the timestep i_n, so i6i5i4 should use the latter 3 elements of Qlist
            Q1_U=Q1_U*Qlist[int(L)-j-3][index,1] #see notes why splitting into 0 and 1
         

            if j ==1 : #because we multiply less Qs onto V than U (excluding p index)
                if i<2:
                    QV=QV*Qlist[int(L/2)   ][index,0] #same done for V, except V takes Q_iL/2 l ...Qi2l as the Qs to multiply in
                    print('QV even:\n', QV)
                else:
                    QV=QV*Qlist[int(L/2)  ][index,1]
                    print('QV odd:\n',QV)
            # Qstring1.append(Qlist[L-j-1][index,1])
        print('Q0u:\n',Q0_U)
        print('Q1u: \n ',Q1_U)
        U0[i, :]=Q0_U* U0[i, :] # multiplying the product of Qs into Us elements appropriately
        U1[i, :]=Q1_U*U1[i, :]
        U_combined= np.hstack((U0,U1))
       
        

        V0[:,i]=QV*V0[:,i]
        V1[:,i]=QV*V1[:,i]
        
    V0[:, [1, 2]] = V0[:, [2, 1]]
    V1[:, [1, 2]] = V1[:, [2, 1]]
        # V0[:, [1, 2]] = V0[:, [2, 1]]
        # V1[:, [1, 2]] = V1[:, [2, 1]]

    
    even_indices = np.arange(V0.shape[1]) % 2 == 0
    V0[:,even_indices] *= Qlist[-1][0,0]
    V0[:,~even_indices] *= Qlist[-1][0,1]  # V for p=0
    V1[:,even_indices] *= Qlist[-1][1,0]
    V1[:,~even_indices] *= Qlist[-1][1,1] #V for p=1
   
    # print('Vp0:', V_p0)
    row1 = np.hstack((V0[0,(0,2)], V1[0,(0,2)]))
    row2 = np.hstack((V0[1,(0,2)], V1[1,(0,2)]))
    row3 = np.hstack((V0[0,(1,3)], V1[0,(1,3)]))
    row4 = np.hstack((V0[1,(1,3)], V1[1,(1,3)]))
    
    V_combined = np.vstack((row1, row2, row3, row4))
    # V_p1[0,even_indices] *= Qlist[-1][1,0]
    # V_p1[0,~even_indices] *= Qlist[-1][1,1]
    # print('Vp0:', V_p0)
    # print('Vp1:', V_p1)

    # Vtilde0_p0,Vtilde0_p1= Vtilde0*Qlist[-1][0,0],Vtilde0*Qlist[-1][1,0],Vtilde1*Qlist[-1][0,0],Vtilde1*Qlist[-1][1,0]
    return  U_combined, V_combined 

U_combined, V_combined =get_nth_permutation(permutation_generator)

P3rd=np.abs(np.exp(cumulants[0])*(np.dot(U_combined[-2, : ],V_combined[:,-1])))
print(np.abs(Pnf[3]))
print(P3rd)
















length = L/2
permutation_generator = generate_permutations(int(length))  # Ensure length is converted to an integer
n=int(2**((L/2)))
def get_nth_permutation(generator):
    V_p00 = V_p0.copy()
    V_p01 = V_p0.copy()
    V_p10 = V_p1.copy()
    V_p11 = V_p1.copy()

  
    for i in range(n): #n is how many permutations we have in total for a given L.
        permutation=next(generator) #generates the possible combinations for a given L, L=4 -> 00,01,10,11, L=4 -> 000,001,010,011,100,101,110,111
        # print(permutation) #checking

        Q0_U=1 #We want to generate the appropriate product of Q matrix elements, starting from 1 then multiplying in the loop
        Q1_U=1
        QV=1
        for j in range(int(L/2)): #This is how many elements are in a specific permutation, L=6 -> 000,001,...  L/2 elements
            #looping to select each element of e.g (i6i5i4 -> 001 -> check if it's 0 or 1 at each position)   
            index=permutation[j]
            # Qstring0.append(Qlist[L-j-1][index,0])
            Q0_U=Q0_U*Qlist[int(L)-j-2][index,0] #Qlist contains matrices ([[e^2K_r, 1],[ 1, 1]]) and K_r depends on the timestep i_n, so i6i5i4 should use the latter 3 elements of Qlist
            Q1_U=Q1_U*Qlist[int(L)-j-2][index,1] #see notes why splitting into 0 and 1
         

            if j < ( int(L/2)-1 ): #because we multiply less Qs onto V than U (excluding p index)
                if i%2==0:
                    QV=QV*Qlist[int(L/2) - j -2 ][index,0] #same done for V, except V takes Q_iL/2 l ...Qi2l as the Qs to multiply in
                    print('QV even:\n', QV)
                if i%2 != 0:
                    QV=QV*Qlist[int(L/2) - j -2 ][index,1]
                    print('QV odd:\n',QV)
            # Qstring1.append(Qlist[L-j-1][index,1])
        print('Q0u:\n',Q0_U)
        print('Q1u: \n ',Q1_U)
        Utilde00[i][0]=Q0_U*Utilde0[i][0] # multiplying the product of Qs into Us elements appropriately
        Utilde01[i][0]=Q1_U*Utilde0[i][0] 
        Utilde10[i][0]=Q0_U*Utilde1[i][0] 
        Utilde11[i][0]=Q1_U*Utilde1[i][0] 

        V_p00[0][i]=QV*V_p00[0][i]
        V_p01[0][i]=QV*V_p01[0][i]
        V_p10[0][i]=QV*V_p10[0][i]
        V_p11[0][i]=QV*V_p11[0][i]
       ###change
        # Vtilde1[i][0]=Q1_V*V[i][0]
   
    
        # Vtilde0[0][i]=QV*Vtilde0[0][i]  ###change

    
    
    
    even_indices = np.arange(V_p00.shape[1]) % 2 == 0
    V_p00[0,even_indices] *= Qlist[-1][0,0]
    V_p00[0,~even_indices] *= Qlist[-1][0,1]
    V_p01[0,even_indices] *= Qlist[-1][1,0]
    V_p01[0,~even_indices] *= Qlist[-1][1,1]
    V_p10[0,even_indices] *= Qlist[-1][0,0]
    V_p10[0,~even_indices] *= Qlist[-1][0,1]
    V_p11[0,even_indices] *= Qlist[-1][1,0]
    V_p11[0,~even_indices] *= Qlist[-1][1,1]
   
    # print('Vp0:', V_p0)
   
    # V_p1[0,even_indices] *= Qlist[-1][1,0]
    # V_p1[0,~even_indices] *= Qlist[-1][1,1]
    # print('Vp0:', V_p0)
    # print('Vp1:', V_p1)

    # Vtilde0_p0,Vtilde0_p1= Vtilde0*Qlist[-1][0,0],Vtilde0*Qlist[-1][1,0],Vtilde1*Qlist[-1][0,0],Vtilde1*Qlist[-1][1,0]
    return Utilde00, Utilde01,Utilde10, Utilde11,V_p00,V_p01,V_p10,V_p11

Utilde00, Utilde01,Utilde10, Utilde11,V_p00,V_p01,V_p10,V_p11=get_nth_permutation(permutation_generator)



P3rd=np.abs(np.exp(cumulants[0])*(Utilde00[-1]*Utilde01[-1]*V_p01[0][0]*V_p01[0][1] +Utilde10[-1]*Utilde11[-1]*V_p01[0][1]*V_p11[0][1]))
P3rd=np.abs(np.exp(cumulants[0])*(Utilde00[-1]*V_p01[0][0] +Utilde01[-1]*V_p01[0][1]+ Utilde10[-1]* V_p11[0][0] + Utilde11[-1]* V_p11[0][1]))

P3rd_2=np.abs(np.exp(cumulants[0])*(Utilde00[-1]*V_p01[0][0] +Utilde00[-1]*V_p11[0][0]+ Utilde11[-1]* V_p01[0][1] + Utilde11[-1]* V_p11[0][1]))

print(P3rd)
print(P3rd_2)
print(np.abs(Pnf[3]))
#%%
'''



























