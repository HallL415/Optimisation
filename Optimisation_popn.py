
import numpy as np
import itertools
from scipy.linalg import expm
import Parameters as params
from Functions import LFpop, LFpol, forster, LFpol_qdqdcav, DiagM_qdcav, S_inin, S_inim, PolaronShift, PolaronShift_inim, phi_inin, phi_inim, K11_smartie, K12_smartie, Kbb2, analytics_bareg, analytics_modified, QDQD_analytics_smartie, FGR_smartie, FGR_spherical, FGR_qdcav_spherical, FGR_qdcav_spherical_det
from Forster_FGR import Gamma1_FGR_det, Gamma2_FGR_det, N21ampli, GammaPh, Gamma1_FGR_det_nrm, Gamma2_FGR_det_nrm, Pan_NQD, Panalyt2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm 
import os
path=os.getcwd()

################ importing parameters for QD-QD-Cavity system from QDQDCAV_paramaters file #################
d = params.d
tfinal=params.tf
fit=0
# Choose threshold factor, removes elements less than S_max * threshold_factor from the diagonal matrix S after applying SVD = U S V, truncating U,S,V
threshold_factor=params.threshold_factor
threshold_str = str(threshold_factor)
factortau=params.factortau
### system parameters ###
cavity=params.cavity
no_of_QDs=params.no_of_QDs
dotshape=params.dotshape
correlator=params.correlator
ec=params.ec
mc=params.mc
tf = params.tf
tauib = params.tauib
dt0 = params.dt1  # long time timestep
hbar = params.hbar
kb = params.kb
w_qd1 = params.w_qd1
detuning=params.detuning
w_qd2 = params.w_qd2
if cavity==1:
    gd=params.gd
    w_c=params.w_c
    g1 = params.g1
    g2 = params.g2
if no_of_QDs==2:
    j0_1 = params.j0_1
g=params.g
DvDc=params.DvDc
j0 = params.j0
j0_FGR = params.j0_FGR
omp=params.omp
shr=params.SHR
w0 = params.w0
T = params.T
r0 = params.r0
l = params.l
lbar=params.lbar
lp=params.lp
Vs = params.Vs
sharebath=params.sharebath
t0=params.t0
L = params.L
dt=params.dt1

##########################################
# calculate maximum neighbours using full tensor approach
RAM_avail=16 #free RAM in GB.
maximum_neighbours=np.emath.logn(d,(RAM_avail / (d*16*10**-9))) # 3^(L+1) tensor elements * 16 bytes per complex number stored * 10^-9 for byte to GB conversion
print('The original full tensor approach has approx. the maximum no. of neighbours:', maximum_neighbours)

def get_nth_permutation(L, Lcycle, r0,dt,length, n_perms, n_perms_U, step_no, M1, exp_k0_factor, Qlist):
    start_time = time.time() # time at which function is called
    print('n perms=',n_perms,'n permsU=',n_perms_U, 'no. of steps=', step_no, 'L=',L, 'r0=',r0 , 'dt=',dt)
    # print('no. of steps=', step_no)
    U = np.ones((n_perms_U, 1), dtype=complex)
    V = np.ones((1, n_perms), dtype=complex)
    for i in range(d):
        V[0][i::d]=M1[i,exc_channel]
    ii = 0
    indices = (np.arange(L) + 1).astype('str')[::-1] # first L/2 indices correspond to U and rest to V
    indices2 = indices
    split_point = (L ) // 2 
    P = []
    if exc_channel==measure_channel:
        P.append(np.array([1+0j]))
    else:  
        P.append(np.array([0+0j]))
    if correlator=='LP':
        P.append(np.array([exp_k0_factor*M1[measure_channel,exc_channel]]))
    if correlator=='NQD':
        P.append(np.array([exp_k0_factor[measure_channel]*M1[measure_channel,exc_channel]]))
        # print(M1[measure_channel,exc_channel])
    def generate_permutations(d,length):
        for perm in itertools.product(list(range(d)), repeat=length):
            yield perm
    permutations = np.array(list(generate_permutations(d, int(length))), dtype=np.int32)
    
    if L % 2 !=0:
        def generate_permutations_odd(d):
            for perm in itertools.product(list(range(d)), repeat=int(length_U)):
                yield perm
        permutations_odd = np.array(list(generate_permutations_odd(d)), dtype=np.int32)
    
    def generate_arrays(n_perms, d):
        return {i: np.ones(n_perms, dtype=complex) for i in range(d)}
    def generate_copies(arr, d):
        return {i: arr.copy() for i in range(d)}
    def generate_true_false_arrays(permutations, i1_pos, d):
        return {i: permutations[:, i1_pos] == i for i in range(d)}
    
    def generate_V_arrays(V, true_false_arrays):
        return {i: V[:, true_false_arrays[i]] for i in true_false_arrays}
    
    def generate_V_ip_splitting(V_arrays, d):
        return {j: V_arrays.copy() for j in range(d)}
    
    def restructure_arrays(V_ips, d, iteration):
        chunk_size = d ** iteration
        V_is = []
        for j in range(d):
            idx = 0
            combined_array = []
            num_cols = V_ips[0][0].shape[1]
            while idx < num_cols:
                combined_array.extend(arr[:, idx:idx + chunk_size] for arr in (V_ips[j][i] for i in range(len(V_ips[j]))))
                idx += chunk_size
            V_is.append(np.concatenate(combined_array, axis=1))
        return V_is
    def optimize_svd(U, V, threshold_factor):
        S1 = np.array([1, 2])
        S2 = np.array([1])
        while len(S1) != len(S2):
            # SVD on V
            A, S1, Vh = np.linalg.svd(V, full_matrices=False)
            threshold = S1[0] * threshold_factor
            thresh = S1 > threshold
            # Apply threshold
            S1 = S1[thresh]
            A = A[:, thresh]
            V = Vh[thresh, :]
            # Update U
            S = np.diag(S1)
            U = U @ A @ S
            # SVD on U
            U, S2, B = np.linalg.svd(U, full_matrices=False)
            threshold = S2[0] * threshold_factor
            thresh = S2 > threshold
            # Apply threshold
            S2 = S2[thresh]
            # print('S1 length:', len(S1), 'S2 length:', len(S2))
            B = B[thresh, :]
            U = U[:, thresh]
            # Update V
            S = np.diag(S2)
            V = S @ B @ V
        # print('SVD ended, equal lengths')
        return U, V
    
    
    # print(f"\n  Step0: Storing permutations {np.round((time.time() - start_time), 2)} seconds")
    for step in tqdm(range(step_no),position=0, leave=True):
        i_values = {}
        i_values_U = {}
        Us = generate_copies(U, d)
        # for L=4, indices on U (left matrix) are i4,i3 and the indices on V are i2,i1, after each time step they shift by 1 due to remapping since
        # we sum over i1 and introduce new index p, then we remap lowering indices by 1, i4i3 i2i1 -> i3i2 i1i4
        # cycle is to track when i1 stops being associated with V (right matrix) and is on the left, we then later set U=V.T and V=U.T
        #to return back to i4i3 on U and i2i1 on V. 
        cycle = ii % (Lcycle/2)
        if ii == 0:
            indices2 = indices
        else: 
            indices2 = np.roll(indices2, -1) #rolling the indices to simulate the remapping i4i3 i2i1 -> i3i2 i1i4
        U_indices = indices2[:split_point] # U is the left matrix, associated with the latter half of the indices
        V_indices = indices2[split_point:] # V right matrix, associated with the right half of the indices
        
        # print(f"U_indices: {U_indices}, V_indices: {V_indices}")
        
        i1_pos = np.where(V_indices == '1')[0][0] # location of i1 in the indices, since permutation order is fixed, i.e. (for qd-cavity options 0,1 only):00,01,10,11, i1 being (i2i1) or (i1i2) means i1 varies differently
        # print(i1_pos)
        # if i1 is located on the right (i2,i1), it alternates as i1= 0101 across the 4 permutations here, if i1 is the left index (i1,i2) then it varies as i1=0011
        # print(f"\n  Step1: after Arsenis' addition {np.round((time.time() - start_time), 2)} seconds")
 ##########################################################################################################################       
        if cycle == 0: #when new cycle begins reset indices (new cycle is when i1 appears on the LHS (on U) we can go back to i4i3 i2i1 this way)
            indices2 = indices 
            if step != 0: #not neccessary for the first application.
                Us = generate_copies(V.T, d)
                V = U.T         
        i_pos = np.ones(len(V_indices))
        i_pos_U = np.ones(len(V_indices))
        
        Q_Us = generate_arrays(n_perms, d)
        QV = np.ones(n_perms, dtype=complex)
        true_false_arrays = generate_true_false_arrays(permutations, i1_pos, d)
        for j in range(len(V_indices)):
            index = list(permutations[:, j])  #values of the index across permutations
            i_values[f'{V_indices[j]}'] = []
            i_values_U[f'{U_indices[j]}'] = []
            i_pos[j] = np.where(V_indices == f'{V_indices[j]}')[0][0]
            i_pos_U[j] = np.where(U_indices == f'{U_indices[j]}')[0][0]
            
            L_j_ii = int(L) - j - 2 - ii
            V_j = int(V_indices[j]) - 2 if V_indices[j] != '1' else None
            index_np = np.array(index)
            for i in range(d):
                Q_Us[i] *= Qlist[L_j_ii][index,i]            
                if V_j is not None:
                    #Use the precomputed NumPy array slice
                    QV[true_false_arrays[i]] *= Qlist[V_j][index_np[true_false_arrays[i]], i]
               
            i_values[f'{V_indices[j]}'] = permutations[:, int(i_pos[j])]
            i_values_U[f'{U_indices[j]}'] = permutations[:, int(i_pos_U[j])]
        # print(f"Step2.0: Calculating Qs {np.round((time.time() - start_time), 2)} seconds")
        Us_shape_last = Us[0].shape[-1]
        for j in range(d):
            Us_j = Us[j]  # Avoid repeated indexing
            Q_Us_j = Q_Us[j]  # Avoid repeated indexing
            for i in range(Us_shape_last):
                # Us[j][:,i] *= Q_Us[j]
                Us_j[:, i] *= Q_Us_j
                if j==0:
                    V[i, :] *= QV
                   
        V_i_splitting=generate_V_arrays(V,true_false_arrays)
        index=[]
        V=[]
        #no need to store these in memory any longer, we have applied them for the current time step
        Q_Us=[]
        QV=[]
        #this is where we apply Q_p containing K^(L) cumulant elements, to V, however p can be 0,1,2, so we will split our current V into 3.
        V_ips = [generate_V_ip_splitting(V_i_splitting[i], d) for i in range(d)]
        Qlist_last = Qlist[-1]
        for j in range(d):
            V_ips_j = V_ips[j] 
            for i in range(d):
                V_ips_j[i] *= Qlist_last[i,j]
        # for j in range(d):
        #     for i in range(d):
        #         V_ips[j][i] *= Qlist[-1][i,j]
        
        U = np.hstack([Us[i] for i in range(len(Us))])
        # U = np.hstack(Us)
        Us=[]
        # print(U)
        V_is = restructure_arrays(V_ips,d,ii)
        V = np.vstack([V_is[i] for i in range(len(V_is))])
        # V = np.vstack(V_is)
        
        V_is=[]
        V_ips=[]

        ii += 1
        if ii % (L/2) == 0:
            ii = 0
        # since to calculate polarisation at a given time step, we must compute F_pi4i3i2 = F_CCCi2, where we choose i2 to be the measurement mode j   
        i2_pos_V = np.where(V_indices == '2')[0]
        i2_pos_U = np.where(U_indices == '2')[0]
        if len(i2_pos_V) == 0:
            V_column = V[:, phonon_uncoupled_permutation]
        else:
            V_val2 = np.where(np.array(i_values['2']) == measure_channel)[0]
            for i_index in V_indices:
                if i_index != '2':
                    V_val2 = V_val2[np.where(np.array(i_values[i_index])[V_val2] == phonon_uncoupled_mode)[0]]
            V_column = V[:, V_val2]
        
        if len(i2_pos_U) == 0:
            U_row = U[phonon_uncoupled_permutation, :]
        else:
            U_val2 = np.where(np.array(i_values_U['2']) == measure_channel)[0]
            for i_index in U_indices:
                if i_index != '2':
                    U_val2 = U_val2[np.where(np.array(i_values_U[i_index])[U_val2] == phonon_uncoupled_mode)[0]]
            U_row = U[U_val2, :]
        
        U, V = optimize_svd(U, V, threshold_factor)
        if correlator=='LP':
            P.append(exp_k0_factor*(np.dot(U_row,V_column)))   
        else:
            P.append(exp_k0_factor[measure_channel]*(np.dot(U_row,V_column)))
    print(f" \n For L = {L} and {step_no} steps, code took {np.round((time.time() - start_time)/60, 2)} minutes to run or {np.round((time.time() - start_time),2)} seconds ")   
    return   np.array(P)

step_no = int(tfinal/dt)
length = L/2
length_U=L/2
n_perms=int(d**((L/2)))
n_perms_U=int(d**((L/2)))
Lcycle=L
if L % 2 != 0:
    length= (L+1)/2
    length_U=(L-1)/2
    n_perms=int(d**((L+1)/2))
    n_perms_U=int(d**((L-1)/2))
    Lcycle=L+1

# To generate the cumulant elements, K_inin are the K_11, K_22 elements (A_k, C_k in SM), K are K_12=K21 elements (B_k in SM)
if no_of_QDs==2:  #QD-QD or QD-QD-Cavity system
    class Cumulants(object):
        """A set of functions to run numerics"""
    
        def __init__(self):
            self.parameters = {}
    
        def update_parameters(self, **kwargs):
            """Update parameters"""
            for key, value in kwargs.items():
                self.parameters[key] = value
    
        def Ku(self, t, r0, j0, j0_1, w0, T, Vs, l, dotshape, sharebath, lp):
            """Short time part of the cumulant"""
            if dotshape == 'spherical':
                Kinf0 = -1j * PolaronShift(j0, w0) * t - S_inin(T, j0, w0)
                Kinf1 = (-1j * PolaronShift_inim(j0, w0, r0, l) * t - S_inim(T, j0_1, w0, r0, Vs))
                K11s = phi_inin(t, j0, w0, T) + Kinf0
                K12s = sharebath * (phi_inim(t, j0_1, w0, T, r0, Vs) + Kinf1)
            elif dotshape == 'smartie':
                K11s = K11_smartie(t, j0, l, lp, Vs, T)
                K12s = sharebath * (K12_smartie(t, j0, l, lp, Vs, T, r0))
            return K11s, K12s
    
        def Kn(self, cumulants, cumulantsinin, dt, n, r0, j0, j0_1, w0, T, Vs, l, dotshape, sharebath, lp):
            """Finds next cumulant in the set"""
            kk = []
            kkinin = []
            for q in range(1, n):
                kk.append(2 * (n + 1 - q) * cumulants[q])
                kkinin.append(2 * (n + 1 - q) * cumulantsinin[q])
            Kinims = 0.5 * (self.Ku((n + 1) * dt, r0, j0, j0_1, w0, T, Vs, l, dotshape, sharebath, lp)[1] -
                            (n + 1) * self.Ku(dt, r0, j0, j0_1, w0, T, Vs, l, dotshape, sharebath, lp)[1] - sum(kk))
            Kinins = 0.5 * (self.Ku((n + 1) * dt, r0, j0, j0_1, w0, T, Vs, l, dotshape, sharebath, lp)[0] -
                            (n + 1) * self.Ku(dt, r0, j0, j0_1, w0, T, Vs, l, dotshape, sharebath, lp)[0] - sum(kkinin))
            return Kinims, Kinins
    
        def cu(self, L, dt, r0, j0, j0_1, w0, T, Vs, l, dotshape, sharebath, lp):
            """Calculates a set of square cumulants for L neighbours and updates an array"""
            K0, K0_inim = self.Ku(dt, r0, j0, j0_1, w0, T, Vs, l, dotshape, sharebath, lp)
            # print(K0,K0_inim)
            cumulants = [K0_inim]
            cumulantsinin = [K0]
            for n in range(1, L + 1):
                Knn, Knn_inin = self.Kn(cumulants, cumulantsinin, dt, n, r0, j0, j0_1, w0, T, Vs, l, dotshape, sharebath, lp)
                cumulants.append(Knn)
                cumulantsinin.append(Knn_inin)
            return cumulants, cumulantsinin

 
if no_of_QDs==1: # QD-cavity system

    class Cumulants_1qd(object):
        """A set of functions to run numerics."""
        def __init__(self):
            self.parameters = {}
    
        def update_parameters(self, **kwargs):
            """Update parameters"""
            for key, value in kwargs.items():
                self.parameters[key] = value
    
        def updateK(self, in_K):
            """Update the set of cumulants."""
            self.K = in_K
            return self.K
    
        def Ku(self, t, omp, shr, T, j0, w0):
            """Short time part of the cumulant."""
            Kinf = -1j * omp * t - shr
            return Kbb2(t, T, j0, w0) + Kinf
    
        def Kn(self, cumulants, dt, omp, shr, T, j0, w0):
            """Finds the next cumulant in the set."""
            kk = [2 * (len(cumulants) + 1 - q) * cumulants[q] for q in range(1, len(cumulants))]
            self.Knn = 0.5 * (self.Ku((len(cumulants) + 1) * dt, omp, shr, T, j0, w0) - 
                              (len(cumulants) + 1) * self.Ku(dt, omp, shr, T, j0, w0) - sum(kk))
    
        def cu(self, L, dt, omp, shr, T, j0, w0):
            """Calculates a set of square cumulants for L neighbours and updates an array."""
       
            K1 = self.Ku(dt, omp, shr, T, j0, w0)
            cumulants = [K1]
            while len(cumulants) < L + 1:
                self.Kn(cumulants, dt, omp, shr, T, j0, w0)
                cumulants.append(self.Knn)
            self.updateK(in_K=cumulants)
            return self.K
        
        
if correlator=='NQD':
    if ec=='2':
        exc_channel=4
    else:
        exc_channel=1
    if mc=='2':
        measure_channel=4
    else:
        measure_channel=1
        
    phonon_uncoupled_mode=0
    phonon_uncoupled_permutation=0

    LF = LFpop(g,  w_qd1.real, w_qd2.real, -params.gamma1/hbar, -params.gamma2/hbar) 
    M1=expm(-1j * LF * dt)
    # print(M1)
    
    params_cumulant = {
        'r0': r0,
        'j0': j0,
        'j0_1': j0_1,
        'w0': w0,
        'T': T,
        'Vs': Vs,
        'l': l,
        'dotshape': dotshape,
        'sharebath': sharebath,
        'lp': lp}  
    
    cumulants = Cumulants()
    cumulants.update_parameters(**params_cumulant)
    cumulants, cumulants_inin = cumulants.cu(L, dt, **params_cumulant)

    size = 5
    alpha=np.array([0,1,1,0,0])
    beta=np.array([0,1,0,1,0])
    mu=np.array([0,0,0,1,1])
    nu=np.array([0,0,1,0,1])
    x=mu
    Qlist=[]
    Q = np.random.rand(size, size) + 1j * np.random.rand(size, size)
    for i in range(size):
        for j in range(size):
            Q[i, j] = M1[i,j] *(np.exp((alpha[j]-beta[j])*alpha[j]*cumulants_inin[0] + (beta[j]-alpha[j])*beta[j]*np.conjugate(cumulants_inin[0]) 
            + (x[j]-nu[j])*x[j] * cumulants_inin[0] + (nu[j] - x[j])*nu[j]*np.conjugate(cumulants_inin[0]) 
            + (  (x[j] - nu[j])*alpha[j] + (alpha[j]-beta[j])*x[j]   )*cumulants[0]
            + ( (nu[j]-x[j])*beta[j]  + (beta[j]-alpha[j])*nu[j]   )*np.conjugate(cumulants[0])) * np.exp ( 2*( (alpha[i]-beta[i])*alpha[j]*cumulants_inin[1] + (beta[i]-alpha[i])*beta[j]*np.conjugate(cumulants_inin[1]) 
            + (x[i]-nu[i])*x[j] * cumulants_inin[1] + (nu[i] - x[i])*nu[j]*np.conjugate(cumulants_inin[1]) 
            + (  (x[i] - nu[i])*alpha[j] + (alpha[i]-beta[i])*x[j]   )*cumulants[1]
            + ( (nu[i]-x[i])*beta[j]  + (beta[i]-alpha[i])*nu[j]   )*np.conjugate(cumulants[1]) )))
    Qlist.append(Q)
    for r in range(int(L-1)):
        Q = np.random.rand(size, size) + 1j * np.random.rand(size, size)
        for i in range(size):
            for j in range(size):
                Q[i, j] = np.exp ( 2*( (alpha[i]-beta[i])*alpha[j]*cumulants_inin[r+2] + (beta[i]-alpha[i])*beta[j]*np.conjugate(cumulants_inin[r+2]) 
                + (x[i]-nu[i])*x[j] * cumulants_inin[r+2] + (nu[i] - x[i])*nu[j]*np.conjugate(cumulants_inin[r+2]) 
                + (  (x[i] - nu[i])*alpha[j] + (alpha[i]-beta[i])*x[j]   )*cumulants[r+2]
                + ( (nu[i]-x[i])*beta[j]  + (beta[i]-alpha[i])*nu[j]   )*np.conjugate(cumulants[r+2]) ))       
        Qlist.append(Q)
    
    K12=cumulants
    K11=cumulants_inin
    K22=cumulants_inin 
    K12s=np.conjugate(K12)
    K11s=np.conjugate(K11)
    K22s=np.conjugate(K22)  
    def twotimecorrdiag(nm): 
        'for diagonal cumulant only'
        tdiff=abs(nm)
        ttc=(beta-alpha)*(beta*K11s[tdiff]-alpha*K11[tdiff]
        +(mu)*K22[tdiff]-nu*K22s[tdiff]
        +(alpha-mu)*K12[tdiff]+(nu-beta)*K12s[tdiff])
        return ttc
    KK=twotimecorrdiag(0)
    exp_k0_factor=np.exp(KK)
    
    
    
if correlator=='LP' and no_of_QDs==2:
    if ec=='1':
        exc_channel=0
    if ec=='2':
        exc_channel=1
    if ec=='C':
        exc_channel=2
    if mc=='1':
        measure_channel=0
    if mc=='2':
        measure_channel=1
    if mc=='C':
       measure_channel=2    
        
    phonon_uncoupled_mode=2
    phonon_uncoupled_permutation=-1
    LF=LFpol_qdqdcav(g1, g2,gd, w_qd1, w_qd2, w_c)
    M1=expm(-1j*LF*dt)
    params_cumulant = {
        'r0': r0,
        'j0': j0,
        'j0_1': j0_1,
        'w0': w0,
        'T': T,
        'Vs': Vs,
        'l': l,
        'dotshape': dotshape,
        'sharebath': sharebath,
        'lp': lp}  
    
    cumulants = Cumulants()
    cumulants.update_parameters(**params_cumulant)
    cumulants, cumulants_inin = cumulants.cu(L, dt, **params_cumulant)
    
    Q0=np.array([[M1[0,0]*np.exp(cumulants_inin[0] +2*cumulants_inin[1]), M1[0,1]*np.exp(cumulants_inin[0] +2*cumulants[1]), M1[0,2] ],
                  [M1[1,0]*np.exp(cumulants_inin[0] +2*cumulants[1]), M1[1,1]*np.exp(cumulants_inin[0] +2*cumulants_inin[1]), M1[1,2] ],
                  [M1[2,0]*np.exp(cumulants_inin[0]), M1[2,1]*np.exp(cumulants_inin[0]), M1[2,2]]])
    Qlist=[]
    Qlist.append(Q0)
    for i in range(int(L-1)):
        Qlist.append(np.array([[np.exp(2*cumulants_inin[i+2]), np.exp(2*cumulants[i+2]), 1 ],
                                [np.exp(2*cumulants[i+2]), np.exp(2*cumulants_inin[i+2]), 1],
                                [1, 1, 1]]))
    exp_k0_factor=np.exp(cumulants_inin[0])
    
if correlator=='LP' and no_of_QDs==1:
    if ec=='1':
        exc_channel=0
    if ec=='C':
        exc_channel= 1
    if mc=='1':
        measure_channel=0
    if mc=='C':
       measure_channel=1    
       
    phonon_uncoupled_mode=1
    phonon_uncoupled_permutation=-1
    D, ww, U1, V1 = DiagM_qdcav(gd, 0, 0,0, detuning, omp)
    DD = np.diag(np.exp(-1j*ww*dt))
    M1 = U1*DD*V1
    
    params_cumulant = {
        'omp': omp,
        'shr': shr,
        'j0': j0,
        'T': T,
        'w0': w0}  
    
    Cumulants = Cumulants_1qd()
    Cumulants.update_parameters(**params_cumulant)
    cumulants_inin = Cumulants.cu(L, dt, **params_cumulant)
    cumulants=cumulants_inin
    
    # cumulants_inin=Cumulants_1qd().cu(L,dt) 
    # cumulants=cumulants_inin
    Q0=np.array([[M1[0,0]*np.exp(cumulants[0] +2*cumulants[1]), M1[0,1] ],[M1[1,0]*np.exp(cumulants[0]), M1[1,1]]])
    Qlist=[]
    Qlist.append(Q0)
    for i in range(int(L-1)):
        Qlist.append(np.array([[np.exp(2*cumulants[i+2]), 1 ],[1, 1]]))
    exp_k0_factor=np.exp(cumulants_inin[0])
# Generate permutations, QD-QD-Cav has states 0, 1, 2 (exciton 1 refers to 0, exciton 2 refers to 1 and cavity mode is 2)


# print('params, dt=',dt,'d=',r0)
# print('cumulants',cumulants)
# print('cumulants_inin',cumulants_inin)
### Attempt to load the data or run the code to generate the data ####
try:
    P=np.load(path+"/data/"+params.label()+".npy",allow_pickle=True)
except:
    P= get_nth_permutation(L, Lcycle, r0,dt,length, n_perms, n_perms_U, step_no, M1, exp_k0_factor, Qlist)
    np.save(path+"/data/"+params.label()+".npy",P)

P=np.ravel(P)
times = np.array([dt*i for i in range(step_no+2)])
# times=times[2:]
# 

def calculate_rmse(y_true, y_pred):
    squared_errors = (np.abs(y_true) - np.abs(y_pred)) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse

###################### Plotting current data #####################

# fig1 = plt.figure(1, figsize=(4.5,3),dpi=150)
# bb = fig1.add_subplot(1, 1, 1)  

# bb.plot(times, np.abs(P), 'r-', linewidth='1', label=f'SVD, L={L}')
# if correlator=='LP':
#     bb.set_ylabel(r'$|P_{1}(t)|$',fontsize='12')
# if correlator=='NQD':
#     if exc_channel==4:
#         label_channel=2
#     bb.set_ylabel(rf'$N_{{{measure_channel}{label_channel}}}(t)$',fontsize='12')

# bb.set_xlabel(r'time $\mathrm{(ps)}$',fontsize='12')
# # bb.set_ylim(0,1)
# bb.set_xlim(0,tf)
# # bb.set_ylim(1e-2,1)
# plt.yscale('log')
# plt.tick_params(axis='both', which='major', labelsize=10)
# plt.tick_params(axis='both', which='major', labelsize=10)
# leg=bb.legend(fontsize=8, loc="best")
# leg.get_frame().set_color('none')
# leg.get_frame().set_alpha(0)
# plt.tight_layout()

plt.figure(10)
plt.plot(times, np.abs(P), linewidth='1', label=f'SVD, L={L}')
plt.legend(loc='best')
plt.yscale('log')
if correlator=='NQD':
    if exc_channel==4:
        label_channel=2
    plt.ylabel(rf'$N_{{{measure_channel}{label_channel}}}(t)$',fontsize='12')
    plt.xlabel(r'time $\mathrm{(ps)}$',fontsize='12')
    plt.title(f' $g$={round(g*hbar*1e3,1)}$\mu eV$, det={round(detuning*hbar*1e3,1)}$\mu eV$, T={round(T*hbar/kb,1)}K, d={r0} nm, Dc-Dv=-{DvDc} eV, l={l} nm')
    plt.legend(loc='best')
    
#%% #!!!
###### Triexponential fitting #######
if fit==1:
    import lmfit
    def Triexponential(t, A1_re,A1_im,A2_re,A2_im,A3_re,A3_im,w1_re,w1_im,w2_re,w2_im,w3_re,w3_im):
        return (A1_re - 1j*A1_im)*np.exp(-1j*(w1_re -1j*w1_im)*t) + (A2_re - 1j*A2_im)*np.exp(-1j*(w2_re -1j*w2_im)*t)+ (A3_re - 1j*A3_im)*np.exp(-1j*(w3_re -1j*w3_im)*t)
    
    def Triexponential_with_constraints(t, A1_re,A1_im,A2_re,A2_im,A3_re,A3_im,w1_re,w1_im,w2_re,w2_im,w3_re,w3_im):
       result= (A1_re - 1j*A1_im)*np.exp(-1j*(w1_re -1j*w1_im)*t) + (A2_re - 1j*A2_im)*np.exp(-1j*(w2_re -1j*w2_im)*t)+ (A3_re - 1j*A3_im)*np.exp(-1j*(w3_re -1j*w3_im)*t)
       
       if np.any(t == 0):
           penalty = np.maximum(0, np.abs(result[t == 0]) - np.abs(P[0]))
           result += penalty
           print(np.abs(result[t==0]))
           print(np.abs(P[0]))
       return result
    


    
    def biexponential(t, A1_re,A1_im,A2_re,A2_im,w1_re,w1_im,w2_re,w2_im):
        return (A1_re - 1j*A1_im)*np.exp(-1j*(w1_re -1j*w1_im)*t) + (A2_re - 1j*A2_im)*np.exp(-1j*(w2_re -1j*w2_im)*t)
    
    def population(t,A,B,C,T1,T2,Rabi,phase):
        return A+B*np.exp(-(T1)*t) + C*np.exp(-(T2)*t)*np.cos(Rabi*t + phase)
    
    def quintexponential(t, A1_re,A1_im,A2_re,A2_im,A3_re,A3_im,A4_re,A4_im,A5_re,A5_im, w1_re,w1_im,w2_re,w2_im,w3_re,w3_im,w4_re,w4_im, w5_re, w5_im):
        return (A1_re - 1j*A1_im)*np.exp(-1j*(w1_re -1j*w1_im)*t) + (A2_re - 1j*A2_im)*np.exp(-1j*(w2_re -1j*w2_im)*t)+ (A3_re - 1j*A3_im)*np.exp(-1j*(w3_re -1j*w3_im)*t) + (A4_re - 1j*A4_im)*np.exp(-1j*(w4_re -1j*w4_im)*t) + (A5_re - 1j*A5_im)*np.exp(-1j*(w5_re -1j*w5_im)*t)
    def quadexponential(t, A1_re,A1_im,A2_re,A2_im,A3_re,A3_im,A4_re,A4_im, w1_re,w1_im,w2_re,w2_im,w3_re,w3_im,w4_re,w4_im):#,A5_re,A5_im, w5_re,w5_im):
        return (A1_re - 1j*A1_im)*np.exp(-1j*(w1_re -1j*w1_im)*t) + (A2_re - 1j*A2_im)*np.exp(-1j*(w2_re -1j*w2_im)*t)+ (A3_re - 1j*A3_im)*np.exp(-1j*(w3_re -1j*w3_im)*t) + (A4_re - 1j*A4_im)*np.exp(-1j*(w4_re -1j*w4_im)*t) #+ (A5_re - 1j*A5_im)*np.exp(-1j*(w5_re -1j*w5_im)*t)
            
    FitParameters_all=[]
    fit_errors=[]
    A1s=[]
    A2s=[]  
    A3s=[]
    A4s=[]
    A5s=[]
    w1s=[]
    w2s=[]
    w3s=[]
    w4s=[]
    w5s=[]
  
    def fitprocedure(g,g1,g2,gd,r0,t0, P, times): 
        A1s=[]
        A2s=[]  
        A3s=[]
        A4s=[]
        A5s=[]
        w1s=[]
        w2s=[]
        w3s=[]
        w4s=[]
        w5s=[]
      
        # diagonalise numerically H_0 (no phonons to provide initial guess)
        if no_of_QDs==2 and cavity==1:
            matrix = np.array([[w_qd1, gd, g1],
                       [gd, w_qd2, g2],
                       [g1, g2, w_c]])
            eigenvalues = np.linalg.eigvals(matrix)
            eigenvalues=  np.sort(eigenvalues)[::-1]
            # print('eigenvalues:',eigenvalues)
            # dephasing1,dephasing2,dephasing3=FGR_smartie(j0_FGR,l,lp,Vs,T,g1,gd,w_qd1,w_c,np.array([r0]))
            # dephasing1=np.asarray(dephasing1, dtype=np.float64)[0]*1e-3/hbar
            # dephasing2=np.asarray(dephasing2, dtype=np.float64)[0]*1e-3/hbar
            # dephasing3=np.asarray(dephasing3, dtype=np.float64)[0]*1e-3/hbar

        if no_of_QDs==1 and cavity==1:   
            matrix = np.array([[w_qd1, gd],
                               [gd, w_c]])
            eigenvalues = np.linalg.eigvals(matrix)
            eigenvalues=  np.sort(eigenvalues)[::-1]
            
        
        # print("Eigenvalues:", eigenvalues) 
        if no_of_QDs==2 and cavity==1 and g1!=0:
            # print('QD-QD-Cavity model')
            A1s.append(0.5 - 0*1j)
            A2s.append(0.5 - 0*1j)  
            A3s.append(0.5- 0*1j)             
            w1s.append(eigenvalues[1] + 0*1j)                
            w2s.append(eigenvalues[0]+ 0*1j)      
            w3s.append(eigenvalues[2] + 0*1j) 
            # w1s.append(0 + 0*1j)                
            # w2s.append(0+ 0*1j)      
            # w3s.append(0 + 0*1j) 
        
        if no_of_QDs==1 and cavity==1:
            # print('QD-cavity model')
            A1s.append(0.5 - 0*1j)
            A2s.append(0.5 - 0*1j)            
            w1s.append(eigenvalues[0] + 0.03*1j)                
            w2s.append(eigenvalues[1]+ 0.02*1j)      

        if no_of_QDs==2 and cavity==1 and g1==0:
            # print('QD-QD using QD-QD-cav model with cavity mode turned off')
            A1s.append(0.5 - 0*1j)
            A2s.append(0.5 - 0*1j)            
            w1s.append(eigenvalues[0] + 0*1j)                
            w2s.append(eigenvalues[2]+ 0*1j)
            # w1s.append(0 + 0*1j)                
            # w2s.append(0+ 0*1j)
        if no_of_QDs==2 and cavity==0:
            # print('QD-QD population dynamics')
            LF = LFpop(g,  w_qd1.real, w_qd2.real, -params.gamma1/hbar, -params.gamma2/hbar) 
            eigenvalues=np.linalg.eigvals(LF)
            eigenvalues=np.sort(eigenvalues)[::-1]
            A1s.append(0 - 0*1j)
            A2s.append(0.0 - 0*1j)  
            A3s.append(0.0 - 0*1j)  
            A4s.append(0.0 - 0*1j) 
            A5s.append(0.0 - 0*1j) 
            # w1s.append(eigenvalues[0] + 0*1j)                
            # w2s.append(eigenvalues[1]+ 0*1j)      
            # w3s.append(eigenvalues[2] + 0*1j) 
            # w4s.append(eigenvalues[3] + 0*1j) 
            # w5s.append(eigenvalues[4] + 0*1j) 
            w1s.append(0 + 0*1j)                
            w2s.append(0+ 0*1j)      
            w3s.append(0 + 0*1j) 
            w4s.append(0 + 0*1j) 
            w5s.append(0 + 0*1j) 
            ##########################
            # print('r0 in fitprocedure is:',r0)
            GamPh,RF,dp,dm, lambdp, lambdm=GammaPh(r0,params.det,params.T)
            gam1=Gamma1_FGR_det(r0,params.det,params.T)/hbar 
            gam2=Gamma2_FGR_det(r0,params.det,params.T)/hbar 
            Ct,AA,ww=Pan_NQD(0,r0,params.det,params.T)
         
            A=AA[4]
            C=AA[2]+AA[3]
            B=AA[1]#-A-C #-(AA[1]+AA[3]) # -A-C
            Gamd=(gam1+gam2)
            Gams=(2*Gamd) 
            Phi=0
            pa=np.array([A,B,C,Gams,Gamd,RF,Phi]).real
            pa=np.asarray(pa)
            # print('guess params a,b,c,T1,T2,R,phi:', pa, 'for d=',r0)

   
        if no_of_QDs==2 and cavity==0: 
            Pnlongt=np.real(P[np.where(times>24)]) #extracting only the longt behaviour
            tmlongt=times[np.where(times>24)] #extracting only the longt behaviour
            popn= lmfit.Model(population)
            guesses = popn.make_params(A=pa[0], B=pa[1], C=pa[2], T1=pa[3], T2=pa[4], Rabi=pa[5], phase=pa[6])
            result = popn.fit(Pnlongt, params=guesses, t=tmlongt, method='leastsq', max_nfev=10000, nan_policy='omit', verbose=True)
            fit_sametimes=popn.eval(params=result.params,t=times)
            ##############
            # popn_2 = lmfit.Model(quadexponential)
            # k=0
            # # guesses = popn_2.make_params(A1_re=np.real(A1s[k]), A1_im=-np.imag(A1s[k]),A2_re=np.real(A2s[k]), A2_im=-np.imag(A2s[k]),A3_re=np.real(A3s[k]), A3_im=-np.imag(A3s[k]),A4_re=np.real(A4s[k]), A4_im=-np.imag(A4s[k]) , A5_re=np.real(A5s[k]), A5_im=-np.imag(A5s[k]),w1_re=np.real(w1s[k]), w1_im=-np.imag(w1s[k]),w2_re=np.real(w2s[k]), w2_im=-np.imag(w2s[k]),w3_re=np.real(w3s[k]), w3_im=-np.imag(w3s[k]),w4_re=np.real(w4s[k]), w4_im=-np.imag(w4s[k]), w5_re=np.real(w5s[k]), w5_im=-np.imag(w5s[k]))
            # guesses = popn_2.make_params(A1_re=np.real(A1s[k]), A1_im=-np.imag(A1s[k]),A2_re=np.real(A2s[k]), A2_im=-np.imag(A2s[k]),A3_re=np.real(A3s[k]), A3_im=-np.imag(A3s[k]),A4_re=np.real(A4s[k]), A4_im=-np.imag(A4s[k]) ,w1_re=np.real(w1s[k]), w1_im=-np.imag(w1s[k]),w2_re=np.real(w2s[k]), w2_im=-np.imag(w2s[k]),w3_re=np.real(w3s[k]), w3_im=-np.imag(w3s[k]),w4_re=np.real(w4s[k]), w4_im=-np.imag(w4s[k])) #,  A5_re=np.real(A5s[k]), A5_im=-np.imag(A5s[k]),w5_re=np.real(w5s[k]), w5_im=-np.imag(w5s[k]))
            # result = popn_2.fit(Pnlongt, params=guesses, t=tmlongt, method='leastsq', max_nfev=10000, nan_policy='omit', verbose=True)
            # fit_sametimes=popn_2.eval(params=result.params,t=times)
        
        elif no_of_QDs==2 and cavity==1 and g1!=0 :
            Pnlongt=P[np.where(times>t0+2.0*tauib)] #extracting only the longt behaviour
            tmlongt=times[np.where(times>t0+2.0*tauib)] #extracting only the longt behaviour
            Triexp=lmfit.Model(Triexponential)
            # Triexp=lmfit.Model(Triexponential_with_constraints)
            k=0
            guesses= Triexp.make_params(A1_re=np.real(A1s[k]), A1_im=-np.imag(A1s[k]), A2_re=np.real(A2s[k]) , A2_im=-np.imag(A2s[k]), A3_re=np.real(A3s[k]) , A3_im=-np.imag(A3s[k]), w1_re=np.real(w1s[k]), w1_im=-np.imag(w1s[k]) , w2_re=np.real(w2s[k]), w2_im=-np.imag(w2s[k]), w3_re=np.real(w3s[k]), w3_im=-np.imag(w3s[k]))
            # value_at_t0_guess = Triexponential(0, **guesses.valuesdict())
            # print(np.abs(value_at_t0_guess))
            # if np.abs(value_at_t0_guess) > np.abs(P[0]):
            #     # Scale down the parameters
            #     scale_factor = np.abs(P[0]) / np.abs(value_at_t0_guess)
            #     for key in ['A1_re', 'A1_im', 'A2_re', 'A2_im', 'A3_re', 'A3_im']:
            #         guesses[key].value *= scale_factor

            result= Triexp.fit(Pnlongt, params=guesses, t=tmlongt, method='leastsq', verbose=True) #fit of current calculated P
            fit_sametimes=Triexp.eval(params=result.params, t=times)
            #Evaluate the fitted value at t = 0
            # value_at_t0 = Triexponential(0, **result.params.valuesdict())
            # print("Fitted value at t=0:", value_at_t0)
            # print("Magnitude of fitted value at t=0:", np.abs(value_at_t0))
            # print("Original data at t=0:", np.abs(P[0]))
                        
            
            
        elif (no_of_QDs==2 and cavity==1 and g1==0) or (no_of_QDs==1 and cavity==1):
            Pnlongt=P[np.where(times>t0+2.0*tauib)] #extracting only the longt behaviour
            tmlongt=times[np.where(times>t0+2.0*tauib)] #extracting only the longt behaviour
            biexp=lmfit.Model(biexponential)

            k=0
            guesses= biexp.make_params(A1_re=np.real(A1s[k]), A1_im=-np.imag(A1s[k]), A2_re=np.real(A2s[k]) , A2_im=-np.imag(A2s[k]), w1_re=np.real(w1s[k]), w1_im=-np.imag(w1s[k]) , w2_re=np.real(w2s[k]), w2_im=-np.imag(w2s[k]))
            # guesses= biexp.make_params(A1_re=np.real(A1s[k]), A1_im=-np.imag(A1s[k]), A2_re=np.real(A2s[k]) , A2_im=-np.imag(A2s[k]), w1_re=np.real(w1s[k]), w2_re=np.real(w2s[k]), w2_im=-np.imag(w2s[k]))
            # guesses.add('w1_im',  min=guesses['w2_im'].value + 0.0000000005)
            tfit=np.linspace(0,100,10000)
            result= biexp.fit(Pnlongt, params=guesses, t=tmlongt, method='leastsq', verbose=True) #fit of current calculated P
            fit_sametimes=biexp.eval(params=result.params, t=tfit)
           
            
            
        FitParameters_all.append(result.best_values)
        errors = {param: result.params[param].stderr for param in result.params}
        fit_errors.append(errors)
        return tmlongt, Pnlongt, fit_sametimes
    
    tfit=np.linspace(0,100,10000)
    avg_error=[]
    # longtimes=np.arange(2,40,1)
    # for i in longtimes:
    #     print(i)
    tmlongt, Pnlongt, fit_sametimes= fitprocedure(g,g1,g2,gd,r0,t0,P,times)
    # error=(np.abs(np.real(P) - np.real(fit_sametimes)))
    # avg_error.append(np.mean(error[np.where(times>t0 + 2*tauib)]))
    # print(avg_error)

    fig1 = plt.figure( figsize=(4.5,3),dpi=150)
    bb = fig1.add_subplot(1, 1, 1)     
    bb.plot(np.abs(times),np.abs(P),'r-',markersize='1',linewidth='0.4', label=f'SVD, L={L}')
    # bb.plot(np.abs(times),np.abs(fit_sametimes),'b--',label=f'fit, L={L}') 
    bb.plot(np.abs(tfit),np.abs(fit_sametimes),'b--',label=f'fit, L={L}') 
    plt.yscale('log')
    FitParameters_all
    # bb.plot(times,np.abs(error),'g-',label=f'error, mean={avg_error:.3e}')
    if exc_channel==4:
        label_channel=2
    bb.set_ylabel(rf'$N_{{{measure_channel}{label_channel}}}(t)$',fontsize='12')
    bb.set_xlabel(r'time $\mathrm{(ps)}$',fontsize='12')
    bb.set_title(f' $g$={round(g*hbar*1e3,1)}$\mu eV$, det={round(detuning*hbar*1e3,1)}$\mu eV$, T={round(T*hbar/kb,1)}K,\n d={r0} nm, Dc-Dv=-{DvDc} eV, l={l} nm')
    plt.legend(loc='best') 
    
    plt.tight_layout()
    FitParameters_all
   
    # plt.xticks(fontsize='10')
    # plt.yticks(fontsize='10')
    # bb.set_ylabel(r'$|P_{1}(t)|$',fontsize='12')
    # bb.set_xlabel(r'time $\mathrm{(ps)}$',fontsize='12')
    # leg=bb.legend(fontsize=8, loc="best")
    # leg.get_frame().set_color('none')
    # leg.get_frame().set_alpha(0)
    
    

#%% #!!! 
############# running through Ls to generate errors and parameters ################
    Ls=np.arange(20,40+1,2)
    # threshold_factor=1e-6
    # threshold_str = str(threshold_factor)
    FitParameters_all=[] 
    errors=[]
    fit_errors=[]
    for L in Ls:
        ###params that change with L ###
        t0=r0/Vs
        if no_of_QDs==1:
            t0=0
        # factortau=2.0
        if sharebath ==1:
            dt=( t0 + factortau*tauib )/(L+1)
            # print('dt',dt)
        if sharebath ==0:
            dt=2*tauib / (L+1)
            
        step_no = int(tfinal/dt)
        length = L/2
        length_U=L/2
        # print(L)
        n_perms=int(d**((L/2)))
        # print(n_perms)
        n_perms_U=int(d**((L/2)))
        Lcycle=L
        if no_of_QDs==1 and cavity==1:
            print(gd*1e3*hbar)
            D, ww, U1, V1 = DiagM_qdcav(gd, 0, 0,0, detuning, omp)
            DD = np.diag(np.exp(-1j*ww*dt))
            M1 = U1*DD*V1
            # print('yes')
        if no_of_QDs==2 and cavity==1:
            LF=LFpol_qdqdcav(g1, g2, gd, w_qd1, w_qd2, w_c)
            M1=expm(-1j*LF*dt)
            # print('no')
        # print('L=', L, 'dt=',np.round(dt,3))
        ### ##########generate data ###
        try:
            P=np.load(path+"/data/"+params.label(L=L,gd=gd,factortau=factortau)+".npy",allow_pickle=True)
        except:
            if no_of_QDs==2 and cavity==1:
                # print('no')
                params_cumulant = {
                    'r0': r0,
                    'j0': j0,
                    'j0_1': j0_1,
                    'w0': w0,
                    'T': T,
                    'Vs': Vs,
                    'l': l,
                    'dotshape': dotshape,
                    'sharebath': 1,
                    'lp': lp}  
                
                cumulants = Cumulants()
                cumulants.update_parameters(**params_cumulant)
                cumulants, cumulants_inin = cumulants.cu(L, dt, **params_cumulant)
                Q0=np.array([[M1[0,0]*np.exp(cumulants_inin[0] +2*cumulants_inin[1]), M1[0,1]*np.exp(cumulants_inin[0] +2*cumulants[1]), M1[0,2] ],
                             [M1[1,0]*np.exp(cumulants_inin[0] +2*cumulants[1]), M1[1,1]*np.exp(cumulants_inin[0] +2*cumulants_inin[1]), M1[1,2] ],
                             [M1[2,0]*np.exp(cumulants_inin[0]), M1[2,1]*np.exp(cumulants_inin[0]), M1[2,2]]])
                Qlist=[]
                Qlist.append(Q0)
                for i in range(int(L-1)):
                    Qlist.append(np.array([[np.exp(2*cumulants_inin[i+2]), np.exp(2*cumulants[i+2]), 1 ],
                                        [np.exp(2*cumulants[i+2]), np.exp(2*cumulants_inin[i+2]), 1],
                                        [1, 1, 1]]))
                exp_k0_factor=np.exp(cumulants_inin[0])

            if no_of_QDs==1 and cavity==1:
                # print('yes')
                params_cumulant = {
                    'omp': omp,
                    'shr': shr,
                    'j0': j0,
                    'T': T,
                    'w0': w0}  
                
                Cumulants = Cumulants_1qd()
                Cumulants.update_parameters(**params_cumulant)
                cumulants_inin = Cumulants.cu(L, dt, **params_cumulant)
                cumulants=cumulants_inin
                Q0=np.array([[M1[0,0]*np.exp(cumulants[0] +2*cumulants[1]), M1[0,1] ],[M1[1,0]*np.exp(cumulants[0]), M1[1,1]]])
                Qlist=[]
                Qlist.append(Q0)
                
                for i in range(int(L-1)):
                    Qlist.append(np.array([[np.exp(2*cumulants[i+2]), 1 ],[1, 1]]))
                exp_k0_factor=np.exp(cumulants_inin[0])
                # print('L=',L,' len Qlist:', len(Qlist))  
            P= get_nth_permutation(L, Lcycle, r0,dt,length, n_perms, n_perms_U, step_no, M1, exp_k0_factor, Qlist)
            np.save(path+"/data/"+params.label(L=L)+".npy",P)
        P=np.ravel(P)
        times = np.array([dt*i for i in range(step_no+2)])
        # times=times[2:]
        
        tmlongt, Pnlongt, fit_sametimes = fitprocedure(g,g1,g2,gd,r0,t0,P,times)   
        
        # errorRMSE=calculate_rmse(np.real(Pnlongt), np.real(fit_sametimes))
        # errors.append(errorRMSE)

        ###
        # if L == 18:
        plt.plot(times,np.abs(P), label=f'{L}')
        plt.legend(loc='best')
            # fig1 = plt.figure( figsize=(4.5,3),dpi=150)
            # bb = fig1.add_subplot(1, 1, 1)  
            # bb.plot(np.abs(times),np.abs(P),'b-',markersize='1',linewidth='0.8', label=f'SVD, L={L}')
            # # bb.plot(np.abs(tmlongt),np.abs(Pnlongt),'r--',markersize='1',linewidth='0.4', label=f'SVD, L={L}')
            # bb.plot(np.abs(times),np.abs(fit_sametimes),'g--',label=f'fit L={L}') 
            # plt.yscale('log')
            # plt.xticks(fontsize='10')
            # plt.yticks(fontsize='10')
            # plt.title(f'd={r0}')
            # bb.set_ylabel(r'$|P_{11}(t)|$',fontsize='12')
            # bb.set_xlabel(r'time $\mathrm{(ps)}$',fontsize='12')
            # leg=bb.legend(fontsize=8, loc="best")
            # leg.get_frame().set_color('none')
            # leg.get_frame().set_alpha(0)
            # plt.tight_layout()
        ###
        

########## Relative error calculation and plotting #################
    w1, w2, w3, gam1, gam2, gam3 = [[] for _ in range(6)]
    for i in range(len(Ls)-1):
        w1.append( (FitParameters_all[i]['w1_re']-FitParameters_all[-1]['w1_re'])  )
        w2.append( (FitParameters_all[i]['w2_re']-FitParameters_all[-1]['w2_re'])  )
        if g1 != 0 and no_of_QDs==2 and cavity==1:
            w3.append( (FitParameters_all[i]['w3_re']-FitParameters_all[-1]['w3_re'])  )
        gam1.append( (FitParameters_all[i]['w1_im']-FitParameters_all[-1]['w1_im'])  )
        gam2.append( (FitParameters_all[i]['w2_im']-FitParameters_all[-1]['w2_im'])   )
        if g1 != 0 and no_of_QDs==2 and cavity==1:
            gam3.append( (FitParameters_all[i]['w3_im']-FitParameters_all[-1]['w3_im'])  )
 
        
    fig1 = plt.figure( figsize=(4.5,3),dpi=150)
    bb = fig1.add_subplot(1, 1, 1)    
    # bb.plot(np.abs(tfit),np.abs(fit),'r-',linewidth='1.2',label='fit across all t')
    bb.plot(Ls[0:-1],np.abs(w1),'g-',label=r'$Re\, \omega_1$') 
    bb.plot(Ls[0:-1],np.abs(w2),'g--',label=r'$Re\, \omega_2$') 
    if g1!= 0 and no_of_QDs==2 and cavity==1:
        bb.plot(Ls[0:-1],np.abs(w3),'g:',label=r'$Re\, \omega_3$') 

    bb.plot(Ls[0:-1],np.abs(gam1),'b-',label=r'$Im\, \omega_1$') 
    bb.plot(Ls[0:-1],np.abs(gam2),'b--',label=r'$Im\, \omega_2$') 
    if g1!= 0 and no_of_QDs==2 and cavity==1:
        bb.plot(Ls[0:-1],np.abs(gam3),'b:',label=r'$Im\, \omega_3$') 
    plt.xticks(np.arange(Ls[0], Ls[-1], 2))

    plt.xticks(fontsize='10')
    plt.yticks(fontsize='10')
    bb.set_title(rf'$\epsilon=${threshold_str}, $g_1$={round(g1*hbar*1e3)}$\mu$eV, $g_2$={round(g2*hbar*1e3)}$\mu$eV, g={round(gd*hbar*1e3)}$\mu$eV, d={np.round(np.float64(r0),2)}nm')
    bb.set_ylabel('absolute error',fontsize='12')
    bb.set_xlabel('$L$ (neighbours)',fontsize='12')
    leg=bb.legend(fontsize=8, loc="best")
    leg.get_frame().set_color('none')
    leg.get_frame().set_alpha(0)
    plt.yscale('log')
    plt.tight_layout()   
       
######### Explicit paramaters plotting ############################
    w1s, w2s, w3s, gam1s, gam2s, gam3s, A1res, A2res, A3res, A1ims, A2ims, A3ims = [[] for _ in range(12)]
  
    for i in range(len(Ls)): 
        w1s.append(FitParameters_all[i]['w1_re']*hbar*1e3)
        w2s.append(FitParameters_all[i]['w2_re']*hbar*1e3)
        A1res.append(FitParameters_all[i]['A1_re'])
        A2res.append(FitParameters_all[i]['A2_re'])
        if g1!= 0 and no_of_QDs==2 and cavity==1:
            w3s.append(FitParameters_all[i]['w3_re']*hbar*1e3)
            A3res.append(FitParameters_all[i]['A3_re'])

        gam1s.append(np.abs(FitParameters_all[i]['w1_im']*hbar*1e3))
        gam2s.append(np.abs(FitParameters_all[i]['w2_im']*hbar*1e3))
        A1ims.append(FitParameters_all[i]['A1_im'])
        A2ims.append(FitParameters_all[i]['A2_im'])
        if g1!= 0 and no_of_QDs==2 and cavity==1:
            gam3s.append(np.abs(FitParameters_all[i]['w3_im']*hbar*1e3))
            A3ims.append(FitParameters_all[i]['A3_im'])

    
    fig1 = plt.figure(figsize=(8, 9), dpi=150)  # Increase figure height to accommodate additional plots
    
    # Upper left subplot for w1s
    ax1 = fig1.add_subplot(321)  # Changed from 221 to 321
    ax1.plot(Ls, w1s, 'r-')
    ax1.set_ylabel(r'$Re\, \omega_1$ ($\mu$eV)', fontsize=12)
    ax1.set_xlabel('$L$ (neighbours)', fontsize=12)
    ax1.set_yticklabels(['{:.2f}'.format(x) for x in ax1.get_yticks()])
    
    # Middle left subplot for w2s
    ax2 = fig1.add_subplot(323)  # New subplot
    ax2.plot(Ls, w2s, 'b-')
    ax2.set_ylabel(r'$Re\, \omega_2$ ($\mu$eV)', fontsize=12)
    ax2.set_xlabel('$L$ (neighbours)', fontsize=12)
    
    if g1!= 0 and no_of_QDs==2 and cavity==1:
        # Bottom left subplot for w3s
        ax3 = fig1.add_subplot(325)  # New subplot
        ax3.plot(Ls, w3s, 'g-')  # Assuming w3s is defined
        ax3.set_ylabel(r'$Re\, \omega_3$ ($\mu$eV)', fontsize=12)
        ax3.set_xlabel('$L$ (neighbours)', fontsize=12)
    
    # Upper right subplot for gam1s
    ax4 = fig1.add_subplot(322)  # Changed from 222 to 322
    ax4.plot(Ls, gam1s, 'r-')
    ax4.set_ylabel(r'$Im\, \omega_1$ ($\mu$eV)', fontsize=12)
    ax4.set_xlabel('$L$ (neighbours)', fontsize=12)
    
    # Middle right subplot for gam2s
    ax5 = fig1.add_subplot(324)  # New subplot
    ax5.plot(Ls, gam2s, 'b-')
    ax5.set_ylabel(r'$Im\, \omega_2$ ($\mu$eV)', fontsize=12)
    ax5.set_xlabel('$L$ (neighbours)', fontsize=12)
    
    if g1!= 0 and no_of_QDs==2 and cavity==1:
        # Bottom right subplot for gam3s
        ax6 = fig1.add_subplot(326)  # New subplot
        ax6.plot(Ls, gam3s, 'g-')  # Assuming gam3s is defined
        ax6.set_ylabel(r'$Im\, \omega_3$ ($\mu$eV)', fontsize=12)
        ax6.set_xlabel('$L$ (neighbours)', fontsize=12)
        
    plt.suptitle(rf'$\epsilon=${threshold_str}, $g_1$={round(g1*hbar*1e3)}$\mu$eV, $g_2$={round(g2*hbar*1e3)}$\mu$eV, g={round(gd*hbar*1e3)}$\mu$eV, d={np.round(np.float64(r0),2)}nm', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
            
 #%%
    from scipy.optimize import curve_fit
# Power law model
    def power_law_model(L, omega_infinity, C, beta):
        return omega_infinity + C * L**-beta
    def exponential_model(L, omega_infinity, C, alpha):
        return omega_infinity + C * np.exp(-alpha * L)
    def PL_2beta(L,omega_infinity,C):
        return omega_infinity + C * np.power(L.astype(float), -2)

    
    def PL_fit(ws):
      ## Define the range you want to select
        # inf_vals_PL=[]
        # inf_vals_exp=[]
        # Cs_PL=[]
        # betas_PL=[]
        
        # selected_range = slice(0, None)  # Change the slice values as needed, e.g., slice(0, 15)
        # initial_guess_pl = [ws[-1],1,1]
        # params_pl, pcov = curve_fit(power_law_model, Ls[selected_range], np.real(ws)[selected_range], p0=initial_guess_pl)
        # omega_inf_pl, C_pl, beta_pl = params_pl
        # # print(pcov)
        # inf_err = np.sqrt(pcov[0, 0])
        # inf_vals_PL.append(omega_inf_pl)
        # Cs_PL.append(C_pl)
        # betas_PL.append(beta_pl)
        # L_ex=np.linspace(Ls[0],Ls[-1],50)
        # # omega_exp_fit = exponential_model(L_ex, *params_exp)
        # omega_pl_fit = power_law_model(L_ex, *params_pl)
        # return omega_inf_pl, C_pl, beta_pl, omega_pl_fit, inf_err
    # Define the range you want to select
       inf_vals_PL=[]
       Cs_PL=[]
       selected_range = slice(np.where(Ls==Ls[-3])[0][0], None)  # Change the slice values as needed, e.g., slice(0, 15)
       # selected_range = slice(0, None)  # Change the slice values as needed, e.g., slice(0, 15)
       initial_guess_pl = [ws[-1],1]
       params_pl, pcov = curve_fit(PL_2beta, Ls[selected_range], np.real(ws)[selected_range], p0=initial_guess_pl)
       omega_inf_pl, C_pl = params_pl
       # print(pcov)
       inf_err = np.sqrt(pcov[0, 0])
       inf_vals_PL.append(omega_inf_pl)
       Cs_PL.append(C_pl)
       L_ex=np.linspace(Ls[0],Ls[-1],50)
       omega_pl_fit = PL_2beta(L_ex, *params_pl)
       return omega_inf_pl, C_pl, omega_pl_fit, inf_err
        
        
    # print(w_infinity)
    L_ex=np.linspace(Ls[0],Ls[-1],50)
    w1_inf,w1_C,w1_beta, omega_pl_fit_w1, w1_inf_err=PL_fit(w1s)
    w2_inf,w2_C,w2_beta, omega_pl_fit_w2, w2_inf_err=PL_fit(w2s)
    w3_inf,w3_C,w3_beta, omega_pl_fit_w3, w3_inf_err=PL_fit(w3s)

    # w_inf, C, beta unknown
    gam1_inf,gam1_C,gam1_beta, omega_pl_fit_gam1, gam1_inf_err=PL_fit(gam1s)
    gam2_inf,gam2_C,gam2_beta, omega_pl_fit_gam2, gam2_inf_err=PL_fit(gam2s)
    gam3_inf,gam3_C,gam3_beta, omega_pl_fit_gam3, gam3_inf_err=PL_fit(gam3s)
    
    #fixing beta=2
    gam1_inf,gam1_C, omega_pl_fit_gam1, gam1_inf_err=PL_fit(gam1s)
    gam2_inf,gam2_C, omega_pl_fit_gam2, gam2_inf_err=PL_fit(gam2s)
    gam3_inf,gam3_C, omega_pl_fit_gam3, gam3_inf_err=PL_fit(gam3s)
    print(gam1_inf)
    print(gam2_inf)
    print(gam3_inf)
    w1_complex=w1_inf -1j*gam1_inf
    

    plt.figure()
    plt.plot(Ls, w1s,'bo', label='Data w1')   
    plt.plot(L_ex, omega_pl_fit_w1, label='Power Law Fit', color='red')
    plt.axhline(y=w1_inf,color='r',linestyle='--', label='$\omega(\infty)$')
    plt.legend(loc='best')
    plt.figure()
    plt.plot(Ls, w2s,'bo', label='Data w2')   
    plt.plot(L_ex, omega_pl_fit_w2, label='Power Law Fit', color='red')
    plt.axhline(y=w2_inf,color='r',linestyle='--', label='$\omega(\infty)$')
    plt.legend(loc='best')
    plt.figure()
    plt.plot(Ls, w3s,'bo', label='Data w3')   
    plt.plot(L_ex, omega_pl_fit_w3, label='Power Law Fit', color='red')
    plt.axhline(y=w3_inf,color='r',linestyle='--', label='$\omega(\infty)$')
    plt.legend(loc='best')
    
    plt.figure()
    plt.plot(Ls, gam1s,'bo', label='Data gam1')   
    plt.plot(L_ex, omega_pl_fit_gam1, label=fr'Power Law Fit: $\beta=1$, $C={np.round(gam1_C,2)}$', color='red')
    plt.axhline(y=gam1_inf,color='r',linestyle='--', label=fr'$\omega(\infty)= {np.round(gam1_inf,2)}$')
    plt.legend(loc='best')
       
    plt.figure()
    plt.plot(Ls, gam2s,'bo', label='Data gam2')   
    plt.plot(L_ex, omega_pl_fit_gam2, label=fr'Power Law Fit: $\beta=2$, $C={np.round(gam2_C,2)}$', color='red')
    plt.axhline(y=gam2_inf,color='r',linestyle='--', label=fr'$\omega(\infty)= {np.round(gam2_inf,2)}$')
    plt.legend(loc='best')
    
    plt.figure()
    plt.plot(Ls, gam3s,'bo', label='Data gam3')   
    plt.plot(L_ex, omega_pl_fit_gam3, label=fr'Power Law Fit: $\beta=2$, $C={np.round(gam3_C,2)}$', color='red')
    plt.axhline(y=gam3_inf,color='r',linestyle='--', label=fr'$\omega(\infty)= {np.round(gam3_inf,2)}$')
    plt.legend(loc='best')

    w1_errors=np.abs(w1s-w1_inf)
    w2_errors=np.abs(w2s-w2_inf)
    w3_errors=np.abs(w3s-w3_inf)

    gam1_errors=np.abs(gam1s-gam1_inf)
    gam2_errors=np.abs(gam2s-gam2_inf)
    gam3_errors=np.abs(gam3s-gam3_inf)


    plt.figure()
    plt.plot(Ls,w1_errors,'ro',label=r'$\omega_1$ error')
    plt.plot(Ls,w2_errors,'bo',label=r'$\omega_2$ error')
    plt.plot(Ls,w3_errors,'go',label=r'$\omega_3$ error')
    L_ex=np.linspace(Ls[0],Ls[-1],50)
    plt.plot(L_ex, np.abs(omega_pl_fit_w2-w2_inf),'b--',label=rf'$|{round(w2_C, 1)}L^{{-{round(w2_beta, 1)}}}|$')
    plt.plot(L_ex, np.abs(omega_pl_fit_w1-w1_inf),'r--',label=rf'$|{round(w1_C, 1)}L^{{-{round(w1_beta, 1)}}}|$')
    plt.plot(L_ex, np.abs(omega_pl_fit_w3-w3_inf),'g--',label=rf'$|{round(w3_C, 1)}L^{{-{round(w3_beta, 1)}}}|$')
    plt.ylabel(r'$|\omega(L)-\omega_\infty|$', fontsize='14')
    plt.xlabel('L', fontsize='14')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='best')
    plt.tight_layout()
 
    plt.figure()
    plt.plot(Ls,gam1_errors,'ro',label=r'$\Gamma_1$ error')
    plt.plot(Ls,gam2_errors,'bo',label=r'$\Gamma_2$ error')
    plt.plot(Ls,gam3_errors,'go',label=r'$\Gamma_3$ error')
    L_ex=np.linspace(Ls[0],Ls[-1],50)
    plt.plot(L_ex, np.abs(omega_pl_fit_gam2-gam2_inf),'b--',label=rf'$|{round(gam2_C, 1)}L^{{-{round(gam2_beta, 1)}}}|$')
    plt.plot(L_ex, np.abs(omega_pl_fit_gam1-gam1_inf),'r--',label=rf'$|{round(gam1_C, 1)}L^{{-{round(gam1_beta, 1)}}}|$')
    plt.plot(L_ex, np.abs(omega_pl_fit_gam3-gam3_inf),'g--',label=rf'$|{round(gam3_C, 1)}L^{{-{round(gam3_beta, 1)}}}|$')
    plt.ylabel(r'$|\Gamma(L)-\Gamma_\infty|$', fontsize='14')
    plt.xlabel('L', fontsize='14')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='best')
    plt.tight_layout()
    
########## NR method ###############
    w1s, w2s, w3s, gam1s, gam2s, gam3s, A1res, A2res, A3res, A1ims, A2ims, A3ims = [[] for _ in range(12)]
  
    for i in range(len(Ls)): 
        w1s.append(FitParameters_all[i]['w1_re']*hbar*1e3)
        w2s.append(FitParameters_all[i]['w2_re']*hbar*1e3)
        A1res.append(FitParameters_all[i]['A1_re'])
        A2res.append(FitParameters_all[i]['A2_re'])
        if g1!= 0 and no_of_QDs==2 and cavity==1:
            w3s.append(FitParameters_all[i]['w3_re']*hbar*1e3)
            A3res.append(FitParameters_all[i]['A3_re'])

        gam1s.append(np.abs(FitParameters_all[i]['w1_im']*hbar*1e3))
        gam2s.append(np.abs(FitParameters_all[i]['w2_im']*hbar*1e3))
        A1ims.append(FitParameters_all[i]['A1_im'])
        A2ims.append(FitParameters_all[i]['A2_im'])
        if g1!= 0 and no_of_QDs==2 and cavity==1:
            gam3s.append(np.abs(FitParameters_all[i]['w3_im']*hbar*1e3))
            A3ims.append(FitParameters_all[i]['A3_im'])

    
    from scipy.optimize import newton
    w1s_complex = np.array(w1s, dtype=np.float64).astype(np.complex128)
    w2s_complex = np.array(w2s, dtype=np.float64).astype(np.complex128) 
    gam1s_complex = np.array(gam1s, dtype=np.float64).astype(np.complex128)
    gam2s_complex = np.array(gam2s, dtype=np.float64).astype(np.complex128) 
    A1sre_complex = np.array(A1res, dtype=np.float64).astype(np.complex128)
    A1sim_complex = np.array(A1ims, dtype=np.float64).astype(np.complex128)
    A2sre_complex = np.array(A2res, dtype=np.float64).astype(np.complex128)
    A2sim_complex = np.array(A2ims, dtype=np.float64).astype(np.complex128)
    try:
        w3s_complex = np.array(w3s, dtype=np.float64).astype(np.complex128)
        gam3s_complex = np.array(gam3s, dtype=np.float64).astype(np.complex128)
        A3sre_complex = np.array(A3res, dtype=np.float64).astype(np.complex128)
        A3sim_complex = np.array(A3ims, dtype=np.float64).astype(np.complex128)
    except:
        pass
    
    w1s_c=w1s_complex-1j*gam1s_complex
    w2s_c=w2s_complex-1j*gam2s_complex
    A1s_c=A1sre_complex-1j*A1sim_complex
    A2s_c=A2sre_complex-1j*A2sim_complex
    try:
        w3s_c=w3s_complex-1j*gam3s_complex
        A3s_c=A3sre_complex-1j*A3sim_complex
    except:
        pass
    
    
    def NR(ws,Ls, selected_range): 
        ws=ws[selected_range]
        Ls=Ls[selected_range]
        w_L3L2=ws[-3]-ws[-2]
        w_L1L3=ws[-1]-ws[-3]
        w_L2L1=ws[-2]-ws[-1]
        def f_alpha(alpha):
            return w_L3L2*Ls[-1]**-alpha + w_L1L3*Ls[-2]**-alpha + w_L2L1*Ls[-3]**-alpha
        def f_alpha_prime(alpha):
            return -( w_L3L2*Ls[-1]**-alpha * np.log(Ls[-1]) + w_L1L3*Ls[-2]**-alpha * np.log(Ls[-2]) + w_L2L1*Ls[-3]**-alpha * np.log(Ls[-3]) )
        initial_guess=2-0j
        root = newton(f_alpha, x0=initial_guess, fprime=f_alpha_prime, maxiter=500)
        C= (ws[-1]-ws[-2])/(Ls[-1]**-root - Ls[-2]**-root)
        w_infty=ws[-1] - C*Ls[-1]**-root
        return w_infty,C,root
    
    def params_NR(ws_c): 
        w_inftys=[]
        C_NRs=[]
        alpha_NR=[]
        Lvals=[]

        # for i in range(6,len(Ls)-3):
        #     print(i)
        selected_range = slice(np.where(Ls==Ls[-3])[0][0], np.where(Ls==Ls[-3])[0][0] +3) 
        print(Ls[selected_range])
        Lvals.append(Ls[selected_range][1])
        w_infty,C,alpha=NR(ws_c,Ls, selected_range)
        w_inftys.append(w_infty)
        C_NRs.append(C)
        alpha_NR.append(alpha) 
        
        return w_inftys,C_NRs,alpha_NR,Lvals
    
    w_1_inftys,C1_NRs,alpha1_NR,Lvals= params_NR(w1s_c)
    w_2_inftys,C2_NRs,alpha2_NR,Lvals= params_NR(w2s_c)
    # A1_inftys,CA1_NR,alphaA1_NR,Lvals= params_NR(A1s_c)
    # A2_inftys,CA2_NR,alphaA2_NR,Lvals= params_NR(A2s_c)
    try:
        w_3_inftys,C3_NRs,alpha3_NR,Lvals= params_NR(w3s_c)
        # A3_inftys,CA3_NR,alphaA3_NR,Lvals= params_NR(A3s_c)
    except:
        pass
    
    g1_inftys,C1_NRs,alpha1_NR,Lvals= params_NR(gam1s)
    g2_inftys,C2_NRs,alpha2_NR,Lvals= params_NR(gam2s)
    g3_inftys,C2_NRs,alpha2_NR,Lvals= params_NR(gam3s)
    
    print(w_1_inftys)
    print(w_2_inftys)
    print(w_3_inftys)
    
    print(gam1_inf)
    print(gam2_inf)
    print(gam3_inf)
    print(g1_inftys)
    print(g2_inftys)
    print(g3_inftys)
    
    plt.plot(Ls,gam1s,'ro',label='data gam1')
    plt.plot(Ls,power_law_model(Ls, g1_inftys[0],C1_NRs[0],alpha1_NR[0]), 'r--', label='PL NR fit')
    plt.plot(Ls,power_law_model(Ls,  gam1_inf,gam1_C,gam1_beta), 'b--', label= f'PL fit beta={np.round(gam1_beta,2)}, inf_val={np.round(gam1_inf,2)}')
    plt.legend(loc='best')
    
    plt.plot(Ls,gam1s,'ro',label='data gam1')
    plt.plot(Ls,power_law_model(Ls, -np.imag(w_1_inftys[0]),-np.imag(C1_NRs[0]),(alpha1_NR[0])), 'r--', label='PL NR fit')
    plt.plot(Ls,power_law_model(Ls,  gam1_inf,gam1_C,gam1_beta), 'b--', label='PL fit ')
    plt.legend(loc='best')
    
    plt.plot(Ls,gam2s,'ro',label='data gam2')
    plt.plot(Ls,power_law_model(Ls, g2_inftys[0],C2_NRs[0],alpha2_NR[0]), 'r--', label='PL NR fit')
    plt.plot(Ls,power_law_model(Ls,  gam2_inf,gam2_C,gam2_beta), 'b--', label=f'PL fit beta={np.round(gam2_beta,2)}, inf_val={np.round(gam2_inf,2)}')
    plt.legend(loc='best')
    
    plt.plot(Ls,gam2s,'ro',label='data gam2')
    plt.plot(Ls,power_law_model(Ls, -np.imag(w_2_inftys[0]),-np.imag(C2_NRs[0]),np.real(alpha2_NR[0])), 'r--', label='PL NR fit')
    plt.plot(Ls,power_law_model(Ls,  gam2_inf,gam2_C,gam2_beta), 'b--', label='PL fit ')
    plt.legend(loc='best')
    
    plt.plot(Ls,gam3s,'ro',label='data gam3')
    # plt.plot(Ls,power_law_model(Ls, g3_inftys[0],C2_NRs[0],alpha2_NR[0]), 'r--', label='PL NR fit')
    plt.plot(Ls,power_law_model(Ls,  gam3_inf,gam3_C,gam3_beta), 'b--', label=f'PL fit beta={np.round(gam3_beta,2)}, inf_val={np.round(gam3_inf,2)}')
    plt.legend(loc='best')
    
    # plt.plot(Lvals,w_1_inftys)
    # plt.plot(Lvals,w_2_inftys)
    # plt.plot(Lvals,A1_inftys)
    # plt.plot(Lvals,A2_inftys)
    
    
    ## fixing alpha or beta = 2, EM notes pages 2+
    def NR_fixed(Ls,ws, selected_range):

        N=len(Ls[selected_range])
        ws=ws[selected_range]
        Ls=Ls[selected_range]
        omega1 = np.sum(ws)
        omega2 = np.sum(ws[i] / (Ls[i]**2) for i in range(len(Ls)))
        S_1 = np.sum(1/Ls[i]**2 for i in range(len(Ls)))
        S_2 = np.sum(1/Ls[i]**4 for i in range(len(Ls)))
    
        ws_inf_NRfix=(omega2 *S_1 - omega1*S_2)/(S_1**2 - S_2 *N)
        C=(omega1 *S_1 - omega2*N)/(S_1**2 - S_2 *N)
        return ws_inf_NRfix,C
    
    # selected_range = slice(np.where(Ls==Ls[-3])[0][0], np.where(Ls==Ls[-3])[0][0] +3) 
    selected_range = slice(9,None) 
    print(Ls[selected_range])
    gam1_NRfix,C1=NR_fixed(Ls,gam1s, selected_range)
    gam2_NRfix,C2=NR_fixed(Ls,gam2s, selected_range)
    print('PL fit:', gam1_inf)
    print('NR fixed beta fit', gam1_NRfix)
    print('PL fit 2:', gam2_inf)
    print('NR fixed beta fit 2', gam2_NRfix)
    
    # selected_range = slice(np.where(Ls==Ls[-3])[0][0], np.where(Ls==Ls[-3])[0][0] +3) 
    selected_range = slice(0,None) 
    w1sc_NRfix,C1=NR_fixed(Ls,w1s_c, selected_range)
    w2sc_NRfix,C2=NR_fixed(Ls,w2s_c, selected_range)
    plt.plot(Ls,w1s,'ro',label='data w1')
    plt.plot(Ls, PL_2beta(Ls, np.real(w1sc_NRfix),np.real(C1)), 'r--', label='PL NR fit')
    plt.legend(loc='best')
    
    plt.plot(Ls,gam1s,'ro',label='data gam1')
    plt.plot(Ls, PL_2beta(Ls, gam1_NRfix,C1), 'r--', label='PL NR fit')
    plt.legend(loc='best')
   
    plt.plot(Ls,gam2s,'ro',label='data gam2')
    plt.plot(Ls, PL_2beta(Ls, gam2_NRfix,C2), 'r--', label='PL NR fit')
    plt.legend(loc='best')
#%%  #!!!
#Writing code to find extrapolated parameters and then compare that with analytics w/ FGR and analytics w/ FGR modified
    Ls=np.arange(22,26+1,2)
    r0s=np.arange(0.01,62.01,2)    # Ls=np.append(Ls,28)
    # r0s=np.array([30.01])
    # threshold_factor=1e-6
    # threshold_str = str(threshold_factor)
    FitParameters_all=[] 
    errors=[]
    fit_errors=[]
    w1s_complex=[]
    w2s_complex=[]
    w3s_complex=[]
    A1s_complex=[]
    A2s_complex=[]
    A3s_complex=[]
    ws_plus=[]
    ws_neg=[]
    ws_plus_mod=[]
    ws_neg_mod=[]
    As_plus=[]
    As_neg=[]
    As_plus_mod=[]
    As_neg_mod=[]
   

    for r in r0s:
        r0=r
        FitParameters_all=[] 
        w1s, w2s, w3s, gam1s, gam2s, gam3s, A1res, A2res, A3res, A1ims, A2ims, A3ims = [[] for _ in range(12)]
        for L in Ls:
            ###params that change with L ###            
            t0=r0/Vs
            if r0< 5.01:
                factortau=1.4
            if r0 > 5.01:
                factortau=1.0
            if sharebath ==1:
                dt=( t0 + factortau*tauib )/(L+1)
            if sharebath ==0:
                dt=2*tauib / (L+1)
            
            j0_1=j0*Vs/r0 
                
            step_no = int(tfinal/dt)
            length = L/2
            length_U=L/2
            n_perms=int(d**((L/2)))
            n_perms_U=int(d**((L/2)))
            Lcycle=L
     
            LF=LFpol_qdqdcav(g1, g2,gd, w_qd1, w_qd2, w_c)
            M1=expm(-1j*LF*dt)
    
            # print('L=', L,'d=',np.round(r0,3), 'dt=',np.round(dt,3))
            ### ##########generate data ###
            try:
                P=np.load(path+"/data/"+params.label(L=L, r0=r0, factortau=factortau)+".npy",allow_pickle=True)
            except:
                params_cumulant = {
                    'r0': r0,
                    'j0': j0,
                    'j0_1': j0_1,
                    'w0': w0,
                    'T': T,
                    'Vs': Vs,
                    'l': l,
                    'dotshape': dotshape,
                    'sharebath': 1,
                    'lp': lp}  
                
                cumulants = Cumulants()
                cumulants.update_parameters(**params_cumulant)
                cumulants, cumulants_inin = cumulants.cu(L, dt, **params_cumulant)
                Q0=np.array([[M1[0,0]*np.exp(cumulants_inin[0] +2*cumulants_inin[1]), M1[0,1]*np.exp(cumulants_inin[0] +2*cumulants[1]), M1[0,2] ],
                              [M1[1,0]*np.exp(cumulants_inin[0] +2*cumulants[1]), M1[1,1]*np.exp(cumulants_inin[0] +2*cumulants_inin[1]), M1[1,2] ],
                              [M1[2,0]*np.exp(cumulants_inin[0]), M1[2,1]*np.exp(cumulants_inin[0]), M1[2,2]]])
                Qlist=[]
                Qlist.append(Q0)
                for i in range(int(L-1)):
                    Qlist.append(np.array([[np.exp(2*cumulants_inin[i+2]), np.exp(2*cumulants[i+2]), 1 ],
                                            [np.exp(2*cumulants[i+2]), np.exp(2*cumulants_inin[i+2]), 1],
                                            [1, 1, 1]]))
                exp_k0_factor=np.exp(cumulants_inin[0])
                P= get_nth_permutation(L, Lcycle, r0,dt,length, n_perms, n_perms_U, step_no, M1, exp_k0_factor, Qlist)
                np.save(path+"/data/"+params.label(L=L, r0=r0,factortau=factortau)+".npy",P)
            P=np.ravel(P)
            times = np.array([dt*i for i in range(step_no+2)])
            # times=times[2:]
            
            tmlongt, Pnlongt, fit_sametimes = fitprocedure(g,g1,g2,gd,r0,t0,P,times)
    
    
    
        
      
        for i in range(len(Ls)): 
            w1s.append(FitParameters_all[i]['w1_re']*hbar*1e3)
            w2s.append(FitParameters_all[i]['w2_re']*hbar*1e3)
            A1res.append(FitParameters_all[i]['A1_re'])
            A2res.append(FitParameters_all[i]['A2_re'])
            if g1!= 0:
                w3s.append(FitParameters_all[i]['w3_re']*hbar*1e3)
                A3res.append(FitParameters_all[i]['A3_re'])

            gam1s.append(np.abs(FitParameters_all[i]['w1_im']*hbar*1e3))
            gam2s.append(np.abs(FitParameters_all[i]['w2_im']*hbar*1e3))
            A1ims.append(FitParameters_all[i]['A1_im'])
            A2ims.append(FitParameters_all[i]['A2_im'])
            if g1!= 0:
                gam3s.append(np.abs(FitParameters_all[i]['w3_im']*hbar*1e3))
                A3ims.append(FitParameters_all[i]['A3_im'])

        
        # if r0 < 6.01:
        #     w1_inf=w1s[-1]
        #     w2_inf=w2s[-1]
        #     w3_inf=w3s[-1]
        #     gam1_inf=gam1s[-1]
        #     gam2_inf=gam2s[-1]
        #     gam3_inf=gam3s[-1]
        #     A1re_inf=A1res[-1]
        #     A2re_inf=A2res[-1]
        #     A3re_inf=A3res[-1]
        #     A1im_inf=A1ims[-1]
        #     A2im_inf=A2ims[-1]
        #     A3im_inf=A3ims[-1]
            
        # if r0 > 6.01:    
        L_ex=np.linspace(Ls[0],Ls[-1],50)
        try:
            w1_inf,w1_C, omega_pl_fit_w1, w1_inf_err=PL_fit(w1s)
        except:
            w1_inf=w1s[-1]
            
        try:
            w2_inf,w2_C, omega_pl_fit_w2, w2_inf_err=PL_fit(w2s)
        except:
            w2_inf=w2s[-1]
            
        if g1!=0:
            try:
                w3_inf,w3_C, omega_pl_fit_w3, w3_inf_err=PL_fit(w3s)
            except:
                
                w3_inf=w3s[-1]
        try: 
            gam1_inf,gam1_C, omega_pl_fit_gam1, gam1_inf_err=PL_fit(gam1s)
        except:
            print('gam1 failed PL fit at d=', np.round(r0,2), 'nm')
            gam1_inf=gam1s[-1]
        try:
            gam2_inf,gam2_C, omega_pl_fit_gam2, gam2_inf_err=PL_fit(gam2s)
        except:
            print('gam2 failed PL fit at d=', np.round(r0,2), 'nm')

            gam2_inf=gam2s[-1]
        if g1!=0:
            try:
                gam3_inf,gam3_C, omega_pl_fit_gam3, gam3_inf_err=PL_fit(gam3s)
            except:
                gam3_inf=gam3s[-1]
                print('gam3 failed PL fit at d=', np.round(r0,2), 'nm')

        try:
            A1re_inf,A1re_C, omega_pl_fit_A1re, A1re_inf_err=PL_fit(A1res)
        except:
            A1re_inf=A1res[-1]
           
        try:
            A2re_inf,A2re_C, omega_pl_fit_A2re, A2re_inf_err=PL_fit(A2res)
        except:
            A2re_inf=A2res[-1]
            
        if g1!=0:
            try:
                A3re_inf,A3re_C, omega_pl_fit_A3re, A3re_inf_err=PL_fit(A3res)
            except:
                A3re_inf=A3res[-1]
               
        try: 
            A1im_inf, A1im_C, omega_pl_fit_A1im, A1im_inf_err=PL_fit(A1ims)
        except:
            A1im_inf=A1ims[-1]
        
        try:
            A2im_inf, A2im_C, omega_pl_fit_A2im, A2im_inf_err=PL_fit(A2ims)
        except:
            A2im_inf=A2ims[-1]
         
        if g1 !=0:
            try:
                A3im_inf, A3im_C, omega_pl_fit_A3im, A3im_inf_err=PL_fit(A3ims)
            except:
                A3im_inf=A3ims[-1]
            
        # w1s, w2s, w3s, gam1s, gam2s, gam3s, A1res, A2res, A3res, A1ims, A2ims, A3ims = [[] for _ in range(12)]
        

        w1_complex=w1_inf -1j*gam1_inf
        w2_complex=w2_inf -1j*gam2_inf
        A1_complex=A1re_inf -1j*A1im_inf
        A2_complex=A2re_inf -1j*A2im_inf
        if g1 !=0:
            A3_complex=A3re_inf -1j*A3im_inf
            w3_complex=w3_inf -1j*gam3_inf
        
        # Sinin= S_inin(T,j0,w0)
        # omp=PolaronShift(j0,w0)    
        # R= np.sqrt((detuning**2 + 4*gd**2)) 
        # FGR_plus,FGR_neg=analytics_bareg(j0, j0_FGR, l, Vs, w0, T, r0, detuning, gd, 0, 0, 0, 0)
        # w_plus=hbar*1e3*((w_qd1+w_qd2)/2  +omp +R/2) -1j*FGR_plus
        # w_neg=hbar*1e3*((w_qd1+w_qd2)/2  +omp - R/2) -1j*FGR_neg
        # Dp_sq=0.5*(1+(detuning/R))
        # Dm_sq=0.5*(1-(detuning/R))
        # A_plus=np.exp(-Sinin)*Dp_sq
        # A_neg=np.exp(-Sinin)*Dm_sq
        
        # Sinin= S_inin(T,j0,w0)      
        # j0_1=j0*Vs/r0 
        # Sinim= S_inim(T, j0_1, w0, r0, Vs)
        # DeltaS=np.exp(-(Sinin-Sinim))
        # FGR_plus_mod,FGR_neg_mod=analytics_modified(j0, j0_FGR, l, Vs, w0, T, r0, detuning, gd, 0, 0, 0,0)
        # R_mod= np.sqrt((detuning**2 + 4*(gd*DeltaS)**2))
        # w_plus_mod=hbar*1e3*((w_qd1+w_qd2)/2  +omp +R_mod/2) -1j*FGR_plus_mod
        # w_neg_mod=hbar*1e3*((w_qd1+w_qd2)/2  +omp - R_mod/2) -1j*FGR_neg_mod
        # Dp_sq_mod=0.5*(1+(detuning/R_mod))
        # Dm_sq_mod=0.5*(1-(detuning/R_mod))
        # A_plus_mod=np.exp(-Sinin)*Dp_sq_mod
        # A_neg_mod=np.exp(-Sinin)*Dm_sq_mod
    
        w1s_complex.append(w1_complex)
        w2s_complex.append(w2_complex)
        A1s_complex.append(A1_complex)
        A2s_complex.append(A2_complex)
        if g1 !=0:
            A3s_complex.append(A3_complex)
            w3s_complex.append(w3_complex)
        # ws_plus.append(w_plus)
        # ws_neg.append(w_neg)
        # ws_plus_mod.append(w_plus_mod)
        # ws_neg_mod.append(w_neg_mod)
        # As_plus.append(A_plus)
        # As_neg.append(A_neg)
        # As_plus_mod.append(A_plus_mod)
        # As_neg_mod.append(A_neg_mod)


    #DEPHASING PART
    if g1==0:
        plt.figure(2)
        plt.xlabel('distance between qubits, d (nm)', fontsize='12')
        plt.ylabel(r'dephasing rate, $\Gamma$ ($\mu$ eV)', fontsize='12')
        plt.plot(r0s,np.abs(np.imag(w1s_complex)), 'b-', label='full calc.')
        plt.plot(r0s,np.abs(np.imag(w2s_complex)), 'r-')
        plt.tight_layout()
        plt.legend(loc='best')
        
        from scipy.interpolate import interp1d
        from scipy.signal import savgol_filter
        interpolation_function = interp1d(r0s, -np.imag(w1s_complex), kind='cubic')
        r0s_new = np.linspace(r0s.min(), r0s.max(), 400)  # 200 points between min and max of gs
        w1_ext_imag_interpolated = interpolation_function(r0s_new)
        w1_ext_imag_smooth = savgol_filter(w1_ext_imag_interpolated, window_length=11, polyorder=3)
        interpolation_function = interp1d(r0s, -np.imag(w2s_complex), kind='cubic')
        w2_ext_imag_interpolated = interpolation_function(r0s_new)
        w2_ext_imag_smooth = savgol_filter(w2_ext_imag_interpolated, window_length=11, polyorder=3)
        # Plot the results 
        fig1 = plt.figure( figsize=(4.5,3),dpi=150)
        bb = fig1.add_subplot(1, 1, 1)  
        bb.plot(r0s_new, w1_ext_imag_smooth, 'b-', label=r'$\Gamma_2$')  #swapping because labelling not right, w1 should be lower level
        bb.plot(r0s_new, w2_ext_imag_smooth, 'r-', label=r'$\Gamma_1$')
        bb.set_xlabel('distance between qubits, $d$ (nm)', fontsize='12')
        bb.set_ylabel('line broadening, $\Gamma$ ($\mu$ eV)', fontsize='12')
        plt.legend(loc=(0.75, 0.7), framealpha=0.0)
        plt.tight_layout()
        
        if dotshape=='spherical':
            omp=PolaronShift(j0,w0)    
            R= np.sqrt((detuning**2 + 4*gd**2)) 
            FGR_w1=[]
            FGR_w2=[]
            for r0 in r0s_new:
                FGR_plus,FGR_neg=analytics_bareg(j0, j0_FGR, l, Vs, w0, T, r0, detuning, gd, 0, 0, 0, 0)
                FGR_w1.append(hbar*1e3*((w_qd1+w_qd2)/2  +omp +R/2) -1j*FGR_plus)
                FGR_w2.append(hbar*1e3*((w_qd1+w_qd2)/2  +omp - R/2) -1j*FGR_neg)
        if dotshape=='smartie':
            FGR_w1=[]
            FGR_w2=[]
            for r0 in r0s_new:
                gam1,gam2=QDQD_analytics_smartie(gd,detuning,l,lp,Vs,r0,T,DvDc)
                FGR_w1.append(gam1)
                FGR_w2.append(gam2)
        fig1 = plt.figure( figsize=(4.5,3),dpi=150)
        bb = fig1.add_subplot(1, 1, 1)  
        bb.plot(r0s_new, w1_ext_imag_smooth, 'b-', label=r'$\Gamma_2$')  #swapping because labelling not right, w1 should be lower level
        bb.plot(r0s_new, w2_ext_imag_smooth, 'r-', label=r'$\Gamma_1$')
        if dotshape=='spherical':
            bb.plot(r0s_new, -np.imag(FGR_w1), 'b--', label=r'$\Gamma_2 $ FGR')  #swapping because labelling not right, w1 should be lower level
            bb.plot(r0s_new, -np.imag(FGR_w2), 'r--', label=r'$\Gamma_1$ FGR')
        if dotshape=='smartie':
            bb.plot(r0s_new, FGR_w1, 'b--', label=r'$\Gamma_2 $ FGR')  #swapping because labelling not right, w1 should be lower level
            bb.plot(r0s_new, FGR_w2, 'r--', label=r'$\Gamma_1$ FGR')
        bb.set_xlabel('distance between qubits, $d$ (nm)', fontsize='12')
        bb.set_ylabel('line broadening, $\Gamma$ ($\mu$ eV)', fontsize='12')
        plt.legend(loc='best')
        plt.tight_layout()
    
    
    
        length = len(r0s_new)  # Or any other list with the same length
        # Extend gam1_indep and gam2_indep to match the length
        # gam1_indep_extended = [gam1_indep] * length
        # gam2_indep_extended = [gam2_indep] * length
    
        # import csv
        # from itertools import zip_longest
        # with open("QDQD_Anisotropic_Dephasing.csv", "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Distances", "gam1","gam2", "FGR_gam1", "FGR_gam2","gam1_indep","gam2_indep"])  # Write header
        #     for row in zip_longest(r0s_new, w1_ext_imag_smooth, w2_ext_imag_smooth, np.real(FGR_w1), np.real(FGR_w2), gam1_indep_extended, gam2_indep_extended, fillvalue=np.nan):
        #         writer.writerow(row)
    if g1 !=0:
        plt.figure(2)
        plt.title(f'thresh={threshold_str}, factortau={factortau} ')
        plt.xlabel('distance between qubits, (nm)', fontsize='12')
        plt.ylabel(r'dephasing rate, ($\mu$ eV)', fontsize='12')
        plt.plot(r0s,np.abs(np.imag(w1s_complex)), 'r-', label='gam1')
        plt.plot(r0s,np.abs(np.imag(w2s_complex)), 'b-',label='gam2')
        plt.plot(r0s,np.abs(np.imag(w3s_complex)), 'g-',label='gam3')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.legend(loc='best')
        
        
        #want to remove problematic data point
        r0s = np.array(r0s)
        w1s_complex = np.array(w1s_complex)
        w2s_complex = np.array(w2s_complex)
        w3s_complex = np.array(w3s_complex)
        remove_indices = np.where(np.round(r0s, 2) == 48.01)[0]
        # Step 2: Create a mask for elements to keep
        keep_mask = np.ones_like(r0s, dtype=bool)
        keep_mask[remove_indices] = False
        
        # Step 3: Apply the mask to filter the arrays
        r0s= r0s[keep_mask]
        w1s_complex = w1s_complex[keep_mask]
        w2s_complex = w2s_complex[keep_mask]
        w3s_complex = w3s_complex[keep_mask]
        
        
        
        from scipy.interpolate import interp1d
        from scipy.signal import savgol_filter
        interpolation_function = interp1d(r0s, -np.imag(w1s_complex), kind='cubic')
        r0s_new = np.linspace(r0s.min(), r0s.max(), 400)  # 200 points between min and max of gs
        w1_ext_imag_interpolated = interpolation_function(r0s_new)
        w1_ext_imag_smooth = savgol_filter(w1_ext_imag_interpolated, window_length=11, polyorder=3)
        interpolation_function = interp1d(r0s, -np.imag(w2s_complex), kind='cubic')
        w2_ext_imag_interpolated = interpolation_function(r0s_new)
        w2_ext_imag_smooth = savgol_filter(w2_ext_imag_interpolated, window_length=11, polyorder=3)
        interpolation_function = interp1d(r0s, -np.imag(w3s_complex), kind='cubic')
        w3_ext_imag_interpolated = interpolation_function(r0s_new)
        w3_ext_imag_smooth = savgol_filter(w3_ext_imag_interpolated, window_length=11, polyorder=3)
        # Plot the results 
        fig1 = plt.figure( figsize=(4.5,3),dpi=150)
        bb = fig1.add_subplot(1, 1, 1)  
        bb.plot(r0s_new, w1_ext_imag_smooth, 'r-', label=r'$\Gamma_1$')  #swapping because labelling not right, w1 should be lower level
        bb.plot(r0s_new, w2_ext_imag_smooth, 'b-', label=r'$\Gamma_2$')
        bb.plot(r0s_new, w3_ext_imag_smooth, 'g-', label=r'$\Gamma_3$')
        bb.set_xlabel('distance between qubits, $d$ (nm)', fontsize='12')
        bb.set_ylabel('line broadening, $\Gamma$ ($\mu$ eV)', fontsize='12')
        plt.legend(loc=(0.75, 0.7), framealpha=0.0)
        plt.tight_layout()
        
        
        # FGR_w1=[]
        # FGR_w2=[]
        # FGR_w3=[]
        FGR_w1,FGR_w2,FGR_w3=FGR_smartie(j0_FGR,l,lp,Vs,T,g1,gd,w_qd1,w_c,r0s_new)
           
    
        fig1 = plt.figure( figsize=(4.5,3),dpi=150)
        bb = fig1.add_subplot(1, 1, 1)  
        bb.plot(r0s_new, w1_ext_imag_smooth, 'r-', label=r'$\Gamma_1$')  #swapping because labelling not right, w1 should be lower level
        bb.plot(r0s_new, w2_ext_imag_smooth, 'b-', label=r'$\Gamma_2$')
        bb.plot(r0s_new, w3_ext_imag_smooth, 'g-', label=r'$\Gamma_3$')

        bb.plot(r0s_new, FGR_w1, 'r--', label=r'$\Gamma_1 $ FGR')  #swapping because labelling not right, w1 should be lower level
        bb.plot(r0s_new, FGR_w2, 'b--', label=r'$\Gamma_2$ FGR')
        bb.plot(r0s_new, FGR_w3, 'g--', label=r'$\Gamma_3$ FGR')

        bb.set_xlabel('distance between qubits, $d$ (nm)', fontsize='12')
        bb.set_ylabel('line broadening, $\Gamma$ ($\mu$ eV)', fontsize='12')
        plt.legend(loc='best')
        plt.tight_layout()
    
    
        # Extend gam1_indep and gam2_indep to match the length
        length = len(r0s_new)  # Or any other list with the same length
       
        # gam1_indep=26.69001
        # gam2_indep=20.924178
        # gam3_indep=12.7947498
        # gam1_indep_extended = [gam1_indep] * length
        # gam2_indep_extended = [gam2_indep] * length
        # gam3_indep_extended = [gam3_indep] * length

        # import csv
        # from itertools import zip_longest
        # with open("QDQDCAV_Anisotropic_Dephasing.csv", "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Distances", "gam1","gam2", "gam3", "FGR_gam1", "FGR_gam2","FGR_gam3", "gam1_indep","gam2_indep", "gam3_indep"])  # Write header
        #     for row in zip_longest(r0s_new, w1_ext_imag_smooth, w2_ext_imag_smooth,w3_ext_imag_smooth, FGR_w1, FGR_w2, FGR_w3, gam1_indep_extended, gam2_indep_extended, gam3_indep_extended, fillvalue=np.nan):
        #         writer.writerow(row)
    
    
    
    #### REAL PART
    # plt.figure(figsize=(10, 8))
    # # Top subplot
    # plt.subplot(2, 1, 1)
    # plt.xlabel('distance between qubits, (nm)', fontsize=12)
    # plt.ylabel(r'Re $\omega_1$, ($\mu$ eV)', fontsize=12)
    # plt.title(f'$g$={round(gd*hbar*1e3, 1)} $\mu$eV, det={round(detuning*hbar*1e3, 1)}, T={round(T*hbar/kb, 1)}')
    # plt.plot(r0s, np.real(w1s_complex), 'b-', label='full calc.')
    # plt.plot(r0s, np.real(ws_plus), 'r--', label='unmod.')
    # plt.plot(r0s, np.real(ws_plus_mod), 'g--', label='modif.')
    # plt.legend()
    # # Bottom subplot
    # plt.subplot(2, 1, 2)
    # plt.xlabel('distance between qubits, (nm)', fontsize=12)
    # plt.ylabel(r'Re $\omega_2$, ($\mu$ eV)', fontsize=12)
    # plt.plot(r0s, np.real(w2s_complex), 'b-', label='full calc.')
    # plt.plot(r0s, np.real(ws_neg), 'r--', label='unmod.')
    # plt.plot(r0s, np.real(ws_neg_mod), 'g--', label='modif.')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
        
    # ##### COEFFICIENTS
    # plt.figure(figsize=(10, 8))
    # # Top subplot
    # plt.subplot(2, 1, 1)
    # plt.xlabel('distance between qubits, (nm)', fontsize=12)
    # plt.ylabel(r'A1', fontsize=12)
    # plt.title(f'$g$={round(gd*hbar*1e3, 1)} $\mu$eV, det={round(detuning*hbar*1e3, 1)}, T={round(T*hbar/kb, 1)}')
    # plt.plot(r0s, np.abs(A1s_complex), 'b-', label='full calc.')
    # plt.plot(r0s, np.abs(As_neg), 'r--', label='unmod.')
    # plt.plot(r0s, np.abs(As_neg_mod), 'g--', label='modif.')
    # plt.legend(loc='best')
    # # Bottom subplot
    # plt.subplot(2, 1, 2)
    # plt.xlabel('distance between qubits, (nm)', fontsize=12)
    # plt.ylabel(r'A2', fontsize=12)
    # plt.plot(r0s, np.abs(A2s_complex), 'b-', label='full calc.')
    # plt.plot(r0s, np.abs(As_plus), 'r--', label='unmod.')
    # plt.plot(r0s, np.abs(As_plus_mod), 'g--', label='modif.')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()
    
    # plt.figure(4)
    # plt.xlabel('distance between qubits, (nm)', fontsize='12')
    # plt.ylabel(r'A', fontsize='12')
    # plt.title(f' $g$={round(gd*hbar*1e3,1)} $\mu eV$, det={round(detuning*hbar*1e3,1)}, T={round(T*hbar/kb,1)} ')
    # plt.plot(r0s,np.abs(A1s_complex), 'b-', label='full calc.')
    # plt.plot(r0s,np.abs(A2s_complex), 'b-')
    # plt.plot(r0s,np.abs(As_plus), 'r--', label='unmod.')
    # plt.plot(r0s,np.abs(As_neg), 'r--')
    # plt.plot(r0s,np.abs(As_plus_mod), 'g--', label=' modif.')
    # plt.plot(r0s,np.abs(As_neg_mod), 'g--')
    # # plt.plot([],[],'k-',label='exact')
    # # plt.plot([],[],'k--',label='FGR unmodified')
    # # plt.plot([],[],'k.',label='FGR modified')
    # plt.tight_layout()
    # plt.legend(loc='best')
#%% #!!!   population fit parameters vs distance
    threshold_factor=1e-8
    threshold_str = str(threshold_factor)
    FitParameters_all=[] 
    L=12
    r0s=np.arange(3.7,12,0.5)
    counter=0
    for r in r0s:
        ###params that change with r0 ###
        r0=r
        t0=r0/Vs
        factortau=1.2
        if sharebath ==1:
            dt=( t0 + factortau*tauib )/(L+1)
        if sharebath ==0:
            dt=2*tauib / (L+1)
        
        j0_1=j0*Vs/r0 
        step_no = int(tfinal/dt)
        print(step_no)
        length = L/2
        length_U=L/2
        n_perms=int(d**((L/2)))
        n_perms_U=int(d**((L/2)))
        Lcycle=L
        if no_of_QDs==2 and cavity==1:
            LF=LFpol(g1, g2, gd, w_qd1, w_qd2, w_c)
            M1=expm(-1j*LF*dt)
            print('no')
        if no_of_QDs==2 and cavity==0:
            dcv=0.6
            incr=0.05
            eps=12.53#*8.33227308696581#*(np.pi*l*2)
            fact=5
            g=forster(l,eps,dcv,r0,-2*fact,10*fact,-4*fact,8*fact)
            print('d=',r0,'dt=',dt,'g=',g*1e3*hbar)
            LF = LFpop(g,  w_qd1.real, w_qd2.real, -params.gamma1/hbar, -params.gamma2/hbar) 
            M1=expm(-1j * LF * dt)

        
       
        # ### ##########generate data ###
        try:
            P=np.load(path+"/data/"+params.label(r0=r0, g=g)+".npy",allow_pickle=True)
        except:
            if no_of_QDs==2 and cavity==1:
                print('no')
    
                cumulants,cumulants_inin=Cumulants().cu(L,dt)
                Q0=np.array([[M1[0,0]*np.exp(cumulants_inin[0] +2*cumulants_inin[1]), M1[0,1]*np.exp(cumulants_inin[0] +2*cumulants[1]), M1[0,2] ],
                             [M1[1,0]*np.exp(cumulants_inin[0] +2*cumulants[1]), M1[1,1]*np.exp(cumulants_inin[0] +2*cumulants_inin[1]), M1[1,2] ],
                             [M1[2,0]*np.exp(cumulants_inin[0]), M1[2,1]*np.exp(cumulants_inin[0]), M1[2,2]]])
                Qlist=[]
                Qlist.append(Q0)
                for i in range(int(L-1)):
                    Qlist.append(np.array([[np.exp(2*cumulants_inin[i+2]), np.exp(2*cumulants[i+2]), 1 ],
                                        [np.exp(2*cumulants[i+2]), np.exp(2*cumulants_inin[i+2]), 1],
                                        [1, 1, 1]]))
            if no_of_QDs==2 and cavity==0:
                params_cumulant = {
                    'r0': r0,
                    'j0': j0,
                    'j0_1': j0_1,
                    'w0': w0,
                    'T': T,
                    'Vs': Vs,
                    'l': l,
                    'dotshape': dotshape,
                    'sharebath': 1,
                    'lp': lp}  
                cumulants = Cumulants()
                cumulants.update_parameters(**params_cumulant)
                cumulants, cumulants_inin = cumulants.cu(L, dt, **params_cumulant)
                # cumulants,cumulants_inin=Cumulants().cu(L,dt)
                print('cumulants:',cumulants)
                print('cumulants_inin:',cumulants_inin)
                size = 5
                alpha=np.array([0,1,1,0,0])
                beta=np.array([0,1,0,1,0])
                mu=np.array([0,0,0,1,1])
                nu=np.array([0,0,1,0,1])
                x=mu
                Qlist=[]
                Q = np.random.rand(size, size) + 1j * np.random.rand(size, size)
                for i in range(size):
                    for j in range(size):
                        Q[i, j] = M1[i,j] *(np.exp((alpha[j]-beta[j])*alpha[j]*cumulants_inin[0] + (beta[j]-alpha[j])*beta[j]*np.conjugate(cumulants_inin[0]) 
                        + (x[j]-nu[j])*x[j] * cumulants_inin[0] + (nu[j] - x[j])*nu[j]*np.conjugate(cumulants_inin[0]) 
                        + (  (x[j] - nu[j])*alpha[j] + (alpha[j]-beta[j])*x[j]   )*cumulants[0]
                        + ( (nu[j]-x[j])*beta[j]  + (beta[j]-alpha[j])*nu[j]   )*np.conjugate(cumulants[0])) * np.exp ( 2*( (alpha[i]-beta[i])*alpha[j]*cumulants_inin[1] + (beta[i]-alpha[i])*beta[j]*np.conjugate(cumulants_inin[1]) 
                        + (x[i]-nu[i])*x[j] * cumulants_inin[1] + (nu[i] - x[i])*nu[j]*np.conjugate(cumulants_inin[1]) 
                        + (  (x[i] - nu[i])*alpha[j] + (alpha[i]-beta[i])*x[j]   )*cumulants[1]
                        + ( (nu[i]-x[i])*beta[j]  + (beta[i]-alpha[i])*nu[j]   )*np.conjugate(cumulants[1]) )))
                Qlist.append(Q)
                for r in range(int(L-1)):
                    Q = np.random.rand(size, size) + 1j * np.random.rand(size, size)
                    for i in range(size):
                        for j in range(size):
                            Q[i, j] = np.exp ( 2*( (alpha[i]-beta[i])*alpha[j]*cumulants_inin[r+2] + (beta[i]-alpha[i])*beta[j]*np.conjugate(cumulants_inin[r+2]) 
                            + (x[i]-nu[i])*x[j] * cumulants_inin[r+2] + (nu[i] - x[i])*nu[j]*np.conjugate(cumulants_inin[r+2]) 
                            + (  (x[i] - nu[i])*alpha[j] + (alpha[i]-beta[i])*x[j]   )*cumulants[r+2]
                            + ( (nu[i]-x[i])*beta[j]  + (beta[i]-alpha[i])*nu[j]   )*np.conjugate(cumulants[r+2]) ))       
                    Qlist.append(Q)
                
                K12=cumulants
                K11=cumulants_inin
                K22=cumulants_inin 
                K12s=np.conjugate(K12)
                K11s=np.conjugate(K11)
                K22s=np.conjugate(K22)  
                def twotimecorrdiag(nm): 
                    'for diagonal cumulant only'
                    tdiff=abs(nm)
                    ttc=(beta-alpha)*(beta*K11s[tdiff]-alpha*K11[tdiff]
                    +(mu)*K22[tdiff]-nu*K22s[tdiff]
                    +(alpha-mu)*K12[tdiff]+(nu-beta)*K12s[tdiff])
                    return ttc
                KK=twotimecorrdiag(0)
                exp_k0_factor=np.exp(KK)
                    
                P= get_nth_permutation(L, Lcycle, r0,dt,length, n_perms, n_perms_U, step_no, M1, exp_k0_factor, Qlist)
                np.save(path+"/data/"+params.label(r0=r0, g=g)+".npy",P)
        P=np.ravel(P)
        times = np.array([dt*i for i in range(step_no+2)])
        #################################################################
        A1s=[]
        A2s=[]  
        A3s=[]
        A4s=[]
        A5s=[]
        w1s=[]
        w2s=[]
        w3s=[]
        w4s=[]
        w5s=[]
      
        # diagonalise numerically H_0 (no phonons to provide initial guess)
        if no_of_QDs==2 and cavity==1:
            matrix = np.array([[w_qd1, gd, g1],
                        [gd, w_qd2, g2],
                        [g1, g2, w_c]])
            eigenvalues = np.linalg.eigvals(matrix)
            eigenvalues=  np.sort(eigenvalues)[::-1]
            
        if no_of_QDs==1 and cavity==1:   
            matrix = np.array([[w_qd1, gd],
                                [gd, w_c]])
            eigenvalues = np.linalg.eigvals(matrix)
            eigenvalues=  np.sort(eigenvalues)[::-1]
            
        
        # print("Eigenvalues:", eigenvalues) 
        if no_of_QDs==2 and cavity==1:
            print('QD-QD-Cavity model')
            A1s.append(0 - 0*1j)
            A2s.append(0.0 - 0*1j)  
            A3s.append(0.0 - 0*1j)             
            w1s.append(eigenvalues[1] + 0*1j)                
            w2s.append(eigenvalues[0]+ 0*1j)      
            w3s.append(eigenvalues[2] + 0*1j) 
        
        if no_of_QDs==1 and cavity==1:
            print('QD-cavity model')
            A1s.append(0 - 0*1j)
            A2s.append(0 - 0*1j)            
            w1s.append(eigenvalues[0] + 0*1j)                
            w2s.append(eigenvalues[2]+ 0*1j)      

        if no_of_QDs==2 and cavity==1 and g1==0:
            print('QD-QD using QD-QD-cav model with cavity mode turned off')
            A1s.append(0 - 0*1j)
            A2s.append(0 - 0*1j)            
            w1s.append(eigenvalues[0] + 0*1j)                
            w2s.append(eigenvalues[2]+ 0*1j)
            
        if no_of_QDs==2 and cavity==0:
            print('QD-QD population dynamics')
            LF = LFpop(g,  w_qd1.real, w_qd2.real, -params.gamma1/hbar, -params.gamma2/hbar) 
            eigenvalues=np.linalg.eigvals(LF)
            eigenvalues=np.sort(eigenvalues)[::-1]
            A1s.append(0 - 0*1j)
            A2s.append(0.0 - 0*1j)  
            A3s.append(0.0 - 0*1j)  
            A4s.append(0.0 - 0*1j) 
            A5s.append(0.0 - 0*1j) 
            w1s.append(eigenvalues[0] + 0*1j)                
            w2s.append(eigenvalues[1]+ 0*1j)      
            w3s.append(eigenvalues[2] + 0*1j) 
            w4s.append(eigenvalues[3] + 0*1j) 
            w5s.append(eigenvalues[4] + 0*1j) 
            ##########################
            print('r0 in fitprocedure is:',r0)
            GamPh,RF,dp,dm, lambdp, lambdm=GammaPh(r0,params.det,params.T)
            gam1=Gamma1_FGR_det(r0,params.det,params.T)/hbar 
            gam2=Gamma2_FGR_det(r0,params.det,params.T)/hbar 
            Ct,AA,ww=Pan_NQD(0,r0,params.det,params.T)
         
            A=AA[4]
            C=AA[2]+AA[3]
            B=AA[1]#-A-C #-(AA[1]+AA[3]) # -A-C
            Gamd=gam1+gam2
            Gams=2*Gamd
            Phi=0
            pa=np.array([A,B,C,Gams,Gamd,RF,Phi]).real
            pa=np.asarray(pa)
            print('guess params a,b,c,T1,T2,R,phi:', pa, 'for d=',r0)
        
        def population(t,A,B,C,T1,T2,Rabi,phase):
            return A+B*np.exp(-T1*t) + C*np.exp(-T2*t)*np.cos(Rabi*t + phase)
        
        
        if no_of_QDs==2 and cavity==0: 
            Pnlongt=np.real(P[np.where(times>24)]) #extracting only the longt behaviour
            tmlongt=times[np.where(times>24)] #extracting only the longt behaviour
            # popn= popnModel()
            # guess=popn.guess()
            # result=popn.fit(Pnlongt,params=guess,t=tmlongt,method='leastsq',max_nfev=10000, nan_policy='omit', verbose=True)
            # fit_sametimes=popn.eval(params=result.params,t=times)
            popn= lmfit.Model(population)
            guesses = popn.make_params(A=pa[0], B=pa[1], C=pa[2], T1=pa[3], T2=pa[4], Rabi=pa[5], phase=pa[6])
            # guesses['A'].set(min=0.4, max=0.6)
            # guesses['B'].set(min=-0.6, max=0)
            # guesses['C'].set(min=-0.6, max=-0.2)
            # guesses['T1'].set(min=0)
            # guesses['T2'].set(min=0)
            # guesses['Rabi'].set(min=0)
            # guesses['phase'].set(min=-np.pi, max=np.pi)
            result = popn.fit(Pnlongt, params=guesses, t=tmlongt, method='leastsq', max_nfev=10000, nan_policy='omit', verbose=True)
            fit_sametimes=popn.eval(params=result.params,t=times)
        
        FitParameters_all.append(result.best_values)
        
        
        
        
        
        ###############################################################
        # tmlongt, Pnlongt, fit_sametimes =fitprocedure(g,r0,t0,P,times,14)
        # counter = counter +1

       
        fig1 = plt.figure( figsize=(4.5,3),dpi=150)
        bb = fig1.add_subplot(1, 1, 1)    
        bb.plot(np.abs(times),np.abs(P),'r-',markersize='1',linewidth='0.4', label=f'SVD, L={L}')
        bb.plot(np.abs(times),np.abs(fit_sametimes),'g--',label=f'fit L={L}') 
        if exc_channel==4:
            label_channel=2
        bb.set_ylabel(rf'$N_{{{measure_channel}{label_channel}}}(t)$',fontsize='12')
        bb.set_xlabel(r'time $\mathrm{(ps)}$',fontsize='12')
        bb.set_title(f' $g$={round(g*hbar*1e3,1)}$\mu eV$, det={round(detuning*hbar*1e3,1)}$\mu eV$, T={round(T*hbar/kb,1)}K,\n d={np.round(r0,2)} nm, Dc-Dv=-{DvDc} eV, l={l} nm')
        plt.legend(loc='best')
        plt.tight_layout()  
        # fig1.savefig(path+"/plots/reduced_fit_form/"+params.label(r0=r0, g=g)+".png")
        
    T1s=[]
    T2s=[]
    for i in range(len(r0s)):
        T1s.append(FitParameters_all[i]['T1'])
        T2s.append(FitParameters_all[i]['T2'])
    T1s=np.asarray(T1s)
    T2s=np.asarray(T2s)
    fig1 = plt.figure( figsize=(4.5,3),dpi=150)
    bb = fig1.add_subplot(1, 1, 1)    
    bb.set_title(f'det={round(detuning*hbar*1e3,1)}$\mu eV$, T={round(T*hbar/kb,1)}K \n Dc-Dv=-{DvDc} eV, l={l} nm')
    bb.plot(r0s,T2s, 'b-', label=r'1/($T_2$)') 
    bb.plot(r0s,T1s/2, 'r-', label=r'1/(2$T_1$)')
    plt.ylim(1e-4,1)
    bb.set_ylabel(r'dephasing rate (ps$^{-1}$)')
    bb.set_xlabel('distance, $d$ (nm)')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.tight_layout()
        


 
    
#%% #!!! QD-Cavity g vary 
    from scipy.optimize import newton
    
    def NR_fixed(Ls,ws):
        N=len(Ls)
        omega1 = np.sum(ws)
        omega2 = np.sum(ws[i] / (Ls[i]**2) for i in range(len(Ls)))
        S_1 = np.sum(1/Ls[i]**2 for i in range(len(Ls)))
        S_2 = np.sum(1/Ls[i]**4 for i in range(len(Ls)))
    
        ws_inf_NRfix=(omega2 *S_1 - omega1*S_2)/(S_1**2 - S_2 *N)
        C=(omega1 *S_1 - omega2*N)/(S_1**2 - S_2 *N)
        return ws_inf_NRfix,C
    
    FitParameters_all=[] 
    gs=np.insert(np.arange(10,9000,10),0,1)
    
    params_cumulant = {
            'omp': omp,
            'shr': shr,
            'j0': j0,
            'T': T,
            'w0': w0}  
    # factortau=1.2   
    Ls=np.arange(24,28+1,2)
    def power_law_model(L, omega_infinity, C, beta):
        return omega_infinity + C * L**-beta

    w1_ext=[]
    w2_ext=[]
    A1_ext=[]
    A2_ext=[]
    for g in gs:
        ###params that change with g ###
        gd=g*1e-3/hbar
        FitParameters_all=[]
        for L in Ls:
            dt= factortau*tauib /(L+1)
            step_no = int(tfinal/dt)
            # print(step_no)
            length = L/2
            length_U=L/2
            n_perms=int(d**((L/2)))
            n_perms_U=int(d**((L/2)))
            Lcycle=L
            print(gd*1e3*hbar)
            D, ww, U1, V1 = DiagM_qdcav(gd, 0, 0,0, detuning, omp)
            DD = np.diag(np.exp(-1j*ww*dt))
            M1 = U1*DD*V1
            
          
           
            # ### ##########generate data ###
            try:
                P=np.load(path+"/data/"+params.label(gd=gd, L=L)+".npy",allow_pickle=True)
            except:
                Cumulants = Cumulants_1qd()
                Cumulants.update_parameters(**params_cumulant)
                cumulants_inin = Cumulants.cu(L, dt, **params_cumulant)
                cumulants=cumulants_inin
                
                Q0=np.array([[M1[0,0]*np.exp(cumulants[0] +2*cumulants[1]), M1[0,1] ],[M1[1,0]*np.exp(cumulants[0]), M1[1,1]]])
                Qlist=[]
                Qlist.append(Q0)
                for i in range(int(L-1)):
                    Qlist.append(np.array([[np.exp(2*cumulants[i+2]), 1 ],[1, 1]]))
                
                exp_k0_factor=np.exp(cumulants_inin[0])       
                P= get_nth_permutation(L, Lcycle, r0,dt,length, n_perms, n_perms_U, step_no, M1, exp_k0_factor, Qlist)
                np.save(path+"/data/"+params.label(gd=gd, L=L)+".npy",P)
            P=np.ravel(P)
            times = np.array([dt*i for i in range(step_no+2)])
            #################################################################
            A1s=[]
            A2s=[]  
           
            w1s=[]
            w2s=[]
           
          
            # diagonalise numerically H_0 (no phonons to provide initial guess)
            
            if no_of_QDs==1 and cavity==1:   
                matrix = np.array([[w_qd1, gd],
                                    [gd, w_c]])
                eigenvalues = np.linalg.eigvals(matrix)
                eigenvalues=  np.sort(eigenvalues)[::-1]
                print('QD-cavity model')
                A1s.append(0.5 - 0*1j)
                A2s.append(0.5 - 0*1j)            
                w1s.append(eigenvalues[0] + 0*1j)                
                w2s.append(eigenvalues[1]+ 0*1j)  
            
            def biexponential(t, A1_re,A1_im,A2_re,A2_im,w1_re,w1_im,w2_re,w2_im):
                return (A1_re - 1j*A1_im)*np.exp(-1j*(w1_re -1j*w1_im)*t) + (A2_re - 1j*A2_im)*np.exp(-1j*(w2_re -1j*w2_im)*t)
            
            
            Pnlongt=P[np.where(times>2.0*tauib)] #extracting only the longt behaviour
            tmlongt=times[np.where(times>2.0*tauib)] #extracting only the longt behaviour
            biexp=lmfit.Model(biexponential)
            k=0
            guesses= biexp.make_params(A1_re=np.real(A1s[k]), A1_im=-np.imag(A1s[k]), A2_re=np.real(A2s[k]) , A2_im=-np.imag(A2s[k]), w1_re=np.real(w1s[k]), w1_im=-np.imag(w1s[k]) , w2_re=np.real(w2s[k]), w2_im=-np.imag(w2s[k]))
            result= biexp.fit(Pnlongt, params=guesses, t=tmlongt, method='leastsq', verbose=True) #fit of current calculated P
            fit_sametimes=biexp.eval(params=result.params, t=times)
         
            FitParameters_all.append(result.best_values)
        ############################ Ls have been calculated here, perform extrapolation now
        w1s, w2s, w3s, gam1s, gam2s, gam3s, A1res, A2res, A3res, A1ims, A2ims, A3ims = [[] for _ in range(12)]
      
        for i in range(len(Ls)): 
            w1s.append(FitParameters_all[i]['w1_re']*hbar*1e3)
            w2s.append(FitParameters_all[i]['w2_re']*hbar*1e3)
            A1res.append(FitParameters_all[i]['A1_re'])
            A2res.append(FitParameters_all[i]['A2_re'])
            
            gam1s.append(np.abs(FitParameters_all[i]['w1_im']*hbar*1e3))
            gam2s.append(np.abs(FitParameters_all[i]['w2_im']*hbar*1e3))
            A1ims.append(FitParameters_all[i]['A1_im'])
            A2ims.append(FitParameters_all[i]['A2_im'])
    
        
        
        w1s_complex = np.array(w1s, dtype=np.float64).astype(np.complex128)
        w2s_complex = np.array(w2s, dtype=np.float64).astype(np.complex128) 
        gam1s_complex = np.array(gam1s, dtype=np.float64).astype(np.complex128)
        gam2s_complex = np.array(gam2s, dtype=np.float64).astype(np.complex128) 
        A1sre_complex = np.array(A1res, dtype=np.float64).astype(np.complex128)
        A1sim_complex = np.array(A1ims, dtype=np.float64).astype(np.complex128)
        A2sre_complex = np.array(A2res, dtype=np.float64).astype(np.complex128)
        A2sim_complex = np.array(A2ims, dtype=np.float64).astype(np.complex128)
       
        w1s_c=w1s_complex-1j*gam1s_complex
        w2s_c=w2s_complex-1j*gam2s_complex
        A1s_c=A1sre_complex-1j*A1sim_complex
        A2s_c=A2sre_complex-1j*A2sim_complex
      
        if gd*1e3*hbar > 590:
            # def NR(ws,Ls, selected_range): 
            #     ws=ws[selected_range]
            #     Ls=Ls[selected_range]
            #     w_L3L2=ws[-3]-ws[-2]
            #     w_L1L3=ws[-1]-ws[-3]
            #     w_L2L1=ws[-2]-ws[-1]
            #     def f_alpha(alpha):
            #         return w_L3L2*Ls[-1]**-alpha + w_L1L3*Ls[-2]**-alpha + w_L2L1*Ls[-3]**-alpha
            #     def f_alpha_prime(alpha):
            #         return -( w_L3L2*Ls[-1]**-alpha * np.log(Ls[-1]) + w_L1L3*Ls[-2]**-alpha * np.log(Ls[-2]) + w_L2L1*Ls[-3]**-alpha * np.log(Ls[-3]) )
            #     initial_guess=2-0j
            #     root = newton(f_alpha, x0=initial_guess, fprime=f_alpha_prime)
            #     C= (ws[-1]-ws[-2])/(Ls[-1]**-root - Ls[-2]**-root)
            #     w_infty=ws[-1] - C*Ls[-1]**-root
            #     return w_infty,C,root
            
            # def params_NR(ws_c): 
            #     w_inftys=[]
            #     C_NRs=[]
            #     alpha_NR=[]
            #     Lvals=[]
    
            #     selected_range = slice(np.where(Ls==Ls[-3])[0][0], np.where(Ls==Ls[-3])[0][0] +3) 
            #     print(selected_range)
            #     Lvals.append(Ls[selected_range][1])
            #     w_infty,C,alpha=NR(ws_c,Ls, selected_range)
            #     w_inftys.append(w_infty)
            #     C_NRs.append(C)
            #     alpha_NR.append(alpha)  
            #     return w_inftys,C_NRs,alpha_NR,Lvals
            
            # w_1_inftys,C1_NRs,alpha1_NR,Lvals= params_NR(w1s_c)
            # w_2_inftys,C2_NRs,alpha2_NR,Lvals= params_NR(w2s_c)
            # A1_inftys,CA1_NR,alphaA1_NR,Lvals= params_NR(A1s_c)
            # A2_inftys,CA2_NR,alphaA2_NR,Lvals= params_NR(A2s_c)
            
            w_1_inftys,C1=NR_fixed(Ls,w1s_c)
            w_2_inftys,C2=NR_fixed(Ls,w2s_c)
            A1_inftys,C1A=NR_fixed(Ls,A1s_c)
            A2_inftys,C2A=NR_fixed(Ls,A2s_c)
            
            w1_ext.append(w_1_inftys)
            w2_ext.append(w_2_inftys)
            A1_ext.append(A1_inftys)
            A2_ext.append( A2_inftys)
            print('g=',gd*1e3*hbar, 'w1 extrapolated', w1_ext, 'w2_extrapolated', w2_ext)
        if gd*1e3*hbar < 600:
            w1_ext.append(w1s_c[-1])
            w2_ext.append(w2s_c[-1])
            A1_ext.append(A1s_c[-1])
            A2_ext.append(A2s_c[-1])

    w1_ext = np.array([elem[0] if isinstance(elem, list) else elem for elem in w1_ext], dtype=np.complex128)
    w2_ext = np.array([elem[0] if isinstance(elem, list) else elem for elem in w2_ext], dtype=np.complex128)
    A1_ext = np.array([elem[0] if isinstance(elem, list) else elem for elem in A1_ext], dtype=np.complex128)
    A2_ext = np.array([elem[0] if isinstance(elem, list) else elem for elem in A2_ext], dtype=np.complex128)

       

    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter
    interpolation_function = interp1d(gs, -np.imag(w1_ext), kind='cubic')
    gs_new = np.linspace(gs.min(), gs.max(), 400)  # 200 points between min and max of gs
    w1_ext_imag_interpolated = interpolation_function(gs_new)
    w1_ext_imag_smooth = savgol_filter(w1_ext_imag_interpolated, window_length=11, polyorder=3)
    interpolation_function = interp1d(gs, -np.imag(w2_ext), kind='cubic')
    w2_ext_imag_interpolated = interpolation_function(gs_new)
    w2_ext_imag_smooth = savgol_filter(w2_ext_imag_interpolated, window_length=11, polyorder=3)
    # Plot the results
    #FGR
    Gamma1,Gamma2=FGR_qdcav_spherical_det(gs_new*1e-3/hbar, detuning, j0_FGR, l, Vs, T) 
    fig1 = plt.figure( figsize=(4.5,3),dpi=150)
    bb = fig1.add_subplot(1, 1, 1)  
    bb.plot(gs_new/1000, w2_ext_imag_smooth/1e3, 'r-', label=r'$\Gamma_1$')  #swapping because labelling not right, w1 should be lower level
    bb.plot(gs_new/1000, w1_ext_imag_smooth/1e3, 'b-', label=r'$\Gamma_2$')
    bb.plot(gs_new/1000, Gamma1/1e3, 'r--', label=r'FGR $\Gamma_1$')  #swapping because labelling not right, w1 should be lower level
    bb.plot(gs_new/1000, Gamma2/1e3, 'b--', label=r'FGR $\Gamma_2$')
    bb.set_xlabel('coupling strength, $g$ (meV)', fontsize='12')
    bb.set_ylabel('line broadening, $\Gamma$ (meV)', fontsize='12')
    # plt.legend(loc=(0.75, 0.7), framealpha=0.0)
    plt.legend(loc='best', framealpha=0.0)
    plt.xlim(0,np.max(gs_new)/1000  +0.01)
    plt.ylim(0,0.2)
    bb.set_xticks(np.arange(0,7+1, 1))
    bb.set_yticks([0.00, 0.05, 0.10, 0.15,0.2])
    plt.tight_layout()
    
    fig1 = plt.figure( figsize=(4.5,3),dpi=150)
    bb = fig1.add_subplot(1, 1, 1)  
    bb.plot(gs*1e-3, A1_ext, 'r-', label=r'$A_1$')  #swapping because labelling not right, w1 should be lower level
    bb.plot(gs*1e-3, A2_ext, 'b-', label=r'$A_2$')
    bb.set_xlabel('coupling strength, $g$ (meV)', fontsize='12')
    bb.set_ylabel('coefficient, $A$', fontsize='12')
    plt.legend(loc=(0.75, 0.7), framealpha=0.0)
    plt.xlim(0,np.max(gs_new)/1000  +0.01)
    # plt.ylim(0,0.15)
    # bb.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    # bb.set_yticks([0.00, 0.05, 0.10, 0.15])
    plt.tight_layout()
    
    delta_omega=[]
    for i in range(len(w1_ext)):
        delta_omega.append((np.real(w1_ext[i]) - np.real(w2_ext[i])) - 2*gs[i])
    delta_omega=np.asarray(delta_omega)
    fig1 = plt.figure( figsize=(4.5,3),dpi=150)
    bb = fig1.add_subplot(1, 1, 1)  
    bb.plot(gs/1000, delta_omega/1e3, 'r-')  #swapping because labelling not right, w1 should be lower level
    bb.set_ylabel('Rabi splitting, $\Delta \omega - 2g$ (meV)', fontsize='12')
    bb.set_xlabel('coupling strength, $g$ (meV)', fontsize='12')
    bb.set_xlim(left=0)
    plt.tight_layout()
    




#%% #!!!
#population extrapolation attempt
    from scipy.optimize import newton
    Ls=np.arange(10,14+1,2)
    FitParameters_all=[]
    for L in Ls:
        P=np.load(path+"/data/"+params.label(L=L)+".npy",allow_pickle=True)
        P=np.ravel(P)
            
        dt=( t0 + factortau*tauib )/(L+1)
        step_no = int(tfinal/dt)
        times = np.array([dt*i for i in range(step_no+2)])
        print(L)
        tmlongt, Pnlongt, fit_sametimes= fitprocedure(g,g1,g2,gd,r0,t0,P,times)
        
    As=[]
    Bs=[]
    Cs=[]    
    T1s=[]
    T2s=[]
    Rs=[]
    phis=[]
    for i in range(len(Ls)):
        T1s.append(FitParameters_all[i]['T1']*1e3*hbar)
        T2s.append(FitParameters_all[i]['T2']*1e3*hbar)
        As.append(FitParameters_all[i]['A'])
        Bs.append(FitParameters_all[i]['B'])
        Cs.append(FitParameters_all[i]['C'])
        Rs.append(FitParameters_all[i]['Rabi'])
        phis.append(FitParameters_all[i]['phase'])
    T1s=np.asarray(T1s)
    T2s=np.asarray(T2s)

        

    def NR(ws,Ls, selected_range): 
        ws=ws[selected_range]
        Ls=Ls[selected_range]
        w_L3L2=ws[-3]-ws[-2]
        w_L1L3=ws[-1]-ws[-3]
        w_L2L1=ws[-2]-ws[-1]
        def f_alpha(alpha):
            return w_L3L2*Ls[-1]**-alpha + w_L1L3*Ls[-2]**-alpha + w_L2L1*Ls[-3]**-alpha
        def f_alpha_prime(alpha):
            return -( w_L3L2*Ls[-1]**-alpha * np.log(Ls[-1]) + w_L1L3*Ls[-2]**-alpha * np.log(Ls[-2]) + w_L2L1*Ls[-3]**-alpha * np.log(Ls[-3]) )
        initial_guess=2
        root = newton(f_alpha, x0=initial_guess, fprime=f_alpha_prime, maxiter=200)
        C= (ws[-1]-ws[-2])/(Ls[-1]**-root - Ls[-2]**-root)
        w_infty=ws[-1] - C*Ls[-1]**-root
        return w_infty,C,root
    
    def params_NR(ws_c): 
        w_inftys=[]
        C_NRs=[]
        alpha_NR=[]
        Lvals=[]
    
        selected_range = slice(np.where(Ls==Ls[-3])[0][0], np.where(Ls==Ls[-3])[0][0] +3) 
        print(selected_range)
        Lvals.append(Ls[selected_range][1])
        w_infty,C,alpha=NR(ws_c,Ls, selected_range)
        w_inftys.append(w_infty)
        C_NRs.append(C)
        alpha_NR.append(alpha)  
        return w_inftys,C_NRs,alpha_NR,Lvals
    
    gam1_inftys,C1_NRs,alpha1_NR,Lvals= params_NR(T1s)
    gam2_inftys,C2_NRs,alpha2_NR,Lvals= params_NR(T2s)
    
    
    def power_law_model(L, omega_infinity, C, beta):
        return omega_infinity + C * L**-beta
    
    Lvals=np.arange(6,18,1)
    gam1_PL=power_law_model(Lvals, gam1_inftys[0], C1_NRs[0], alpha1_NR[0])
    gam2_PL=power_law_model(Lvals, gam2_inftys[0], C2_NRs[0], alpha2_NR[0])

    plt.plot(Ls,T1s,'rx',label='gam1')
    plt.plot(Lvals,gam1_PL,'r--',label='gam1 PL')
    plt.legend(loc='best')
    
    plt.plot(Ls,T2s,'bx',label='gam2')
    plt.plot(Lvals,gam2_PL,'b--', label='gam2 PL')
    plt.legend(loc='best')
    
    
    def PL_fit(ws):
      # Define the range you want to select
        inf_vals_PL=[]
        inf_vals_exp=[]
        Cs_PL=[]
        betas_PL=[]
        
        selected_range = slice(np.where(Ls==12)[0][0], None)  # Change the slice values as needed, e.g., slice(0, 15)
     
        initial_guess_pl = [ws[-1], 1, 1]
        # Fit the power law model
        params_pl, pcov = curve_fit(power_law_model, Ls[selected_range], np.real(ws)[selected_range], p0=initial_guess_pl)
        omega_inf_pl, C_pl, beta_pl = params_pl
        # print(pcov)
        inf_err = np.sqrt(pcov[0, 0])
        inf_vals_PL.append(omega_inf_pl)
        Cs_PL.append(C_pl)
        betas_PL.append(beta_pl)
        
        L_ex=np.linspace(Ls[0],Ls[-1],50)
        # omega_exp_fit = exponential_model(L_ex, *params_exp)
        omega_pl_fit = power_law_model(L_ex, *params_pl)
        # w_infinity=params_pl[0]
        return omega_inf_pl, C_pl, beta_pl, omega_pl_fit, inf_err
    
    L_ex=np.linspace(Ls[0],Ls[-1],50)
    gam1_inf,gam1_C,gam11_beta, omega_pl_fit_gam1, gam11_inf_err=PL_fit(T1s)
    gam2_inf,gam2_C,gam21_beta, omega_pl_fit_gam2, gam11_inf_err=PL_fit(T2s)

    plt.plot(Ls,T1s,'rx',label='gam1')
    plt.plot(L_ex,omega_pl_fit_gam1,'r--',label='gam1 PL fit')
    plt.plot(Ls,T2s,'bx',label='gam2')
    plt.plot(L_ex,omega_pl_fit_gam2,'b--',label='gam2 PL fit')

    plt.legend(loc='best')
    
#%%  #!!!
#Writing code to find extrapolated parameters for QD-QD-CAVITY
    from scipy.optimize import newton

    Ls=np.arange(18,22+1,2)
    r0s=np.arange(4.01,55,2)
    # threshold_factor=1e-6
    # threshold_str = str(threshold_factor)
    
    
    w1_infs=[]
    w2_infs=[]
    w3_infs=[]
    for r in r0s:
        r0=r
        w1s, w2s, w3s, gam1s, gam2s, gam3s, A1res, A2res, A3res, A1ims, A2ims, A3ims = [[] for _ in range(12)]
        FitParameters_all=[] 
        for L in Ls:
            ###params that change with L ###            
            t0=r0/Vs
            factortau=1.2
            if sharebath ==1:
                dt=( t0 + factortau*tauib )/(L+1)
            if sharebath ==0:
                dt=2*tauib / (L+1)
            
            j0_1=j0*Vs/r0 
                
            step_no = int(tfinal/dt)
            # print(step_no)
            length = L/2
            length_U=L/2
            n_perms=int(d**((L/2)))
            n_perms_U=int(d**((L/2)))
            Lcycle=L
            
            
            LF=LFpol_qdqdcav(g1, g2,gd, w_qd1, w_qd2, w_c)
            M1=expm(-1j*LF*dt)
    
            print('L=', L,'d=',np.round(r0,3), 'dt=',np.round(dt,3))
            ### ##########generate data ###
            try:
                P=np.load(path+"/data/"+params.label(L=L, r0=r0)+".npy",allow_pickle=True)
            except:
                params_cumulant = {
                    'r0': r0,
                    'j0': j0,
                    'j0_1': j0_1,
                    'w0': w0,
                    'T': T,
                    'Vs': Vs,
                    'l': l,
                    'dotshape': dotshape,
                    'sharebath': 1,
                    'lp': lp}  
                cumulants = Cumulants()
                cumulants.update_parameters(**params_cumulant)
                cumulants, cumulants_inin = cumulants.cu(L, dt, **params_cumulant)
                Q0=np.array([[M1[0,0]*np.exp(cumulants_inin[0] +2*cumulants_inin[1]), M1[0,1]*np.exp(cumulants_inin[0] +2*cumulants[1]), M1[0,2] ],
                              [M1[1,0]*np.exp(cumulants_inin[0] +2*cumulants[1]), M1[1,1]*np.exp(cumulants_inin[0] +2*cumulants_inin[1]), M1[1,2] ],
                              [M1[2,0]*np.exp(cumulants_inin[0]), M1[2,1]*np.exp(cumulants_inin[0]), M1[2,2]]])
                Qlist=[]
                Qlist.append(Q0)
                for i in range(int(L-1)):
                    Qlist.append(np.array([[np.exp(2*cumulants_inin[i+2]), np.exp(2*cumulants[i+2]), 1 ],
                                            [np.exp(2*cumulants[i+2]), np.exp(2*cumulants_inin[i+2]), 1],
                                            [1, 1, 1]]))
                exp_k0_factor=np.exp(cumulants_inin[0])
                P= get_nth_permutation(L, Lcycle, r0,dt,length, n_perms, n_perms_U, step_no, M1, exp_k0_factor, Qlist)
                np.save(path+"/data/"+params.label(L=L, r0=r0)+".npy",P)
            
            P=np.ravel(P)
            times = np.array([dt*i for i in range(step_no+2)])
        
            tmlongt, Pnlongt, fit_sametimes= fitprocedure(g,g1,g2,gd,r0,t0,P,times)
        ####### end of L loop #######
      
        for i in range(len(Ls)): 
            w1s.append(FitParameters_all[i]['w1_re']*hbar*1e3)
            w2s.append(FitParameters_all[i]['w2_re']*hbar*1e3)
            A1res.append(FitParameters_all[i]['A1_re'])
            A2res.append(FitParameters_all[i]['A2_re'])
            if g1!= 0:
                w3s.append(FitParameters_all[i]['w3_re']*hbar*1e3)
                A3res.append(FitParameters_all[i]['A3_re'])

            gam1s.append(np.abs(FitParameters_all[i]['w1_im']*hbar*1e3))
            gam2s.append(np.abs(FitParameters_all[i]['w2_im']*hbar*1e3))
            A1ims.append(FitParameters_all[i]['A1_im'])
            A2ims.append(FitParameters_all[i]['A2_im'])
            if g1!= 0:
                gam3s.append(np.abs(FitParameters_all[i]['w3_im']*hbar*1e3))
                A3ims.append(FitParameters_all[i]['A3_im'])

        from scipy.optimize import newton
        w1s_complex = np.array(w1s, dtype=np.float64).astype(np.complex128)
        w2s_complex = np.array(w2s, dtype=np.float64).astype(np.complex128) 
        gam1s_complex = np.array(gam1s, dtype=np.float64).astype(np.complex128)
        gam2s_complex = np.array(gam2s, dtype=np.float64).astype(np.complex128) 
        A1sre_complex = np.array(A1res, dtype=np.float64).astype(np.complex128)
        A1sim_complex = np.array(A1ims, dtype=np.float64).astype(np.complex128)
        A2sre_complex = np.array(A2res, dtype=np.float64).astype(np.complex128)
        A2sim_complex = np.array(A2ims, dtype=np.float64).astype(np.complex128)
        try:
            w3s_complex = np.array(w3s, dtype=np.float64).astype(np.complex128)
            gam3s_complex = np.array(gam3s, dtype=np.float64).astype(np.complex128)
            A3sre_complex = np.array(A3res, dtype=np.float64).astype(np.complex128)
            A3sim_complex = np.array(A3ims, dtype=np.float64).astype(np.complex128)
        except:
            pass
        
        w1s_c=w1s_complex-1j*gam1s_complex
        w2s_c=w2s_complex-1j*gam2s_complex
        A1s_c=A1sre_complex-1j*A1sim_complex
        A2s_c=A2sre_complex-1j*A2sim_complex
        try:
            w3s_c=w3s_complex-1j*gam3s_complex
            A3s_c=A3sre_complex-1j*A3sim_complex
        except:
            pass
        
        if r0 > 4:
            def NR(ws,Ls, selected_range): 
                ws=ws[selected_range]
                Ls=Ls[selected_range]
                w_L3L2=ws[-3]-ws[-2]
                w_L1L3=ws[-1]-ws[-3]
                w_L2L1=ws[-2]-ws[-1]
                def f_alpha(alpha):
                    return w_L3L2*Ls[-1]**-alpha + w_L1L3*Ls[-2]**-alpha + w_L2L1*Ls[-3]**-alpha
                def f_alpha_prime(alpha):
                    return -( w_L3L2*Ls[-1]**-alpha * np.log(Ls[-1]) + w_L1L3*Ls[-2]**-alpha * np.log(Ls[-2]) + w_L2L1*Ls[-3]**-alpha * np.log(Ls[-3]) )
                initial_guess=2-0j
                root = newton(f_alpha, x0=initial_guess, fprime=f_alpha_prime, maxiter=200)
                C= (ws[-1]-ws[-2])/(Ls[-1]**-root - Ls[-2]**-root)
                w_infty=ws[-1] - C*Ls[-1]**-root
                return w_infty,C,root
            
            def params_NR(ws_c): 
                w_inftys=[]
                C_NRs=[]
                alpha_NR=[]
                Lvals=[]
                
                selected_range = slice(np.where(Ls==Ls[-3])[0][0], np.where(Ls==Ls[-3])[0][0] +3) 
                print(selected_range)
                Lvals.append(Ls[selected_range][1])
                w_infty,C,alpha=NR(ws_c,Ls, selected_range)
                w_inftys.append(w_infty)
                C_NRs.append(C)
                alpha_NR.append(alpha) 
                
                return w_inftys,C_NRs,alpha_NR,Lvals
            
            w_1_inftys,C1_NRs,alpha1_NR,Lvals= params_NR(np.imag(w1s_c))
            w_2_inftys,C2_NRs,alpha2_NR,Lvals= params_NR(np.imag(w2s_c))
            if g1!=0:
                w_3_inftys,C3_NRs,alpha3_NR,Lvals= params_NR(np.imag(w3s_c))

        # A1_inftys,CA1_NR,alphaA1_NR,Lvals= params_NR(A1s_c)
        # A2_inftys,CA2_NR,alphaA2_NR,Lvals= params_NR(A2s_c)
        # A3_inftys,CA3_NR,alphaA3_NR,Lvals= params_NR(A3s_c)
            w1_infs.append(w_1_inftys)
            w2_infs.append(w_2_inftys)
            if g1!=0:
                w3_infs.append(w_3_inftys)

        if r0 < 3.99:
            w1_infs.append(w1s_c[-1])
            w2_infs.append(w2s_c[-1])
            if g1!=0:
                w3_infs.append(w3s_c[-1])
            
    w1_infs= np.array([elem[0] if isinstance(elem, list) else elem for elem in  w1_infs], dtype=np.complex128)
    w2_infs = np.array([elem[0] if isinstance(elem, list) else elem for elem in  w2_infs], dtype=np.complex128)
    if g1 !=0:
        w3_infs = np.array([elem[0] if isinstance(elem, list) else elem for elem in  w3_infs], dtype=np.complex128)

    plt.plot(r0s,np.abs(np.imag(w1_infs)), 'r-', label='gam1')
    plt.plot(r0s,np.abs(np.imag(w2_infs)),'b-', label='gam2')
    if g1!=0:
        plt.plot(r0s,np.abs(np.imag(w3_infs)),'g-', label='gam3')
    plt.legend(loc='best')
    # dephasing1,dephasing2,dephasing3=FGR_spherical(j0_FGR,l,Vs,T,g1,gd,w_qd1,w_c,r0s)
    dephasing1,dephasing2,dephasing3=FGR_smartie(j0_FGR,l,lp,Vs,T,g1,gd,w_qd1,w_c,r0s)
    plt.plot(r0s, dephasing1 ,'r--', label='FGR gam1')
    plt.plot(r0s, dephasing2 ,'b--', label='FGR gam2')
    if g1 !=0:
        plt.plot(r0s, dephasing3, 'g--', label='FGR gam3')
    plt.legend(loc='best')
        



        # L_ex=np.linspace(Ls[0],Ls[-1],50)
        # try:
        #     w1_inf,w1_C,w1_beta, omega_pl_fit_w1, w1_inf_err=PL_fit(w1s)
        # except:
        #     w1_inf=w1s[-1]
        # try:
        #     w2_inf,w2_C,w2_beta, omega_pl_fit_w2, w2_inf_err=PL_fit(w2s)
        # except:
        #     w2_inf=w2s[-1]
        # try: 
        #     gam1_inf,gam1_C,gam1_beta, omega_pl_fit_gam1, gam1_inf_err=PL_fit(gam1s)
        # except:
        #     gam1_inf=gam1s[-1]
        # try:
        #     gam2_inf,gam2_C,gam2_beta, omega_pl_fit_gam2, gam2_inf_err=PL_fit(gam2s)
        # except:
        #     gam2_inf=gam2s[-1]
        # try:
        #     A1re_inf,A1re_C,A1re_beta, omega_pl_fit_A1re, A1re_inf_err=PL_fit(A1res)
        # except:
        #     A1re_inf=A1res[-1]
        # try:
        #     A2re_inf,A2re_C,A2re_beta, omega_pl_fit_A2re, A2re_inf_err=PL_fit(A2res)
        # except:
        #     A2re_inf=A2res[-1]
        # try: 
        #     A1im_inf, A1im_C, A1im_beta, omega_pl_fit_A1im, A1im_inf_err=PL_fit(A1ims)
        # except:
        #     A1im_inf=A1ims[-1]
        # try:
        #     A2im_inf, A2im_C, A2im_beta, omega_pl_fit_A2im, A2im_inf_err=PL_fit(A2ims)
        # except:
        #     A2im_inf=A2ims[-1]
        # w1s, w2s, w3s, gam1s, gam2s, gam3s, A1res, A2res, A3res, A1ims, A2ims, A3ims = [[] for _ in range(12)]
        # FitParameters_all=[] 

        # w1_complex=w1_inf -1j*gam1_inf
        # w2_complex=w2_inf -1j*gam2_inf
        # A1_complex=A1re_inf -1j*A1im_inf
        # A2_complex=A2re_inf -1j*A2im_inf

            

        