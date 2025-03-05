import dask.array as da
import numpy as np
from scipy.optimize import minimize_scalar
import cupy as cp
import opt_einsum as oe
import itertools
from scipy.linalg import expm
import Parameters as params
from Functions import LFpop, LFpol, forster, LFpol_qdqdcav, DiagM_qdcav, S_inin, S_inim, PolaronShift, PolaronShift_inim, phi_inin, phi_inim, K11_smartie, K12_smartie, Kbb2, analytics_bareg, analytics_modified, QDQD_analytics_smartie, FGR_smartie, FGR_spherical, FGR_qdcav_spherical, FGR_qdcav_spherical_det
from Forster_FGR import Gamma1_FGR_det, Gamma2_FGR_det, N21ampli, GammaPh, Gamma1_FGR_det_nrm, Gamma2_FGR_det_nrm, Pan_NQD, Panalyt2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm 
import os
from scipy.optimize import least_squares
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
vc=params.vc
dens=params.dens
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
RAM_avail=50000000 #free RAM in GB.
maximum_neighbours=np.emath.logn(d,(RAM_avail / (d*16*10**-9))) # 3^(L+1) tensor elements * 16 bytes per complex number stored * 10^-9 for byte to GB conversion
print('The original full tensor approach has approx. the maximum no. of neighbours:', maximum_neighbours)



# P = []
# if exc_channel==measure_channel:
#     P.append(np.array([1+0j]))
# else:  
#     P.append(np.array([0+0j]))
# if correlator=='LP':
#     P.append(np.array([exp_k0_factor*M1[measure_channel,exc_channel]]))
# if correlator=='NQD':
#     P.append(np.array([exp_k0_factor[measure_channel]*M1[measure_channel,exc_channel]]))
    # print(M1[measure_channel,exc_channel])
def generate_permutations(d,length):
    for perm in itertools.product(list(range(d)), repeat=length):
        yield perm


# if L % 2 !=0:
#     def generate_permutations_odd(d):
#         for perm in itertools.product(list(range(d)), repeat=int(length_U)):
#             yield perm
#     permutations_odd = np.array(list(generate_permutations_odd(d)), dtype=np.int32)

def generate_arrays(n_perms, d):
    return {i: np.ones(n_perms, dtype=complex) for i in range(d)}
def generate_arrays2(n_perms, d, n):
    return {i: np.ones((n, n_perms), dtype=complex) for i in range(d)}
# def generate_copies(arr, d):
#     return {i: np.array(arr.copy()) for i in range(d)}
def generate_copies(arr, d):
    return {i: [a.copy() for a in arr] for i in range(d)}
def generate_true_false_arrays(permutations, i1_pos, d):
    return {i: permutations[:, i1_pos] == i for i in range(d)}

def generate_V_arrays(V, true_false_arrays):
    return {i: V[:, true_false_arrays[i]] for i in true_false_arrays}

def generate_U_arrays(U, true_false_arrays):
    return {i: U[true_false_arrays[i], :] for i in true_false_arrays}

def generate_ip_splitting(arrays, d):
    return {j: arrays.copy() for j in range(d)}

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


def restructure_arrays_U(U_ips, d, iteration):
    chunk_size = d ** iteration
    U_is = []
    for j in range(d):
        idx = 0
        combined_array = []
        num_rows = U_ips[0][0].shape[0]
        while idx < num_rows:
            combined_array.extend(arr[idx:idx + chunk_size, :] for arr in (U_ips[j][i] for i in range(len(U_ips[j]))))
            idx += chunk_size
        U_is.append(np.concatenate(combined_array, axis=0))
    return U_is

    
def optimize_svd(U, V, threshold_factor):
    S1 = np.array([1, 2])
    S2 = np.array([1])
    # first_iteration = True
    while len(S1) != len(S2):
        # SVD on V
        A, S1, Vh = np.linalg.svd(V, full_matrices=False)
        threshold = S1[0] * threshold_factor
        thresh = S1 > threshold
        # Apply threshold
        S1 = S1[thresh]
        # if first_iteration:
        #     S1s.append(S1[0])
        #     first_iteration = False  # Update the flag after the first time
        A = A[:, thresh]
        V = Vh[thresh, :]
        # Update U
        S = np.diag(S1)
        U = U @ A @ S
        #SVD on U
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
        # print('max S1 val from SVD is:', np.max(S1))
        # print('max S2 val from SVD is:', np.max(S2))

    # print('SVD ended, equal lengths')
    return U, V




#%%
matrix_no = 8
indices_no = int(L/matrix_no)
length = indices_no
indices = (np.arange(L) + 1).astype('str')[::-1] # first L/2 indices correspond to U and rest to V
permutations = np.array(list(generate_permutations(d, int(length))), dtype=np.int32)
split_point = (L ) // 4

dic = {}
value = []
tfinal=50
step_no = int(tfinal/dt)
# step_no = 22
n_perms=int(d**(indices_no))
threshold_factor=1e-8


V_matrices = [np.ones((1, n_perms), dtype=np.complex128) for i in range(matrix_no)]
A_matrices = [np.ones((1), dtype=np.complex128) for i in range(matrix_no)]
# for matrix in range(1, matrix_no+1):
#     dic[f'V{matrix}'] = np.ones((1, n_perms), dtype=np.complex128)    
    
for i in range(d):
    V_matrices[0][0][i::d]=M1[i,exc_channel]
    # dic['V1'][0][i::d]=M1[i,exc_channel]
    
times = np.array([dt*i for i in range(step_no+2)])       

P = []
if exc_channel==measure_channel:
    P.append(np.array([1+0j]))
else:  
    P.append(np.array([0+0j]))
if correlator=='LP':
    P.append(np.array([exp_k0_factor*M1[measure_channel,exc_channel]]))
if correlator=='NQD':
    P.append(np.array([exp_k0_factor[measure_channel]*M1[measure_channel,exc_channel]]))

from joblib import Parallel, delayed
from scipy import linalg
from sklearn.utils.extmath import randomized_svd

def optimize_svd(V,threshold_factor,step):
   
    # SVD on V            
    A, S1, Vh = linalg.svd(V, full_matrices=False)
    threshold = S1[0] * threshold_factor
    thresh = S1 > threshold
    # print('S1 length before:', len(S1))
    S1 = S1[thresh]
    # print('S1 length after:', len(S1))
    A = A[:, thresh] 
    V = Vh[thresh, :]
    # S1 = S1[:4]
    # A = A[:, :4] 
    # V = Vh[:4, :]
    
    # Update U
    S = np.diag(S1)
    if step % 1 == 0:
        V = S @ V 
    else:
        A = A @ S
    if abs(V[0][0]) > 1e6:
        value = 2.5
    else:
        value = 1.3
    print(f"\n SVD each matrix {np.round((time.time() - start_time), 2)} seconds")  
    return V/value, A*value


# def optimize_svd_parallel(V_matrices, threshold_factor, step, n_jobs=-1):
#     # n_jobs=-1 uses all available processors
#     results = Parallel(n_jobs=n_jobs)(delayed(optimize_svd)(V, threshold_factor, step) for V in V_matrices)
#     return results
# import dask.array as da

# def optimize_svd_dask(V, threshold_factor, step):
#     # Convert to dask array
#     dV = da.from_array(V, chunks=(1000, 1000))  # Adjust chunk size based on your matrix
    
#     # Compute SVD in parallel
#     A, S1, Vh = da.linalg.svd(dV, full_matrices=False)
    
#     # Convert back to numpy for thresholding
#     A, S1, Vh = A.compute(), S1.compute(), Vh.compute()
    
#     threshold = S1[0] * threshold_factor
#     thresh = S1 > threshold
#     S1 = S1[thresh]

#     A = A[:, thresh] 
#     V = Vh[thresh, :]
#     S = np.diag(S1)
#     A = A @ S
#     return V, A

start_time = time.time() # time at which function is called


for step in tqdm(range(step_no),position=0, leave=True):     
   
    print('STEP NUMBER IS:', step+1)
    values = []
    if step!=0:
        indices = np.roll(indices, -1) #rolling the indices to simulate the remapping i4i3 i2i1 -> i3i2 i1i4
    
    i2_matrix1 = []
    V_indices2 = []
    
    i1_matrix_pos = None
    i1_pos = None
    matrix_numbers = np.arange(matrix_no)
    for matrix in range(0, matrix_no):
        # print('Matrix loop stage:', matrix)
        # V = dic[f'V{matrix}']
        V = V_matrices[matrix]
         
        if matrix == 0:
            V_indices = indices[int(-L/matrix_no):]
        else:
            # V_indices = indices[int(-L/matrix_no)*(matrix):int(-L/matrix_no)*(matrix-1)]
            V_indices = indices[int(-L/matrix_no)*(matrix+1):int(-L/matrix_no)*(matrix)]
            
        V_indices2.append(V_indices)
        
        if i1_matrix_pos is None:
            if '1' in V_indices:
                i1_matrix_pos = matrix
        
                i1_pos = np.where(V_indices == '1')[0][0]
                if i1_pos is not None:
                    true_false_arrays = generate_true_false_arrays(permutations, i1_pos, d)        

    # print(V_indices2, i1_matrix_pos, i1_pos)
        
        # if i1 is located on the right (i2,i1), it alternates as i1= 0101 across the 4 permutations here, if i1 is the left index (i1,i2) then it varies as i1=0011
        # print(f"\n  Step1: after Arsenis' addition {np.round((time.time() - start_time), 2)} seconds")
 ##########################################################################################################################       
        

    QV = np.ones(n_perms, dtype=complex)   
    for j, V_index in enumerate(V_indices2[i1_matrix_pos]):
        index_val = list(permutations[:, j])  #values of the index across permutations
        V_index = int(V_index)
        
        V_j = V_index - 2 if V_index != 1 else None
        index_np = np.array(index_val)
        for i in range(d):       
            if V_j is not None:
                #Use the precomputed NumPy array slice
                QV[true_false_arrays[i]] *= Qlist[V_j][index_np[true_false_arrays[i]], i]                                 
    Vs_shape_last = V_matrices[i1_matrix_pos].shape[0]         
    for j in range(Vs_shape_last):
        V_matrices[i1_matrix_pos][j, :] *= QV                        
    V_i_splitting=generate_V_arrays(V_matrices[i1_matrix_pos],true_false_arrays)

    #this is where we apply Q_p containing K^(L) cumulant elements, to V, however p can be 0,1,2, so we will split our current V into 3.
    V_ips = [generate_ip_splitting(V_i_splitting[i], d) for i in range(d)]
    Qlist_last = Qlist[-1]
    for i in range(d):
        for p in range(d):
            V_ips[i][p] *= Qlist_last[p,i]
          
    V_is = restructure_arrays(V_ips,d,step%indices_no)
    V_matrices[i1_matrix_pos] = np.vstack([V_is[i] for i in range(len(V_is))])       

    print(f"\n  Q on i_1 matrix applied at {np.round((time.time() - start_time), 2)} seconds")  

    ############
    Vs = generate_copies([V_matrices[i] for i in matrix_numbers[np.arange(len(V_matrices)) != i1_matrix_pos]], d)
    Q_Vs = generate_arrays2(n_perms, d, matrix_no-1)
    V_indices_no_i1 = [V_indices2[i] for i in matrix_numbers[np.arange(len(V_matrices)) != i1_matrix_pos]]
    for j in range(len(V_indices)):
        index_val = list(permutations[:, j])  #values of the index across permutations
        V_index = np.array([int(index[j]) for index in V_indices_no_i1]) 
        V_j = list(V_index - 2) #if V_index != '1' else None
        index_np = np.array(index_val)
        for i in range(d):
            Q_Vs[i] *= np.array(Qlist)[V_j][:,index_val,i]        #  [:,index_val,i], 0th value corresponds to matriceses, first to rows, second to columns
    Vs_shape_last = Vs[0][0].shape[0]  
    for i in range(d):
        for number in range(matrix_no - 1):
        # Multiply all rows at once for each matrix
            Vs[i][number] = Vs[i][number] * Q_Vs[i][number]
    # for i in range(d):  # Avoid repeated indexing
    #     # Vs[i] = [arr[:, j, :] for arr in Vs[i]]
    #     for j in range(Vs_shape_last):
    #         # Vs[i][:, j, :] *= Q_Vs[i]
    #         # Vs[i] = [arr[j, :] * Q_Vs[i] for arr in Vs[i]]
    #         Vs[i][:, j, :] = [Vs[i][number][j, :] * Q_Vs[i][number] for number in range(matrix_no - 1)]

    V = [np.vstack([Vs[i][j] for i in range(len(Vs))]) for j in range(matrix_no - 1)]
    del Vs, Q_Vs
    print(f"\n  Q on other matrixs applied at {np.round((time.time() - start_time), 2)} seconds")  
    
    
    for j, i in enumerate(matrix_numbers[np.arange(len(V_matrices)) != i1_matrix_pos]):
        V_matrices[i] = V[j]

    del V
    print(f"\n Redefining V_matrices {np.round((time.time() - start_time), 2)} seconds")
    for matrix in range(0, matrix_no):
        i2_matrix = (((step + 1) % L) // indices_no)
        V_indices = V_indices2[matrix]
        if '2' in V_indices:
            i2_pos = np.where(V_indices == '2')[0][0]
            desired_perm = np.ones([1, indices_no])
            desired_perm[:, i2_pos] = 0
            col_pos = np.where(np.all(permutations == desired_perm, axis=1))[0][0]
            i2_matrix1.append(col_pos)
        else:
            i2_matrix1.append(-1)       
        if matrix == i2_matrix:
            values.append(V_matrices[matrix][:, col_pos])
        else:
            values.append(V_matrices[matrix][:, -1])  
        # print(f"\n Finding i2  {np.round((time.time() - start_time), 2)} seconds")   
        
        V_matrices[matrix], A_matrices[matrix] = optimize_svd(V_matrices[matrix], threshold_factor, step)
    # SVD_results = optimize_svd_parallel(V_matrices, threshold_factor, step)  
    # # print(f"\n SVD each matrix {np.round((time.time() - start_time), 2)} seconds")  
    # for matrix in range(0,matrix_no):
    #     V_matrices[matrix]=SVD_results[matrix][0]
    #     A_matrices[matrix]=SVD_results[matrix][1]
    
    
   
        
    # print(f"\n  Step0: Applied Qs on all V matrices {np.round((time.time() - start_time), 2)} seconds")  
    # import cupy as cp
    
    if step == 0:
        all_ind = "abcdefghijklmnopqrstuvwxyz"
        einsum_ind = all_ind[:matrix_no ] 
        # einsum_str = f"{einsum_ind} -> "
        
        einsum_str = "".join("z" + einsum_ind[i] + "," for i in range(matrix_no-1)) + "z" + einsum_ind[-1] + "->" + f"{einsum_ind}"
        # R = cp.einsum('ln,lm,lk,lo->nmko', A, B, C, D)
        # R = np.einsum('ln,lm,lk,lo->nmko', A, B, C, D)
        # R =oe.contract('ln,lm,lk,lo->nmko', A, B, C, D)
        # A_matrices2 = [A_matrices[i] for i in range(matrix_no)]
        R =oe.contract(einsum_str, *A_matrices)
        
        inputs = [R] + [V_matrices[i][:, i2_matrix1[i]] for i in range(matrix_no)]
        einsum_str_result = f"{einsum_ind}, " + ", ".join(  einsum_ind[i] for i in range(matrix_no)) + " ->"
        # result = exp_k0_factor*cp.einsum('nmko,n,m,k,o->', R, V1[:, 13], V2[:, -1], V3[:, -1], V4[:, -1])
        # result = exp_k0_factor*oe.contract('nmko,n,m,k,o->', R, V1[:, i2_matrix1[0]], V2[:, -i2_matrix1[1]], V3[:, i2_matrix1[2]], V4[:, i2_matrix1[3]])
        result = exp_k0_factor * oe.contract(einsum_str_result, *inputs)
        del inputs
        # print(f"\n  Step1: Generated first result with einsum, first R {np.round((time.time() - start_time), 2)} seconds")  
    else:   
        # R1 = np.einsum('nmko,na,mb,kc,od->abcd', R, A[:int(A.shape[0]/2), :], B[:int(B.shape[0]/2), :], C[:int(C.shape[0]/2), :], D[:int(D.shape[0]/2), :] )
        # R1 = cp.einsum('nmko,na,mb,kc,od->abcd', R, A[:int(A.shape[0]/2), :], B[:int(B.shape[0]/2), :], C[:int(C.shape[0]/2), :], D[:int(D.shape[0]/2), :] )
        
        output_indices = all_ind[matrix_no : 2 * matrix_no]  # Next N indices for output
        V_terms = [f"{einsum_ind[i]}{output_indices[i]}" for i in range(matrix_no)]  # Each V has 'xi'
        # Construct einsum string dynamically
        einsum_str = f"{einsum_ind}, " + ", ".join(V_terms) + f" -> {output_indices}"
        
        
        inputs = [R] + [A_matrices[i][:int(A_matrices[i].shape[0]/2), :] for i in range(matrix_no)]
        
        # R1 = oe.contract('nmko,na,mb,kc,od->abcd', R, A[:int(A.shape[0]/2), :], B[:int(B.shape[0]/2), :], C[:int(C.shape[0]/2), :], D[:int(D.shape[0]/2), :] )
        R1 = oe.contract(einsum_str, *inputs)
        del inputs
        print(f"\n   R1 calculation {np.round((time.time() - start_time), 2)} seconds")  
        
        inputs = [R] + [A_matrices[i][int(A_matrices[i].shape[0]/2):, :] for i in range(matrix_no)]
        # R2 = np.einsum('nmko,na,mb,kc,od->abcd', R, A[int(A.shape[0]/2):, :], B[int(B.shape[0]/2):, :], C[int(C.shape[0]/2):, :], D[int(D.shape[0]/2):, :]) 
        # R2 = cp.einsum('nmko,na,mb,kc,od->abcd', R, A[int(A.shape[0]/2):, :], B[int(B.shape[0]/2):, :], C[int(C.shape[0]/2):, :], D[int(D.shape[0]/2):, :]) 
        # R2 =oe.contract('nmko,na,mb,kc,od->abcd', R, A[int(A.shape[0]/2):, :], B[int(B.shape[0]/2):, :], C[int(C.shape[0]/2):, :], D[int(D.shape[0]/2):, :])
        R2 =oe.contract(einsum_str, *inputs)
        del inputs
        print(f"\n   R2 calculation {np.round((time.time() - start_time), 2)} seconds")  

        R = R1 + R2
        del R1, R2
        # print('R shape before truncation', R.shape)
        
        
        print('R shape: ', R.shape)

        # result = exp_k0_factor*np.einsum('abc,a,b,c->', R, V1[:, -1], V2[:, 2], V3[:, -1])
        # result = exp_k0_factor*np.einsum('abcd,a,b,c,d->', R, V1[:, i2_matrix1[0]], V2[:, i2_matrix1[1]], V3[:, i2_matrix1[2]], V4[:, i2_matrix1[3]])
        
        inputs = [R] + [V_matrices[i][:, i2_matrix1[i]] for i in range(matrix_no)]
        result = exp_k0_factor*oe.contract(einsum_str_result, *inputs)
        del inputs
        print(f"\n   result einstum {np.round((time.time() - start_time), 2)} seconds")  
    P.append(result)

    # value.append(exp_k0_factor*np.sum(np.prod(np.stack(values), axis=0)))
    # print('STEP:', step, 'VALUE: ', value[-1])
    
# print(P_original[2:2+step_no])
# errors=np.abs(P-P_original[:2+step_no])

# results=cp.asarray(results)
# results=results.get()
plt.figure(10) 
plt.plot(times, np.abs(P), label='Opt')   
plt.legend(loc='best')
plt.yscale('log')    
# P_80=P
# times_80=times
#%%%


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
    
    
#%%
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
        w2s.append(eigenvalues[0]+ 0*1j)            
        if gd < g1:
            w1s.append(eigenvalues[1] + 0*1j)   
            w3s.append(eigenvalues[2] + 0*1j) 
        else:
            w1s.append(eigenvalues[2] + 0*1j)   
            w3s.append(eigenvalues[1] + 0*1j)
        
        
        # w1s.append(0 + 0*1j)                
        # w2s.append(0+ 0*1j)      
        # w3s.append(0 + 0*1j) 
    
    if no_of_QDs==1 and cavity==1:
        # print('QD-cavity model')
        A1s.append(0- 0*1j)
        A2s.append(0 - 0*1j)            
        w1s.append(eigenvalues[0] + 0*1j)                
        w2s.append(eigenvalues[1]+ 0*1j)      

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
        Pnlongt=np.real(P[np.where(times>t0+3*tauib)]) #extracting only the longt behaviour
        tmlongt=times[np.where(times>t0+3*tauib)] #extracting only the longt behaviour
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
        Pnlongt=P[np.where(times>t0+3.0*tauib)] #extracting only the longt behaviour
        tmlongt=times[np.where(times>t0+3.0*tauib)] #extracting only the longt behaviour
        Triexp=lmfit.Model(Triexponential)
        k=0
        guesses= Triexp.make_params(A1_re=np.real(A1s[k]), A1_im=-np.imag(A1s[k]), A2_re=np.real(A2s[k]) , A2_im=-np.imag(A2s[k]), A3_re=np.real(A3s[k]) , A3_im=-np.imag(A3s[k]), w1_re=np.real(w1s[k]), w1_im=-np.imag(w1s[k]) , w2_re=np.real(w2s[k]), w2_im=-np.imag(w2s[k]), w3_re=np.real(w3s[k]), w3_im=-np.imag(w3s[k]))
        result= Triexp.fit(Pnlongt, params=guesses, t=tmlongt, method='leastsq', verbose=True) #fit of current calculated P
        fit_sametimes=Triexp.eval(params=result.params, t=times)
     
                    
        
        
    elif (no_of_QDs==2 and cavity==1 and g1==0) or (no_of_QDs==1 and cavity==1):
        Pnlongt=P[np.where((times > t0 + 2 * tauib) & (times < 200 * tauib))] #extracting only the longt behaviour
        tmlongt=times[np.where((times > t0 + 2 * tauib) & (times < 200 * tauib))] #extracting only the longt behaviour
        biexp=lmfit.Model(biexponential)

        k=0
        guesses= biexp.make_params(A1_re=np.real(A1s[k]), A1_im=-np.imag(A1s[k]), A2_re=np.real(A2s[k]) , A2_im=-np.imag(A2s[k]), w1_re=np.real(w1s[k]), w1_im=-np.imag(w1s[k]) , w2_re=np.real(w2s[k]), w2_im=-np.imag(w2s[k]))
        # guesses= biexp.make_params(A1_re=np.real(A1s[k]), A1_im=-np.imag(A1s[k]), A2_re=np.real(A2s[k]) , A2_im=-np.imag(A2s[k]), w1_re=np.real(w1s[k]), w2_re=np.real(w2s[k]), w2_im=-np.imag(w2s[k]))
        # guesses.add('w1_im',  min=guesses['w2_im'].value + 0.0000000005)
        # tfit=np.linspace(0,100,10000)
        result= biexp.fit(Pnlongt, params=guesses, t=tmlongt, method='leastsq', verbose=True) #fit of current calculated P
        fit_sametimes=biexp.eval(params=result.params, t=times)
       
        
        
    FitParameters_all.append(result.best_values)
    errors = {param: result.params[param].stderr for param in result.params}
    fit_errors.append(errors)
    return tmlongt, Pnlongt, fit_sametimes

tfit=np.linspace(0,100,10000)
avg_error=[]

tmlongt, Pnlongt, fit_sametimes= fitprocedure(g,g1,g2,gd,r0,t0,P,times)
fiterror_ours=(P-fit_sametimes)/P


fig1 = plt.figure( figsize=(4.5,3),dpi=150)
bb = fig1.add_subplot(1, 1, 1)     
bb.plot(np.abs(times),np.abs(P),'r-',markersize='1',linewidth='0.4', label=f'SVD, L={L}')
# bb.plot(np.abs(times),np.abs(fit_sametimes),'b--',label=f'fit, L={L}') 
bb.plot(times,np.abs(fit_sametimes),'b--',label=f'fit, L={L}')  
plt.yscale('log')
print(FitParameters_all)
  
    
















































































