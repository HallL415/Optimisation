'''
params = Parameters() loads the parameters set in params.py with default values
update params with, for example: params.update(L=40, matrix_no=6, r0=5, g=200)
Note that L and matrix_no must be trialled to find optimal computational time/RAM usage
generally low L (sub 50) should use matrix_no=4, and larger L (50-100+) should use
matrix_no=6,8,10 depending on the system.

To compute system dynamics use: P, times = Compute_dynamics().

The data can then be fed into a fitting function called fitprocedure(P,times) 
to extract the mixed states parameters, such as the dephasing rates.

One can extract the fit parameters using extract_parameters

If parameters have been extracted across multiple L (neighbors) values,
a PL fit can be applied using perform_PL_fits, this provides the extrapolated
data for any extracted parameter, such as the dephasing rate for L=infinity neighbors
'''

import numpy as np
import gc
import lmfit
from scipy.optimize import curve_fit
import opt_einsum as oe
import itertools
from params import Parameters
from Functions import LFpop, LFpol, forster, LFpol_qdqdcav, DiagM_qdcav, S_inin, S_inim, PolaronShift, PolaronShift_inim, phi_inin, phi_inim, K11_smartie, K12_smartie, Kbb2, analytics_bareg, analytics_modified, QDQD_analytics_smartie, FGR_smartie, FGR_spherical, FGR_qdcav_spherical, FGR_qdcav_spherical_det
import matplotlib.pyplot as plt
import time
from tqdm import tqdm 
import os
# import gc
# gc.collect()
# import psutil
# process = psutil.Process(os.getpid())
# mem_info = process.memory_info()
# print(f"Memory used: {mem_info.rss / 1024 ** 2:.2f} MB")



path=os.getcwd()
params = Parameters()
if params.correlator=='NQD':
    from Forster_FGR import Gamma1_FGR_det, Gamma2_FGR_det, N21ampli, GammaPh, Gamma1_FGR_det_nrm, Gamma2_FGR_det_nrm, Pan_NQD, Panalyt2

print_times=0
#GPU acceleration, offers speed improvements if L or matrix_no is large, however need a good GPU or runs out of memory quickly
use_cupy=1
if use_cupy==1:
    import cupy as cp
   

def generate_permutations(d,length):
    for perm in itertools.product(list(range(d)), repeat=length):
        yield perm

def generate_arrays(n_perms, d):
    return {i: np.ones(n_perms, dtype=complex) for i in range(d)}

def generate_copies(arr, d):
    return {i: [a.copy() for a in arr] for i in range(d)}

def generate_true_false_arrays(permutations, i1_pos, d):
    return {i: permutations[:, i1_pos] == i for i in range(d)}

def generate_V_arrays(V, true_false_arrays):
    return {i: V[:, true_false_arrays[i]] for i in true_false_arrays}

# def generate_ip_splitting(arrays, d):
#     return {j: arrays.copy() for j in range(d)}

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


# def restructure_arrays(V_ips, d, iteration):
#     chunk_size = d ** iteration
#     num_cols = V_ips[0][0].shape[1]  # e.g., n_perms/d
    
#     V_is = []
#     for j in range(d):
#         # Stack all arrays for this j horizontally
#         stacked = np.hstack([V_ips[j][i] for i in range(d)])  # Shape: (k_i, n_perms)
#         if chunk_size > 1 and stacked.shape[1] > chunk_size:
#             num_chunks = (num_cols + chunk_size - 1) // chunk_size
#             padded_size = num_chunks * chunk_size * d  # Total columns after stacking
#             if padded_size > stacked.shape[1]:
#                 stacked = np.pad(stacked, ((0, 0), (0, padded_size - stacked.shape[1])), mode='constant')
#             # Reshape into (k_i, d, num_chunks, chunk_size), then rearrange
#             reshaped = stacked.reshape((stacked.shape[0], d, num_chunks, chunk_size))
#             # Transpose to (k_i, num_chunks, d, chunk_size), then flatten to (k_i, n_perms)
#             reshaped = reshaped.transpose((0, 2, 1, 3)).reshape((stacked.shape[0], -1))
#             V_is.append(reshaped[:, :num_cols * d])  # Trim to original n_perms
#         else:
#             V_is.append(stacked)  # No chunking needed
#     return V_is



#%%
def Compute_dynamics():
    if params.no_of_QDs==1:
        print('Computing the {} dynamics up to {}ps using {} neighbours using a SVD threshold {} with excitation and measurement in channels {} and {}, respectively. \n The system has 1 QD and 1 cavity with coupling strength {:.0f}micro eV, and environment temperature {}K.'.format(params.correlator, params.tfinal, params.L, params.threshold_factor, params.exc_channel, params.measure_channel, params.g, params.T))
    if params.no_of_QDs==2 and params.cavity==1:
       print('Computing the {} dynamics up to tf={}ps using {} neighbours using a SVD threshold {} '
             'with excitation and measurement in channels {} and {}, respectively.\n'
             'The system has 2 QDs and 1 cavity with direct coupling strength g={:.0f} micro eV and '
             'cavity coupling strengths g1={:.0f} micro eV and g2={:.0f} micro eV, '
             'with environment temperature T={}K.'.format(
                 params.correlator, params.tfinal, params.L, params.threshold_factor,
                 params.exc_channel, params.measure_channel, params.g, params.g1, params.g2, params.T))
    if params.no_of_QDs==2 and params.cavity==0:
       print('Computing the population dynamics up to tf={}ps using {} neighbours using a SVD threshold {} '
             'with excitation and measurement in channels {} and {}, respectively.\n'
             'The system has 2 QDs with direct coupling strength g={:.0f} micro eV with environment temperature T={}K.'.format(
                  params.tfinal, params.L, params.threshold_factor,
                 params.exc_channel, params.measure_channel, params.g, params.T))
       
    if use_cupy==1:
        def optimize_svd(V, threshold_factor, step):  
            # Perform SVD on the GPU
            A, S1, Vh = cp.linalg.svd(V, full_matrices=False)
            
            # Apply thresholding
            threshold = S1[0] * threshold_factor
            thresh = S1 > threshold
            S1 = S1[thresh]
            trunc = len(S1)
        
            A = A[:, thresh] 
            V = Vh[thresh, :]
            S = cp.diag(S1)
        
            # Matrix update step
            # if step % 1 == 0:
            V = S @ V 
            # else:
            #     A = A @ S
        
            # Conditional check (move data to CPU only for scalar check)
            if abs(V[0, 0].get()) > 1e6:
                value = 2.5
            else:
                value = 1.3
        
            # Return rescaled matrices
            return V / value, A * value, trunc
    else:
        def optimize_svd(V,threshold_factor,step):  
            A, S1, Vh = np.linalg.svd(V, full_matrices=False)
            threshold = S1[0] * threshold_factor
            thresh = S1 > threshold
            S1 = S1[thresh]
            trunc=len(S1)
            A = A[:, thresh] 
            V = Vh[thresh, :]
            S = np.diag(S1)
            # if step % 1 == 0:
            V = S @ V 
            # else:
            #     A = A @ S
            # value=1
            if abs(V[0][0]) > 1e6:
                value = 2.5
            else:
                value = 1.3
            return V/value, A*value, trunc
    
    
    indices_no = (params.L // params.matrix_no) * np.ones(params.matrix_no, dtype=int)
    RP = params.L % params.matrix_no
    n_perms=int(params.d**(indices_no[0]))
    V_matrices = [np.ones((1, n_perms), dtype=np.complex128) for i in range(params.matrix_no)]
    A_matrices = [np.ones((1), dtype=np.complex128) for i in range(params.matrix_no)]
    if RP > 0:
        indices_no1 = params.L // params.matrix_no + 1
        for j in range(RP):
            V_matrices[j] = np.ones((1, int(params.d**(indices_no1))), dtype=np.complex128)
            indices_no[j] = indices_no1 
    permutations = {}
    for i in indices_no:
        permutations[f'{i}'] = np.array(list(generate_permutations(params.d, int(i))), dtype=np.int32)
    indices = (np.arange(params.L) + 1).astype('str')[::-1] # first L/2 indices correspond to U and rest to V

    for i in range(params.d):
        V_matrices[0][0][i::params.d]=params.M1[i,params.exc_channel]

    step_no = int(params.tfinal/params.dt)    
    times = np.array([params.dt*i for i in range(step_no+2)])       
    
    P = []
    if params.exc_channel==params.measure_channel:
        P.append(1+0j)
    else:  
        P.append(0+0j)
    if params.correlator=='LP':
        P.append(params.exp_k0_factor*params.M1[params.measure_channel,params.exc_channel])
    if params.correlator=='NQD':
        P.append(params.exp_k0_factor[params.measure_channel]*params.M1[params.measure_channel,params.exc_channel])
         
    
    for step in tqdm(range(step_no),position=0, leave=True):     
        # print(' \n STEP NUMBER IS:', step+1)
        values = []
        if step!=0:
            indices = np.roll(indices, -1) #rolling the indices to simulate the remapping i4i3 i2i1 -> i3i2 i1i4
        i2_matrix1 = []
        V_indices2 = []
        V = []
        i1_matrix_pos = None
        i1_pos = None
        matrix_numbers = np.arange(params.matrix_no)
        # Q_application_start_time=time.time()
        for matrix in range(0, params.matrix_no):
            permutations_matrix = permutations[f'{indices_no[matrix]}']
            n_perms=int(params.d**(indices_no[matrix]))
            if matrix == 0:
                V_indices = indices[-indices_no[matrix]:]
            else:
                V_indices = indices[-np.sum(indices_no[:matrix+1]):-np.sum(indices_no[:matrix])]
                
            V_indices2.append(V_indices)   
            if i1_matrix_pos is None:
                if '1' in V_indices:
                    i1_matrix_pos = matrix
            
                    i1_pos = np.where(V_indices == '1')[0][0]
                    if i1_pos is not None:
                        true_false_arrays = generate_true_false_arrays(permutations_matrix, i1_pos, params.d)        
            if matrix == i1_matrix_pos:
                QV_first_generate_time=time.time()
                QV = np.ones(n_perms, dtype=complex)   
                for j, V_index in enumerate(V_indices):
                    index_val = list(permutations_matrix[:, j])  #values of the index across permutations
                    V_index = int(V_index)
                    
                    V_j = V_index - 2 if V_index != 1 else None
                    index_np = np.array(index_val)
                    for i in range(params.d):       
                        if V_j is not None:
                            #Use the precomputed NumPy array slice
                            QV[true_false_arrays[i]] *= params.Qlist[V_j][index_np[true_false_arrays[i]], i]  
                QV_first_generate_time_end=time.time() 
                if print_times==1:
                    print(f"QV for matrix{matrix} with i1 generated in  {QV_first_generate_time_end -  QV_first_generate_time:.2f} seconds")

                QV_apply_i1_time=time.time()        
                Vs_shape_last = V_matrices[i1_matrix_pos].shape[0]         
                for j in range(Vs_shape_last):
                    V_matrices[i1_matrix_pos][j, :] *= QV  
                QV_apply_i1_end=time.time()   
                if print_times==1:
                    print(f"QV for matrix{matrix} with i1 applied in  {QV_apply_i1_end -  QV_apply_i1_time:.2f} seconds")
                
                V_splitting_time=time.time()
                V_i_splitting=generate_V_arrays(V_matrices[i1_matrix_pos],true_false_arrays)
                V_splitting_time_end=time.time()
                if print_times==1:
                    print(f"V_i splitting matrix{matrix} with i1 done in  {V_splitting_time_end -  V_splitting_time:.2f} seconds")

                Vips_splitting_time=time.time()
                # V_ips = [generate_ip_splitting(V_i_splitting[i], params.d) for i in range(params.d)]
                V_ips = [[V_i_splitting[i].copy() for _ in range(params.d)] for i in range(params.d)]
                V_ips_splitting_time_end=time.time()
                if print_times==1:
                    print(f"V_ips splitting matrix{matrix} with i1 done in  {V_ips_splitting_time_end -  Vips_splitting_time:.2f} seconds")
                
                QP_multiplying_time=time.time()
                Qlist_last = params.Qlist[-1]
                for i in range(params.d):
                    for p in range(params.d):
                        V_ips[i][p] *= Qlist_last[p,i]
                QP_multiplying_time_end=time.time()
                if print_times==1:
                    print(f"Qp multiplying matrix{matrix} with i1 done in  {QP_multiplying_time_end -  QP_multiplying_time:.2f} seconds")

                V_restructure_time=time.time()
                V_is = restructure_arrays(V_ips,params.d,indices_no[matrix] - i1_pos - 1)
                V_matrices[i1_matrix_pos] = np.vstack([V_is[i] for i in range(len(V_is))]) 
                V_restructure_end_time=time.time()
                if print_times==1:
                    print(f"Restructure V matrix{matrix} with i1 done in  {V_restructure_end_time -  V_restructure_time:.2f} seconds")
            else:
            ############
                QV_generate_time=time.time()
                Vs = generate_copies(V_matrices[matrix], params.d)
                Q_Vs = generate_arrays(n_perms, params.d)
                for j in range(len(V_indices)):
                    index_val = list(permutations_matrix[:, j])  #values of the index across permutations
                    V_index = int(V_indices[j])
                    V_j = V_index - 2 #if V_index != '1' else None
                    index_np = np.array(index_val)
                    for i in range(params.d):
                        Q_Vs[i] *= np.array(params.Qlist)[V_j][index_val,i]
                
                QV_generate_time_end=time.time()
                if print_times==1:
                    print(f"QV generated of matrix {matrix} in  {QV_generate_time_end - QV_generate_time:.2f} seconds")
                
                Q_application_start_time = time.time()
                Vs_shape_last = Vs[0][0].shape[0]  
                for i in range(params.d):
                    # Multiply all rows at once for each matrix
                    Vs[i] = Vs[i] * Q_Vs[i]    
                V.append(np.vstack([Vs[i] for i in range(len(Vs))]))
                del Vs, Q_Vs
                Q_application_end_time = time.time()
                if print_times==1:
                    print(f"QV applied on matrix{matrix} in {Q_application_end_time - Q_application_start_time:.2f} seconds")
        # Q_application_end_time=time.time()
        # print(f"\n  Q on all matrices applied in {round(Q_application_end_time - Q_application_start_time,2)} seconds")  
                
            
        for j, i in enumerate(matrix_numbers[np.arange(len(V_matrices)) != i1_matrix_pos]):
            V_matrices[i] = V[j]
    
        del V
        
        
        # if step % 30 == 0:
        #     global_V_matrices.append([v.copy() for v in V_matrices])
            
            
        ### CuPy SVD ###
        if use_cupy==1:
            V_results = []
            A_results = []
            truncations = []
            SVD_start_time=time.time()
            for matrix in range(params.matrix_no):
                V_gpu = cp.array(V_matrices[matrix])  # Individual transfer
                V_out, A_out, trunc = optimize_svd(V_gpu, params.threshold_factor, step)
                V_results.append(V_out)
                A_results.append(A_out)
                truncations.append(trunc)        
            V_matrices = [V.get() for V in V_results]  # List of (k_i, n_perms)
            A_matrices = [A.get() for A in A_results] 
            SVD_end_time=time.time()
            if print_times==1:
                print(f"\n SVD calcs done {round(SVD_end_time- SVD_start_time,2)} seconds")
        else:
        # numpy SVD ###
            SVD_start_time=time.time()
            for matrix in range(params.matrix_no):
                V_matrices[matrix], A_matrices[matrix], trunc = optimize_svd(V_matrices[matrix], params.threshold_factor, step)
            SVD_end_time=time.time()
            if print_times==1:
                print(f"\n SVD calcs done {round(SVD_end_time- SVD_start_time,2)} seconds")
        
        #finding i2 position for calculation
        for matrix in range(0, params.matrix_no):
            permutations_matrix = permutations[f'{indices_no[matrix]}']
            V_indices = V_indices2[matrix]
            # i2_matrix = (((step + 1) % L) // indices_no[matrix])
            
            if '2' in V_indices:
                i2_pos = np.where(V_indices == '2')[0][0]
                desired_perm = params.phonon_uncoupled_mode*np.ones([1, indices_no[matrix]])
                desired_perm[:, i2_pos] = 0
                col_pos = np.where(np.all(permutations_matrix == desired_perm, axis=1))[0][0]
                i2_matrix1.append(col_pos)
                values.append(V_matrices[matrix][:, col_pos])
            else:
                i2_matrix1.append(-1)  
                values.append(V_matrices[matrix][:, -1]) 
            
            # V_matrices[matrix], A_matrices[matrix], trunc = optimize_svd(V_matrices[matrix], params.threshold_factor, step)
            
 
        # print(f"\n SVD calcs done {np.round((time.time() - start_time), 2)} seconds")
        einsum_start_time=time.time()    
        if step == 0:
            all_ind = "abcdefghijklmnopqrstuvwxyz"
            einsum_ind = all_ind[:params.matrix_no ]      
            einsum_str = "".join("z" + einsum_ind[i] + "," for i in range(params.matrix_no-1)) + "z" + einsum_ind[-1] + "->" + f"{einsum_ind}"
            # if use_cupy==1:
            #     R =oe.contract(einsum_str, *A_matrices, backend='cupy')  
            # else: 
            R =oe.contract(einsum_str, *A_matrices)

            inputs = [R] + [V_matrices[i][:, i2_matrix1[i]] for i in range(params.matrix_no)]
            einsum_str_result = f"{einsum_ind}, " + ", ".join(  einsum_ind[i] for i in range(params.matrix_no)) + " ->"
            if use_cupy==1:
                inputs = [cp.array(x, dtype=cp.complex128) for x in inputs]
                result = params.exp_k0_factor * oe.contract(einsum_str_result, *inputs, backend='cupy')
                result=result.get()
            else:
                result = params.exp_k0_factor * oe.contract(einsum_str_result, *inputs)
            del inputs
        else:   
           ####################################### 
            output_indices = all_ind[params.matrix_no : 2 * params.matrix_no]  # Next N indices for output
            V_terms = [f"{einsum_ind[i]}{output_indices[i]}" for i in range(params.matrix_no)]  # Each V has 'xi'
            einsum_str = f"{einsum_ind}, " + ", ".join(V_terms) + f" -> {output_indices}"
            for j in range(params.d):
                if step%2 == 1: # odd step
                    inputs = [R] + [A_matrices[i][j*int(A_matrices[i].shape[0]/params.d):(j+1)*int(A_matrices[i].shape[0]/params.d), :] for i in range(params.matrix_no)]
                    if j==0:
                        if use_cupy==1:
                            inputs = [cp.array(x, dtype=cp.complex128) for x in inputs]
                            R_temp = oe.contract(einsum_str, *inputs, backend='cupy')
                        else:
                            R_temp = oe.contract(einsum_str, *inputs)


                    else:
                        if use_cupy==1:
                            inputs = [cp.array(x, dtype=cp.complex128) for x in inputs]
                            R_temp +=  oe.contract(einsum_str, *inputs, backend='cupy')
                        else:
                            R_temp +=  oe.contract(einsum_str, *inputs)

                else:
                    inputs = [R_temp] + [A_matrices[i][j*int(A_matrices[i].shape[0]/params.d):(j+1)*int(A_matrices[i].shape[0]/params.d), :] for i in range(params.matrix_no)]
                    if j==0:
                        if use_cupy==1:
                            inputs = [cp.array(x, dtype=cp.complex128) for x in inputs]
                            R = oe.contract(einsum_str, *inputs, backend='cupy')
                        else:
                            R = oe.contract(einsum_str, *inputs)

                    else:
                        if use_cupy==1:
                            inputs = [cp.array(x, dtype=cp.complex128) for x in inputs]
                            R +=  oe.contract(einsum_str, *inputs, backend='cupy')
                        else:
                            R +=  oe.contract(einsum_str, *inputs)

            del inputs
            if step%2 == 1:
                del R
                if print_times==1:
                    print('\n R shape: ', R_temp.shape)
                inputs = [R_temp] + [V_matrices[i][:, i2_matrix1[i]] for i in range(params.matrix_no)]
                if use_cupy==1:
                    inputs = [cp.array(x, dtype=cp.complex128) for x in inputs]
            else:
                del R_temp
                if print_times==1:
                    print('\n R shape: ', R.shape)
                inputs = [R] + [V_matrices[i][:, i2_matrix1[i]] for i in range(params.matrix_no)]
                if use_cupy==1:
                    inputs = [cp.array(x, dtype=cp.complex128) for x in inputs]
            if use_cupy==1:
                result = params.exp_k0_factor*oe.contract(einsum_str_result, *inputs, backend='cupy')
                result=result.get()
            else:
                result = params.exp_k0_factor*oe.contract(einsum_str_result, *inputs)
            
            del inputs
            einsum_end_time=time.time()
            if print_times==1:
                print(f"\n einsum total time {round(einsum_end_time-einsum_start_time,2)} seconds")  
        P.append(result)
    if use_cupy==1:
        cp.get_default_memory_pool().free_all_blocks()
    return np.asarray(P), np.asarray(times)


# params.update(L=30, matrix_no=6, no_of_QDs=2, cavity=1, DvDc=10.5, g=0, g1=500, g2=500, T=20, tfinal=50, threshold_factor=1e-6)
# # # params.update(L=36, matrix_no=4, no_of_QDs=2, cavity=1, g1=500, g2=500, g=0, T=20, r0=10, tfinal=50, threshold_factor=1e-6)

# P, times = Compute_dynamics()

# label = params.generate_label()
# data_to_save = np.column_stack((times, P))
# np.save(path+"/data/"+label+".npy",data_to_save)

# plt.figure(10) 
# plt.plot(times, np.abs(P), label=fr'L={params.L}, matrix_no={params.matrix_no}, g={params.g}, DvDc={params.DvDc}, thresh={params.threshold_factor}')   
# plt.legend(loc='best')
# plt.yscale('log')    



#%%
def fitprocedure(P, times): 
    
    def biexponential(t, A1_re,A1_im,A2_re,A2_im,w1_re,w1_im,w2_re,w2_im):
        return (A1_re - 1j*A1_im)*np.exp(-1j*(w1_re -1j*w1_im)*t) + (A2_re - 1j*A2_im)*np.exp(-1j*(w2_re -1j*w2_im)*t)

    def Triexponential(t, A1_re,A1_im,A2_re,A2_im,A3_re,A3_im,w1_re,w1_im,w2_re,w2_im,w3_re,w3_im):
        return (A1_re - 1j*A1_im)*np.exp(-1j*(w1_re -1j*w1_im)*t) + (A2_re - 1j*A2_im)*np.exp(-1j*(w2_re -1j*w2_im)*t)+ (A3_re - 1j*A3_im)*np.exp(-1j*(w3_re -1j*w3_im)*t)

    def population(t,A,B,C,T1,T2,Rabi,phase):
        return A+B*np.exp(-(T1)*t) + C*np.exp(-(T2)*t)*np.cos(Rabi*t + phase)

    def quintexponential(t, A1_re,A1_im,A2_re,A2_im,A3_re,A3_im,A4_re,A4_im,A5_re,A5_im, w1_re,w1_im,w2_re,w2_im,w3_re,w3_im,w4_re,w4_im, w5_re, w5_im):
        return (A1_re - 1j*A1_im)*np.exp(-1j*(w1_re -1j*w1_im)*t) + (A2_re - 1j*A2_im)*np.exp(-1j*(w2_re -1j*w2_im)*t)+ (A3_re - 1j*A3_im)*np.exp(-1j*(w3_re -1j*w3_im)*t) + (A4_re - 1j*A4_im)*np.exp(-1j*(w4_re -1j*w4_im)*t) + (A5_re - 1j*A5_im)*np.exp(-1j*(w5_re -1j*w5_im)*t)

    def quadexponential(t, A1_re,A1_im,A2_re,A2_im,A3_re,A3_im,A4_re,A4_im, w1_re,w1_im,w2_re,w2_im,w3_re,w3_im,w4_re,w4_im):#,A5_re,A5_im, w5_re,w5_im):
        return (A1_re - 1j*A1_im)*np.exp(-1j*(w1_re -1j*w1_im)*t) + (A2_re - 1j*A2_im)*np.exp(-1j*(w2_re -1j*w2_im)*t)+ (A3_re - 1j*A3_im)*np.exp(-1j*(w3_re -1j*w3_im)*t) + (A4_re - 1j*A4_im)*np.exp(-1j*(w4_re -1j*w4_im)*t) #+ (A5_re - 1j*A5_im)*np.exp(-1j*(w5_re -1j*w5_im)*t)
            
    A1s, A2s, A3s, A4s, A5s, w1s, w2s, w3s, w4s, w5s = [[] for _ in range(10)]
  
    # diagonalise numerically H_0 (no phonons to provide initial guess)
    if params.no_of_QDs==2 and params.cavity==1:
        matrix = np.array([[params.w_qd1, params.g_comp, params.g1_comp],
                   [params.g_comp, params.w_qd2, params.g2_comp],
                   [params.g1_comp, params.g2_comp, params.w_c]])
        eigenvalues = np.linalg.eigvals(matrix)
        eigenvalues=  np.sort(eigenvalues)[::-1]
     
    if params.no_of_QDs==1 and params.cavity==1:   
        matrix = np.array([[params.w_qd1, params.g_comp],
                           [params.g_comp, params.w_c]])
        eigenvalues = np.linalg.eigvals(matrix)
        eigenvalues=  np.sort(eigenvalues)[::-1]
        
    if params.no_of_QDs==2 and params.cavity==1 and params.g1_comp!=0:
        A1s.append(0.5 - 0*1j)
        A2s.append(0.5 - 0*1j)  
        A3s.append(0.5- 0*1j)  
        w1s.append(eigenvalues[0]+ 0*1j)         #upper level   
        if params.g < params.g1:
            w2s.append(eigenvalues[1] + 0*1j)     #middle level
            w3s.append(eigenvalues[2] + 0*1j)     #lower level
        else:
            w2s.append(eigenvalues[1] + 0*1j)   
            w3s.append(eigenvalues[2] + 0*1j)
        
    if params.no_of_QDs==1 and params.cavity==1:
        A1s.append(0- 0*1j)
        A2s.append(0 - 0*1j)            
        w1s.append(eigenvalues[0] + 0*1j)                
        w2s.append(eigenvalues[1]+ 0*1j)      

    if params.no_of_QDs==2 and params.cavity==1 and params.g1==0:
        A1s.append(0.5 - 0*1j)
        A2s.append(0.5 - 0*1j)            
        w1s.append(eigenvalues[0] + 0*1j)                
        w2s.append(eigenvalues[2]+ 0*1j)
     
    if params.no_of_QDs==2 and params.cavity==0:
        LF = LFpop(params.g,  np.real(params.w_qd1), np.real(params.w_qd2), -params.gamma1_comp, -params.gamma2_comp) 
        eigenvalues=np.linalg.eigvals(LF)
        eigenvalues=np.sort(eigenvalues)[::-1]
        A1s.append(0 - 0*1j)
        A2s.append(0.0 - 0*1j)  
        A3s.append(0.0 - 0*1j)  
        A4s.append(0.0 - 0*1j) 
        A5s.append(0.0 - 0*1j) 

        w1s.append(0 + 0*1j)                
        w2s.append(0+ 0*1j)      
        w3s.append(0 + 0*1j) 
        w4s.append(0 + 0*1j) 
        w5s.append(0 + 0*1j) 
        ##########################
        GamPh,RF,dp,dm, lambdp, lambdm=GammaPh(params.r0,params.det,params.T_ps)
        gam1=Gamma1_FGR_det(params.r0,params.det,params.T_ps)/params.hbar 
        gam2=Gamma2_FGR_det(params.r0,params.det,params.T_ps)/params.hbar 
        Ct,AA,ww=Pan_NQD(0,params.r0,params.det,params.T_ps)
     
        A=AA[4]
        C=AA[2]+AA[3]
        B=AA[1]#-A-C #-(AA[1]+AA[3]) # -A-C
        Gamd=(gam1+gam2)
        Gams=(2*Gamd) 
        Phi=0
        pa=np.array([A,B,C,Gams,Gamd,RF,Phi]).real
        pa=np.asarray(pa)
        # print('guess params a,b,c,T1,T2,R,phi:', pa, 'for d=',r0)


    if params.no_of_QDs==2 and params.cavity==0: 
        Pnlongt=np.real(P[np.where(times>params.t0+3*params.tauib)]) #extracting only the longt behaviour
        tmlongt=times[np.where(times>params.t0+3*params.tauib)] #extracting only the longt behaviour
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
    
    elif params.no_of_QDs==2 and params.cavity==1 and params.g1!=0 :
        Pnlongt=P[np.where(times>params.t0+3.0*params.tauib)] #extracting only the longt behaviour
        tmlongt=times[np.where(times>params.t0+3.0*params.tauib)] #extracting only the longt behaviour
        Triexp=lmfit.Model(Triexponential)
        k=0
        guesses= Triexp.make_params(A1_re=np.real(A1s[k]), A1_im=-np.imag(A1s[k]), A2_re=np.real(A2s[k]) , A2_im=-np.imag(A2s[k]), A3_re=np.real(A3s[k]) , A3_im=-np.imag(A3s[k]), w1_re=np.real(w1s[k]), w1_im=-np.imag(w1s[k]) , w2_re=np.real(w2s[k]), w2_im=-np.imag(w2s[k]), w3_re=np.real(w3s[k]), w3_im=-np.imag(w3s[k]))
        result= Triexp.fit(Pnlongt, params=guesses, t=tmlongt, method='leastsq', verbose=True) #fit of current calculated P
        fit_sametimes=Triexp.eval(params=result.params, t=times)
     
    elif (params.no_of_QDs==2 and params.cavity==1 and params.g1==0) or (params.no_of_QDs==1 and params.cavity==1):
        Pnlongt=P[np.where(times > params.t0 + 3 * params.tauib)]#extracting only the longt behaviour
        tmlongt=times[np.where(times > params.t0 + 3 * params.tauib) ] #extracting only the longt behaviour
        biexp=lmfit.Model(biexponential)

        k=0
        guesses= biexp.make_params(A1_re=np.real(A1s[k]), A1_im=-np.imag(A1s[k]), A2_re=np.real(A2s[k]) , A2_im=-np.imag(A2s[k]), w1_re=np.real(w1s[k]), w1_im=-np.imag(w1s[k]) , w2_re=np.real(w2s[k]), w2_im=-np.imag(w2s[k]))
        # guesses= biexp.make_params(A1_re=np.real(A1s[k]), A1_im=-np.imag(A1s[k]), A2_re=np.real(A2s[k]) , A2_im=-np.imag(A2s[k]), w1_re=np.real(w1s[k]), w2_re=np.real(w2s[k]), w2_im=-np.imag(w2s[k]))
        # guesses.add('w1_im',  min=guesses['w2_im'].value + 0.0000000005)
        # tfit=np.linspace(0,100,10000)
        result= biexp.fit(Pnlongt, params=guesses, t=tmlongt, method='leastsq', verbose=True) #fit of current calculated P
        fit_sametimes=biexp.eval(params=result.params, t=times)
      
    FitParameters=result.best_values
    # errors = {param: result.params[param].stderr for param in result.params}
    # fit_errors.append(errors)
    fiterrors=np.abs(fit_sametimes[np.where(times>params.t0+3.0*params.tauib)]-P[np.where(times>params.t0+3.0*params.tauib)])
    fiterrors=np.mean(fiterrors)
    return tmlongt, Pnlongt, fit_sametimes, FitParameters, fiterrors

def compute_fitparameters(Ls):
    """
    Compute or load fit parameters over a range of L values.
    """
    fitparameters = []
    fiterrors=[]
    for L in Ls:
        # Update params with current L
        params.update(L=L)
        label = params.generate_label()
        print(f'Trying to load/compute data for {L} neighbours')
        # Try to load existing data
        try:
            data = np.load(f"{path}/data/{label}.npy", allow_pickle=True)
            P, times = data[:, 1], data[:, 0]
        # If loading fails, compute and save
        except:
            P, times = Compute_dynamics()  # Assumes this is defined globally
            data_to_save = np.column_stack((times, P))
            np.save(f"{path}/data/{label}.npy", data_to_save)
        # Perform fitting
        tmlongt, Pnlongt, fit_sametimes, fitparams, fiterror = fitprocedure(P, times)  # Assumes this is defined globally
        fitparameters.append(fitparams)
        fiterrors.append(fiterror)
        print(f'average fit error for L={L} is {fiterrors}')
        # plt.figure(10)
        # plt.plot(times,np.abs(fit_sametimes), label='fit')
        # plt.plot(times,np.abs(P),label='data')
        # plt.legend(loc='best')
        # plt.yscale('log')
    return fitparameters, fiterrors

# plt.plot(times,np.abs(P))
def extract_parameters(fitparameters):
    # Define the parameter groups and their properties
    param_groups = {
        'w': {
            'keys': ['w1_re', 'w2_re', 'w3_re'],
            'factor': params.hbar * 1e3,
            'conditional': [True, True, params.g1 != 0 and params.no_of_QDs == 2 and params.cavity == 1]
        },
        'gam': {
            'keys': ['w1_im', 'w2_im', 'w3_im'],
            'factor': params.hbar * 1e3,
            'transform': np.abs,
            'conditional': [True, True, params.g1 != 0 and params.no_of_QDs == 2 and params.cavity == 1]
        },
        'A_re': {
            'keys': ['A1_re', 'A2_re', 'A3_re'],
            'factor': 1.0,
            'conditional': [True, True, params.g1 != 0 and params.no_of_QDs == 2 and params.cavity == 1]
        },
        'A_im': {
            'keys': ['A1_im', 'A2_im', 'A3_im'],
            'factor': 1.0,
            'conditional': [True, True, params.g1 != 0 and params.no_of_QDs == 2 and params.cavity == 1]
        }
    }
    
    # Initialize result dictionary with only the arrays that meet their conditions
    results = {}
    for group, props in param_groups.items():
        results[group] = [[] for condition in props['conditional'] if condition]
    
    # Process each fit parameter
    for i in range(len(fitparameters)):
        for group, props in param_groups.items():
            for j, (key, condition) in enumerate(zip(props['keys'], props['conditional'])):
                if condition:
                    value = fitparameters[i][key] * props['factor']
                    if 'transform' in props:
                        value = props['transform'](value)
                    # Append to the correct index based on how many conditions were True before this one
                    idx = sum(1 for c in props['conditional'][:j] if c)
                    results[group][idx].append(value)    
    return results


def PL_fit(ws, selected_range):
    def PowerLawModel(L,omega_infinity,C): 
        #fixing beta= 2
        return omega_infinity + C * np.power(L.astype(float), -2)
    initial_guess_pl = [ws[-1],1]
    params_pl, pcov = curve_fit(PowerLawModel, Ls[selected_range], np.real(ws)[selected_range], p0=initial_guess_pl)
    omega_inf_pl, C_pl = params_pl
    # inf_err = np.sqrt(pcov[0, 0])
    L_ex=np.linspace(10,200,200)
    omega_pl_fit = PowerLawModel(L_ex, *params_pl)
    return omega_inf_pl, C_pl, omega_pl_fit

def perform_PL_fits(data_group, selected_range, names=None):
    fit_results = {}
    # Use provided names or generate default numeric indices
    if names is None:
        names = [str(i) for i in range(len(data_group))]
    else:
        names = names[:len(data_group)]  # Truncate to match data_group length
    
    for name, data in zip(names, data_group):
        omega_inf, C, omega_fit = PL_fit(data, selected_range)
        fit_results[name] = {
            'inf': omega_inf,
            'C': C,
            'fit': omega_fit
        }
    return fit_results

def compute_r0_variation(r0_values, L_range):
    """
    Compute fits and extrapolations for multiple r0 values.
    
    """
    r0_results = {}
    r0_results_raw = {}
    all_results = {}
    # Loop over each r0 value
    for r0 in r0_values:
        
        params.update(r0=r0)   # L=16 as initial value, will be overridden by Ls
        print(f'Running d={params.r0}nm calculation')
        # Compute fit parameters for all L values
        fitparameters, fiterrors = compute_fitparameters(L_range)
        
        # Extract the fit parameters
        results = extract_parameters(fitparameters)
        all_results[r0] =  {
            'w': results['w'],
            'gam': results['gam'],
            'A_re': results['A_re'],
            'A_im': results['A_im']          
        }
        # if params.extrapolation==0:
        if params.g1 > 0:
            r0_results_raw[r0] = {
                'gam1': results['gam'][0][-1],
                'gam2': results['gam'][1][-1],
                'gam3': results['gam'][2][-1]}
   
        if params.g1 == 0:
            r0_results_raw[r0] = {
                'gam1': results['gam'][0][-1],
                'gam2': results['gam'][1][-1] }
    
        if params.extrapolation==1:
            # Define fit range and perform power law fits
            indices = np.arange(len(L_range))[-3:]
            gam_fits = perform_PL_fits(
                results['gam'], 
                slice(min(indices), max(indices) + 1), 
                names=['gam1', 'gam2', 'gam3']
            )
            
            # Store the extrapolated infinity values
            if params.g1 > 0:
                r0_results[r0] = {
                    'gam1_inf': gam_fits['gam1']['inf'],
                    'gam2_inf': gam_fits['gam2']['inf'],
                    'gam3_inf': gam_fits['gam3']['inf']
                }
            if params.g1 == 0:
                if params.g_comp >= 200*1e-3/params.hbar:
                    print('g is more or eq. to 200:', params.g_comp*1e3*params.hbar)
                    r0_results[r0] = {
                        'gam1_inf': gam_fits['gam1']['inf'],
                        'gam2_inf': gam_fits['gam2']['inf']    }  
                if params.g_comp < 200*1e-3/params.hbar:
                    print('g is less than 200:', params.g_comp*1e3*params.hbar)
                    r0_results[r0] = {
                        'gam1_inf': results['gam'][0][-1],
                        'gam2_inf': results['gam'][1][-1]   }  
    
    if params.extrapolation == 1:
        return r0_results, r0_results_raw, all_results
    else:
        return r0_results_raw, all_results

def compute_g_variation(g_values, L_range):
    """
    Compute fits and extrapolations for multiple g values.
    """
    g_results = {}
    g_results_raw = {}
    w_results = {}
    w_results_raw = {}
    all_results = {}
    for g in g_values:
        # Update parameters with current g

        params.update(g=g)     # L=16 as initial value, overridden by L_range
        
        # Compute fit parameters for all L values
        fitparameters, fiterrors = compute_fitparameters(L_range)
        
        # Extract the fit parameters
        results = extract_parameters(fitparameters)
        all_results[g] =  {
            'w': results['w'],
            'gam': results['gam'],
            'A_re': results['A_re'],
            'A_im': results['A_im']          
        }
        
        # if params.extrapolation==0:
        g_results_raw[g] = {
            'gam1': results['gam'][0][0],
            'gam2': results['gam'][1][0]   }  
        w_results_raw[g] = {
            'w1': results['w'][0][-1],
            'w2': results['w'][1][-1]   }  
        
        if params.extrapolation==1:
            # Define fit range (last 3 L values) and perform power law fits
            indices = np.arange(len(L_range))[-3:]
            indicesY = np.arange(len(L_range))[-4:-1]
            gam_fits = perform_PL_fits(
                results['gam'], 
                slice(min(indices), max(indices) + 1), 
                names=['gam1', 'gam2']
            )
            gam_fits_Y = perform_PL_fits(
                results['gam'], 
                slice(min(indicesY), max(indicesY) + 1), 
                names=['gam1', 'gam2']
            )
            print('The L values for fit X are:', L_range[ slice(min(indices), max(indices) + 1)])
            print('The L values for fit Y are:', L_range[ slice(min(indicesY), max(indicesY) + 1)])
            w_fits = perform_PL_fits(
                results['w'], 
                slice(min(indices), max(indices) + 1), 
                names=['w1', 'w2']
            )
            
            if params.g_comp >= 100*1e-3/params.hbar:
                print('g is more or eq. to 100:', params.g_comp*1e3*params.hbar)
                gam1_inf = gam_fits['gam1']['inf']
                gam1_inf_Y = gam_fits_Y['gam1']['inf']
                gam1_error = 0.5 * np.abs((gam1_inf / gam1_inf_Y) - 1) + 0.5 * np.abs((gam1_inf_Y / gam1_inf) - 1)
        
                gam2_inf = gam_fits['gam2']['inf']
                gam2_inf_Y = gam_fits_Y['gam2']['inf']
                gam2_error = 0.5 * np.abs((gam2_inf / gam2_inf_Y) - 1) + 0.5 * np.abs((gam2_inf_Y / gam2_inf) - 1)
        
                g_results[g] = {
                'gam1_inf': gam1_inf,
                'gam2_inf': gam2_inf,
                'gam1_error': gam1_error,
                'gam2_error': gam2_error
                }
                # g_results[g] = {
                #     'gam1_inf': gam_fits['gam1']['inf'],
                #     'gam2_inf': gam_fits['gam2']['inf']
                # }
                w_results[g] = {
                    'w1_inf': w_fits['w1']['inf'],
                    'w2_inf': w_fits['w2']['inf']
                }
            if params.g_comp < 100*1e-3/params.hbar:
                print('g is less than 100:', params.g_comp*1e3*params.hbar)
                g_results[g] = {
                    'gam1_inf': results['gam'][0][-1],
                    'gam2_inf': results['gam'][1][-1]   }  
                w_results[g] = {
                    'w1_inf': results['w'][0][-1],
                    'w2_inf': results['w'][1][-1]   }  
    if params.extrapolation==1:       
        return g_results, w_results, g_results_raw,  w_results_raw, all_results
    else: 
        return g_results_raw, w_results_raw, all_results

def compute_g1g2_variation(g_values, L_range):
    """
    Compute fits and extrapolations for multiple g values.
    """
    g_results = {}
    g_results_raw = {}
    for g1 in g_values:
        g2=g1
        # Update parameters with current g
        params.update(g1=g1, g2=g2)     # L=16 as initial value, overridden by L_range
        
        # Compute fit parameters for all L values
        fitparameters, fiterrors = compute_fitparameters(L_range)
        
        # Extract the fit parameters
        results = extract_parameters(fitparameters)
        # if params.extrapolation == 0:
        # Use the last gamma value for all three
        g_results_raw[g1] = {
            'gam1': results['gam'][0][-1],
            'gam2': results['gam'][1][-1],
            'gam3': results['gam'][2][-1]
        }
        if params.extrapolation == 1:
            # Define fit range (last 3 L values) and perform power law fits
            indices = np.arange(len(L_range))[-3:]
            gam_fits = perform_PL_fits(
                results['gam'], 
                slice(min(indices), max(indices) + 1), 
                names=['gam1', 'gam2', 'gam3']
            )
            
            # Store the extrapolated infinity values
            g_results[g1] = {
                'gam1_inf': gam_fits['gam1']['inf'],
                'gam2_inf': gam_fits['gam2']['inf'],
                'gam3_inf': gam_fits['gam3']['inf']
            }
    if params.extrapolation==1:
        return g_results_raw, g_results
    else:
        return g_results_raw


#%%
params.update(L=16, matrix_no=6, no_of_QDs=1, cavity=1, dotshape='smartie', l=3.3, lp=3.3, DvDc=6.5, g=3600, T=50, tfinal=100, threshold_factor=1e-8)

fitparameters, fiterror = compute_fitparameters(np.array([54]))
results = extract_parameters(fitparameters)

'''
params.update() with whatever parameters you want, then define the range of L values, e.g. 20->50
then compute_fitparameters -> extract_parameters -> perform_PL_fits
'''

# params.update(L=16, matrix_no=6, no_of_QDs=1, cavity=1, g=500, T=50, tfinal=50, threshold_factor=1e-6)
params.update(L=16, matrix_no=6, no_of_QDs=1, cavity=1, dotshape='spherical', l=3.3, lp=3.3, DvDc=6.5, g=3750, T=50, tfinal=100, threshold_factor=1e-8)

Ls=np.arange(16,73+1,1)
# Ls=np.array([16])
fitparameters, fiterrors = compute_fitparameters(Ls)
#extract the fit parameters
results = extract_parameters(fitparameters)
#defining PL fit data range and extrapolating the parameter data to L-> infinity
indices = np.arange(len(Ls))[-3:]
indicesY = np.arange(len(Ls))[-4:-1]
indices_middle=np.arange(len(Ls))[-40:-37]
gam_fits = perform_PL_fits(results['gam'], slice(min(indices), max(indices) + 1), names=['gam1', 'gam2', 'gam3'])
gam_fits_Y = perform_PL_fits(results['gam'], slice(min(indicesY), max(indicesY) + 1), names=['gam1', 'gam2', 'gam3'])
gam_fits_middle = perform_PL_fits(results['gam'], slice(min(indices_middle), max(indices_middle) + 1), names=['gam1', 'gam2', 'gam3'])


gam1_raw = results['gam'][0]
gam2_raw = results['gam'][1]  
# gam3_raw = results['gam'][2]
L_ex = np.linspace(10, 200, 200)
gam1_fit = gam_fits['gam1']['fit']
gam2_fit = gam_fits['gam2']['fit']

# gam3_fit = gam_fits['gam3']['fit']
gam1_inf = gam_fits['gam1']['inf']
gam2_inf = gam_fits['gam2']['inf'] 

gam1_inf_Y = gam_fits_Y['gam1']['inf']
gam2_inf_Y = gam_fits_Y['gam2']['inf'] 
# gam3_inf = gam_fits['gam3']['inf']

gam1_inf_middle = gam_fits_middle['gam1']['inf']
gam2_inf_middle = gam_fits_middle['gam2']['inf'] 
#error calc
gam1_error = 0.5 * np.abs((gam1_inf / gam1_inf_Y) - 1) + 0.5 * np.abs((gam1_inf_Y / gam1_inf) - 1)
gam2_error = 0.5 * np.abs((gam2_inf / gam2_inf_Y) - 1) + 0.5 * np.abs((gam2_inf_Y / gam2_inf) - 1)


# Create figure with 3 subplots
plt.figure(figsize=(15, 5))

# Subplot 1: gam1
plt.subplot(131)
plt.plot(Ls, gam1_raw, 'o', label='Raw Data', color='blue', markersize=8)
plt.plot(L_ex, gam1_fit, '-', label='Power Law Fit', color='red', linewidth=2)
plt.axhline(y=gam1_inf, color='green', linestyle='--', label=fr'$\Gamma_1(\infty)$ = {gam1_inf:.3f}')
plt.xlabel('L')
plt.ylabel(r'$\mathrm{dephasing\ rate,\ }\Gamma_{+,+}\ (\mu\mathrm{eV})$')  # Added r and adjusted spacing
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Subplot 2: gam2
plt.subplot(132)
plt.plot(Ls, gam2_raw, 'o', label='Raw Data', color='blue', markersize=8)
plt.plot(L_ex, gam2_fit, '-', label='Power Law Fit', color='red', linewidth=2)
plt.axhline(y=gam2_inf, color='green', linestyle='--', label=fr'$\Gamma_2(\infty)$ = {gam2_inf:.3f}')
plt.xlabel('L')
plt.ylabel(r'$\mathrm{dephasing\ rate,\ }\Gamma_{-}\ (\mu\mathrm{eV})$')  # Added r and adjusted spacing
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Subplot 3: gam3
plt.subplot(133)
plt.plot(Ls, gam3_raw, 'o', label='Raw Data', color='blue', markersize=8)
plt.plot(L_ex, gam3_fit, '-', label='Power Law Fit', color='red', linewidth=2)
plt.axhline(y=gam3_inf, color='green', linestyle='--', label=fr'$\Gamma_3(\infty)$ = {gam3_inf:.3f}')
plt.xlabel('L')
plt.ylabel(r'$\mathrm{dephasing\ rate,\ }\Gamma_{+,-}\ (\mu\mathrm{eV})$')  # Added r and adjusted spacing
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout to prevent overlap
plt.tight_layout()



#%%
# Vary g in QD-cavity system
params.update(L=40, matrix_no=4, no_of_QDs=1, cavity=1, dotshape='spherical', l=3.3, lp=3.3, DvDc=6.5, g=50, T=50, tfinal=100, threshold_factor=1e-8)
Ls=np.arange(40,43+1,1)
Ls=np.array([32,34,36,38])
# Ls=np.array([52,54,56,58])
Ls=np.array([44,48,52,56])

Ls = np.insert(Ls, 0, 24) #original calc data
g_values=np.arange(50,4000,300)
g_values_FGR=np.arange(50,4000,10)
# g_values=np.arange(50,4950,100)
# g_values_FGR=np.arange(50,4960,10)


Gamma1,Gamma2=FGR_qdcav_spherical_det(g_values_FGR*1e-3/params.hbar, params.detuning, params.j0_FGR, params.l, params.Vs, params.T_ps) 

if params.extrapolation==1:
    g_results, w_results, g_results_raw,  w_results_raw, all_results = compute_g_variation(g_values, Ls)
    gam1_values = [g_results_raw[g1]['gam1'] for g1 in g_values]
    gam2_values = [g_results_raw[g1]['gam2'] for g1 in g_values]
    gam1_inf_values = [g_results[g1]['gam1_inf'] for g1 in g_values]
    gam2_inf_values = [g_results[g1]['gam2_inf'] for g1 in g_values]
    gam1_error_values = [g_results[g1]['gam1_error'] for g1 in g_values if g1 >= 100]
    gam2_error_values = [g_results[g1]['gam2_error'] for g1 in g_values if g1 >= 100]
else: 
    g_results_raw,  w_results_raw, all_results = compute_g_variation(g_values, Ls)
    gam1_values = [g_results[g1]['gam1'] for g1 in g_values]
    gam2_values = [g_results[g1]['gam2'] for g1 in g_values]



# # including virtual transitions
# def qdcav_spherical_inc_virtual_integrand(w,g,detuning,j0_FGR,l,Vs,T):
#     hbar= 0.6582119569# [meV ps]
#     R= np.sqrt((detuning**2 + 4*g**2)) 
#     # print(hbar*1e3*R)
#     Dplus=(1/np.sqrt(2)) * np.sqrt(1 + (detuning/R))
#     Dneg=(1/np.sqrt(2)) * np.sqrt(1 - (detuning/R))
#     Gamma_ph= Dplus**2 * Dneg**2 * w**3 *j0_FGR/2 * np.exp(-(l**2 * w**2) / (2*Vs**2))
#     N=1/(np.exp(w/T)-1)
#     Gamma_minus= N*Gamma_ph
#     Gamma_plus= (N+1)*Gamma_ph
#     integrand_plus= (1/np.pi) * (Gamma_plus*Gamma_minus * (4*R**2  + (Gamma_plus - Gamma_minus)**2 )  ) / ( ((R+w)**2 + Gamma_plus**2 ) * ((R-w)**2 + Gamma_minus**2) )
#     integrand_minus= (1/np.pi) * (Gamma_plus*Gamma_minus * (4*R**2  + (Gamma_plus - Gamma_minus)**2)  ) / ( ((R+w)**2 + Gamma_minus**2 ) * ((R-w)**2 + Gamma_plus**2) )

#     return integrand_plus, integrand_minus

# from scipy import integrate
# def qdcav_spherical_inc_virtual(g,detuning,j0_FGR,l,Vs,T):
#     def re_fun(w,g,detuning,j0_FGR,l,Vs,T):
#         return np.real(qdcav_spherical_inc_virtual_integrand(w,g,detuning,j0_FGR,l,Vs,T)[0])
#     def im_fun(w,g,detuning,j0_FGR,l,Vs,T):
#         return np.imag(qdcav_spherical_inc_virtual_integrand(w,g,detuning,j0_FGR,l,Vs,T)[0])
#     w1 = 0 #limits
#     w2 = np.inf
#     re_int = integrate.quad(re_fun, w1, w2, args=(g,detuning,j0_FGR,l,Vs,T))
#     im_int = integrate.quad(im_fun, w1, w2, args=(g,detuning,j0_FGR,l,Vs,T))
#     return re_int[0] + 1j*im_int[0]

# def qdcav_spherical_inc_virtual_lower(g,detuning,j0_FGR,l,Vs,T):
#     def re_fun(w,g,detuning,j0_FGR,l,Vs,T):
#         return np.real(qdcav_spherical_inc_virtual_integrand(w,g,detuning,j0_FGR,l,Vs,T)[1])
#     def im_fun(w,g,detuning,j0_FGR,l,Vs,T):
#         return np.imag(qdcav_spherical_inc_virtual_integrand(w,g,detuning,j0_FGR,l,Vs,T)[1])
#     w1 = 0 #limits
#     w2 = np.inf
#     re_int = integrate.quad(re_fun, w1, w2, args=(g,detuning,j0_FGR,l,Vs,T))
#     im_int = integrate.quad(im_fun, w1, w2, args=(g,detuning,j0_FGR,l,Vs,T))
#     return re_int[0] + 1j*im_int[0]




# gam_plus_virtual=[]
# gam_neg_virtual=[]
# for g in g_values_FGR:
#     g=g*1e-3/params.hbar
#     Gamma1_single,Gamma2_single=FGR_qdcav_spherical_det(g, params.detuning, params.j0_FGR, params.l, params.Vs, params.T_ps) 
#     print('gam plus real is:', Gamma2_single)
#     # gam_plus_v=(Gamma2_single +  1e3*params.hbar*qdcav_spherical_inc_virtual(g,params.detuning,params.j0_FGR,params.l,params.Vs,params.T))
#     R= np.sqrt((params.detuning**2 + 4*g**2)) 
#     # N=1/(np.exp(R/params.T_ps)-1)
#     gam_plus_v=( 1e3*params.hbar*qdcav_spherical_inc_virtual(g,params.detuning,params.j0_FGR,params.l,params.Vs,params.T_ps))
#     gam_neg_v=( 1e3*params.hbar*qdcav_spherical_inc_virtual_lower(g,params.detuning,params.j0_FGR,params.l,params.Vs,params.T_ps))

#     print('gam_plus_virtual is', gam_plus_v)
#     gam_plus_virtual.append(gam_plus_v)
#     gam_neg_virtual.append(gam_neg_v)



############ reduced equation valid for only large g
def qdcav_spherical_inc_virtual_integrand_reduced(w,g,detuning,j0_FGR,l,Vs,T):
    R= np.sqrt((detuning**2 + 4*g**2)) 
    # print(hbar*1e3*R)
    Dplus=(1/np.sqrt(2)) * np.sqrt(1 + (detuning/R))
    Dneg=(1/np.sqrt(2)) * np.sqrt(1 - (detuning/R))
    Gamma_ph= Dplus**2 * Dneg**2 * w**3 *j0_FGR/2 * np.exp(-(l**2 * w**2) / (2*Vs**2))
    N=1/(np.exp(w/T)-1)
    Gamma_minus= N*Gamma_ph
    Gamma_plus= (N+1)*Gamma_ph

    integrand_plus= (1/np.pi) * Gamma_plus * Gamma_minus
    return integrand_plus

from scipy import integrate
def qdcav_spherical_inc_virtual_reduced(g,detuning,j0_FGR,l,Vs,T):
    def re_fun(w,g,detuning,j0_FGR,l,Vs,T):
        return np.real(qdcav_spherical_inc_virtual_integrand_reduced(w,g,detuning,j0_FGR,l,Vs,T))
    def im_fun(w,g,detuning,j0_FGR,l,Vs,T):
        return np.imag(qdcav_spherical_inc_virtual_integrand_reduced(w,g,detuning,j0_FGR,l,Vs,T))
    w1 = 0 #limits
    w2 = np.inf
    re_int = integrate.quad(re_fun, w1, w2, args=(g,detuning,j0_FGR,l,Vs,T))
    im_int = integrate.quad(im_fun, w1, w2, args=(g,detuning,j0_FGR,l,Vs,T))
    return re_int[0] + 1j*im_int[0]


gam_plus_virtual_reduced=[]
gam_neg_virtual_reduced=[]
for g in g_values_FGR:
    g=g*1e-3/params.hbar
    Gamma1_single,Gamma2_single=FGR_qdcav_spherical_det(g, params.detuning, params.j0_FGR, params.l, params.Vs, params.T_ps) 
    print('gam plus real is:', Gamma2_single)
    R= np.sqrt((params.detuning**2 + 4*g**2)) 
    gam_plus_v=(Gamma2_single +  1e3*params.hbar* (4/R**2) * qdcav_spherical_inc_virtual_reduced(g,params.detuning,params.j0_FGR,params.l,params.Vs,params.T_ps))
    gam_neg_v=(Gamma1_single +  1e3*params.hbar* (4/R**2) * qdcav_spherical_inc_virtual_reduced(g,params.detuning,params.j0_FGR,params.l,params.Vs,params.T_ps))
    print('gam_plus_virtual is', gam_plus_v)
    gam_plus_virtual_reduced.append(gam_plus_v)
    gam_neg_virtual_reduced.append(gam_neg_v)


##########################################




## Saving gammas including virtual transitions using Egors formula ## 

# import csv
# from itertools import zip_longest
# with open("optimisation_plot_data/QDCAV_Virtual_only.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["g_values_FGR", "gam_upper_FGR_reduced_virtual", "gam_lower_FGR_reduced_virtual"])  # Write header
#     for row in zip_longest(g_values_FGR, np.real(gam_plus_virtual_reduced), np.real(gam_neg_virtual_reduced), fillvalue=np.nan):
#         writer.writerow(row)
 



# plt.figure(figsize=(10, 6))
# # plt.plot(g_values,Gamma1, 'r--',  label=r'FGR $\Gamma_{-}$')
# plt.plot(g_values,Gamma2, 'b--',  label=r'FGR $\Gamma_{+}$')
# plt.plot(g_values,gam_plus_virtual, 'g--',  label=r'$\Gamma_{+}$ inc. virtual')

# plt.legend(loc='best')
# plt.ylabel(r'Dephasing rates, $\Gamma$ ($\mu$ eV)')
# plt.xlabel(r'Coupling strength, $g$ ($\mu$ eV)')

plt.figure(figsize=(10, 6))
#here gam1 should be upper level, gam2 middle, gam3 lower
plt.plot(g_values, gam1_inf_values, color='blue', linestyle='-', label=r'$\Gamma_{+}(\infty)$')
plt.plot(g_values, gam1_values, 'bx', label=r'$\Gamma_{+}(\infty)$')
# plt.plot(g_values, gam2_inf_values, 'r-', label=r'$\Gamma_{-}(\infty)$')
# plt.plot(g_values_FGR,Gamma1, 'r--',  label=r'FGR $\Gamma_{-}$')
plt.plot(g_values_FGR,Gamma2, 'b--',  label=r'FGR $\Gamma_{+}$')
# plt.plot(g_values_FGR,gam_plus_virtual, 'g--',  label=r'$\Gamma_{+}$ inc. virtual Eq 2')
plt.plot(g_values_FGR,gam_plus_virtual_reduced, 'r--',  label=r'$\Gamma_{+}$ inc. virtual Eq.1')
plt.legend(loc='best')
plt.ylabel(r'Dephasing rates, $\Gamma$ ($\mu$eV)')
plt.xlabel(r'Coupling strength, $g$ ($\mu$eV)')
# plt.yscale('log')

plt.figure(figsize=(4.5,3))
mask = g_values >= 100
filtered_g_values = g_values[mask]
plt.plot(filtered_g_values, gam1_error_values, 'b-', label=r'$\Gamma_{+}(\infty)$ error')
plt.plot(filtered_g_values, gam2_error_values, 'r-', label=r'$\Gamma_{-}(\infty)$ error')
plt.yscale('log')
###########################################################
plt.figure(figsize=(10, 6))
#here gam1 should be upper level, gam2 middle, gam3 lower
# plt.plot(g_values, gam1_inf_values, color='blue', linestyle='-', label=r'$\Gamma_{+}(\infty)$')
plt.plot(g_values, gam2_inf_values, 'b-', label=r'$\Gamma_{-}(\infty)$')
plt.plot(g_values, gam2_values, 'bx', label=r'$\Gamma_{+}(\infty)$')

plt.plot(g_values_FGR,Gamma1, 'b--',  label=r'FGR $\Gamma_{-}$')
# plt.plot(g_values_FGR,Gamma2, 'b--',  label=r'FGR $\Gamma_{+}$')
# plt.plot(g_values_FGR,gam_neg_virtual, 'g--',  label=r'$\Gamma_{-}$ inc. virtual Eq 2')
plt.plot(g_values_FGR,gam_neg_virtual_reduced, 'r--',  label=r'$\Gamma_{-}$ inc. virtual Eq.1')
plt.legend(loc='best')
plt.ylabel(r'Dephasing rates, $\Gamma$ ($\mu$eV)')
plt.xlabel(r'Coupling strength, $g$ ($\mu$eV)')
# plt.yscale('log')

import csv
from itertools import zip_longest
with open("optimisation_plot_data/QDCAV_include_virtual_largeg.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["g_values","g_values_FGR", "gam_upper_inf","gam_lower_inf", "gam_upper","gam_lower",  "gam_upper_FGR", "gam_lower_FGR", "gam_upper_FGR_reduced_virtual", "gam_lower_FGR_reduced_virtual", "gam_upper_errors", "gam_lower_errors"])  # Write header
    for row in zip_longest(g_values, g_values_FGR, np.real(gam1_inf_values),np.real(gam2_inf_values), np.real(gam1_values),np.real(gam2_values), np.real(Gamma2), np.real(Gamma1), np.real(gam_plus_virtual_reduced),np.real(gam_neg_virtual_reduced), gam1_error_values, gam2_error_values, fillvalue=np.nan):
        writer.writerow(row)


# import csv
# from itertools import zip_longest
# with open("optimisation_plot_data/QDCAV_Virtual_transitions.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["g_values","g_values_FGR", "gam_upper","gam_lower", "gam_upper_FGR", "gam_lower_FGR", "gam_upper_FGR_virtual", "gam_lower_FGR_virtual","gam_upper_FGR_reduced_virtual", "gam_lower_FGR_reduced_virtual"])  # Write header
#     for row in zip_longest(g_values, g_values_FGR, np.real(gam1_inf_values),np.real(gam2_inf_values), np.real(Gamma2), np.real(Gamma1), np.real(gam_plus_virtual), np.real(gam_neg_virtual), np.real(gam_plus_virtual_reduced),np.real(gam_neg_virtual_reduced), fillvalue=np.nan):
#         writer.writerow(row)
 




#%%


params.update(L=16, matrix_no=5, no_of_QDs=2, cavity=1, extrapolation=1, dotshape='smartie', l=7.5, lp=2.5, DvDc=6.5, g1=2750, g2=2750, g=0, T=20, r0=5.0, tfinal=50, threshold_factor=1e-6)
r0_values=np.arange(0.1,10.5,1)
Ls=np.arange(45,47+1,1)

if params.extrapolation==1:
    r0_results, r0_results_raw, all_results = compute_r0_variation(r0_values, Ls)
    gam1_inf_values = [r0_results[r0]['gam1_inf'] for r0 in r0_values]
    gam2_inf_values = [r0_results[r0]['gam2_inf'] for r0 in r0_values]
    gam3_inf_values = [r0_results[r0]['gam3_inf'] for r0 in r0_values]
    gam1_values = [r0_results_raw[r0]['gam1'] for r0 in r0_values]
    gam2_values = [r0_results_raw[r0]['gam2'] for r0 in r0_values]
    gam3_values = [r0_results_raw[r0]['gam3'] for r0 in r0_values]
else:
    r0_results_raw, all_results = compute_r0_variation(r0_values, Ls)
    gam1_values = [r0_results_raw[r0]['gam1'] for r0 in r0_values]
    gam2_values = [r0_results_raw[r0]['gam2'] for r0 in r0_values]
    gam3_values = [r0_results_raw[r0]['gam3'] for r0 in r0_values]


plt.figure(figsize=(10, 6))
#here gam1 should be upper level, gam2 middle, gam3 lower
plt.plot(r0_values, gam1_inf_values, color='blue', linestyle='-', label=r'$\Gamma_{+,+}(\infty)$')
plt.plot(r0_values, gam2_inf_values, 'r-', label=r'$\Gamma_{-}(\infty)$')
plt.plot(r0_values, gam3_inf_values, 'g-', label=r'$\Gamma_{+,-}(\infty)$')

plt.plot(r0_values, gam1_values, color='blue', linestyle='-', label=r'$\Gamma_{+,+}$')
plt.plot(r0_values, gam2_values, 'r-', label=r'$\Gamma_{-}$')
plt.plot(r0_values, gam3_values, 'g-', label=r'$\Gamma_{+,-}$')
# plt.title('Förster-like coupled QDs')
plt.ylabel(r'Dephasing rates, $\Gamma$ ($\mu$ eV)')
plt.xlabel(r'QD separation, $d$ (nm)')
# r0_values=np.arange(0.01,55.,0.1)
FGR_gam1,FGR_gam2,FGR_gam3=FGR_smartie(params.j0_FGR,params.l,params.lp,params.Vs,params.T_ps,2750*1e-3 /params.hbar,0,0,0,r0_values)
#gam1 middle level, gam2 upper level, gam3 lower level
plt.plot(r0_values, FGR_gam2, 'b--', label=r' $FGR \Gamma_{+,+}$')     
plt.plot(r0_values, FGR_gam1, color='red', linestyle='--', label=r' $FGR \Gamma_{-}$')
plt.plot(r0_values, FGR_gam3, color='green', linestyle='--', label=r'FGR  $\Gamma_{+,-}$')
plt.legend(loc='best')



# r0_values=np.arange(1.0,30.0,0.01)
# FGR_gam1s=[]
# FGR_gam2s=[]
# FGR_gam3s=[]
# r_start=15.0
# for r0 in r0_values:
#     g=1000*1e-3/params.hbar
#     g= g* (r_start/r0)**3
#     FGR_gam1,FGR_gam2,FGR_gam3=FGR_smartie(params.j0_FGR,params.l,params.lp,params.Vs,params.T_ps,params.g1_comp, g , params.w_qd1,params.w_c,r0)
#     FGR_gam1s.append(FGR_gam1)
#     FGR_gam2s.append(FGR_gam2)
#     FGR_gam3s.append(FGR_gam3)
# plt.figure(figsize=(10, 6))
# plt.plot(r0_values, FGR_gam2s, 'b-', label=r' $\Gamma_{+}$')     
# plt.plot(r0_values, FGR_gam1s, color='red', linestyle='-', label=r' $\Gamma_{-}$')
# plt.xlabel(r'distance between qubits, $d$ (nm)')
# plt.ylabel(r'dephasing rates, $\Gamma$ ($\mu$eV)')
# plt.legend()
    
# r0_values=np.arange(2.5,30.0,0.1)
# FGR_gam1,FGR_gam2,FGR_gam3=FGR_smartie(params.j0_FGR,params.l,params.lp,params.Vs,params.T_ps,params.g1_comp,1000*1e-3/params.hbar,params.w_qd1,params.w_c,r0_values)
# plt.figure(figsize=(10, 6))
# plt.plot(r0_values, FGR_gam2, 'b-', label=r'$\Gamma_{+}$')
# plt.plot(r0_values, FGR_gam1, color='red', linestyle='-', label=r'$\Gamma_{-}$')
# # plt.plot(r0_values, FGR_gam3, 'g--', label=r'FGR $\Gamma_{+,-}$')
# plt.legend(loc='best')
# plt.xlabel(r'distance between qubits, $d$ (nm)')
# plt.ylabel(r'dephasing rates, $\Gamma$ ($\mu$eV)')
# plt.figure(figsize=(10, 6))
# plt.plot(r0_values, FGR_gam1_nocav, 'rx', label=r'FGR $\Gamma_{+}$')
# plt.plot(r0_values, FGR_gam2_nocav, 'bx', label=r'FGR $\Gamma_{-}$')
# plt.legend(loc='best')





plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.title(r'extrapolated dephasing rates vs $d$')
plt.show()

#%%
#varying g1g2 

params.update(L=16, matrix_no=5, extrapolation=1, no_of_QDs=2, cavity=1, dotshape='smartie', l=7.5, lp=2.5, DvDc=6.5, g1=300, g2=300, g=0, T=20, r0=2.5, tfinal=50, threshold_factor=1e-6)
g_values=np.arange(100,4000,100)
# g_values = np.insert(g_values, np.searchsorted(g_values, 2750), 2750)
Ls=np.arange(45,47+1,1)

if params.extrapolation==1:
    g_results_raw, g_results= compute_g1g2_variation(g_values,Ls)
    gam1_values = [g_results_raw[g1]['gam1'] for g1 in g_values]
    gam2_values = [g_results_raw[g1]['gam2'] for g1 in g_values]
    gam3_values = [g_results_raw[g1]['gam3'] for g1 in g_values]
    gam1_inf_values = [g_results[g1]['gam1_inf'] for g1 in g_values]
    gam2_inf_values = [g_results[g1]['gam2_inf'] for g1 in g_values]
    gam3_inf_values = [g_results[g1]['gam3_inf'] for g1 in g_values]
else: 
    g_results_raw = compute_g1g2_variation(g_values,Ls)
    gam1_values = [g_results[g1]['gam1'] for g1 in g_values]
    gam2_values = [g_results[g1]['gam2'] for g1 in g_values]
    gam3_values = [g_results[g1]['gam3'] for g1 in g_values]


plt.figure(figsize=(10, 6))
#here gam1 should be upper level, gam2 middle, gam3 lower
plt.plot(g_values, gam1_inf_values, color='blue', linestyle='-', label=r'$\Gamma_{+,+}(\infty)$')
plt.plot(g_values, gam2_inf_values, 'r-', label=r'$\Gamma_{-}(\infty)$')
plt.plot(g_values, gam3_inf_values, 'g-', label=r'$\Gamma_{+,-}(\infty)$')

plt.plot(g_values, gam1_values, color='blue', linestyle='-', label=r'$\Gamma_{+,+}$')
plt.plot(g_values, gam2_values, 'r-', label=r'$\Gamma_{-}$')
plt.plot(g_values, gam3_values, 'g-', label=r'$\Gamma_{+,-}$')
# plt.title('Förster-like coupled QDs')
plt.title(f'distance between QDs is {round(params.r0,2)}nm')
plt.ylabel(r'Dephasing rates, $\Gamma$ ($\mu$eV)')
plt.xlabel(r'Coupling strength, $g_1=g_2$ ($\mu$eV)')
# r0_values=np.arange(0.01,55.,0.1)



FGR_gam1 = []
FGR_gam2 = []
FGR_gam3 = []
g_values = np.arange(100,4000,10)
g_inputs = g_values * 1e-3 / params.hbar
for g1 in g_inputs:
    gam1, gam2, gam3 = FGR_smartie(
        params.j0_FGR, params.l, params.lp, params.Vs, params.T_ps,
        g1, 0, 0, 0, params.r0)
    FGR_gam1.append(gam1)
    FGR_gam2.append(gam2)
    FGR_gam3.append(gam3)
FGR_gam1 = np.array(FGR_gam1)
FGR_gam2 = np.array(FGR_gam2)
FGR_gam3 = np.array(FGR_gam3)
# FGR_gam1,FGR_gam2,FGR_gam3=FGR_smartie(params.j0_FGR,params.l,params.lp,params.Vs,params.T_ps,g_values*1e-3 /params.hbar,0,0,0,params.r0)
#gam1 middle level, gam2 upper level, gam3 lower level
plt.plot(g_values, FGR_gam2, 'b--', label=r'FGR $\Gamma_{+,+}$')     
plt.plot(g_values, FGR_gam1, color='red', linestyle='--', label=r'FGR $\Gamma_{-}$')
plt.plot(g_values, FGR_gam3, color='green', linestyle='--', label=r'FGR $\Gamma_{+,-}$')
plt.legend(loc='best')
plt.yscale('log')




