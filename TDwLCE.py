'''
params = Parameters() loads the parameters set in params.py with default values
update params with, for example: params.update(L=40,r0=5, gc=200)
Then use the following to compute system dynamics: P, times = Compute_dynamics()
'''

import numpy as np
import lmfit
import opt_einsum as oe
import itertools
from scipy.linalg import expm
from params import Parameters
from Functions import LFpop, LFpol, forster, LFpol_qdqdcav, DiagM_qdcav, S_inin, S_inim, PolaronShift, PolaronShift_inim, phi_inin, phi_inim, K11_smartie, K12_smartie, Kbb2, analytics_bareg, analytics_modified, QDQD_analytics_smartie, FGR_smartie, FGR_spherical, FGR_qdcav_spherical, FGR_qdcav_spherical_det
import matplotlib.pyplot as plt
import time
from tqdm import tqdm 
import os
path=os.getcwd()
params = Parameters()
if params.correlator=='NQD':
    from Forster_FGR import Gamma1_FGR_det, Gamma2_FGR_det, N21ampli, GammaPh, Gamma1_FGR_det_nrm, Gamma2_FGR_det_nrm, Pan_NQD, Panalyt2

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

def optimize_svd(V,threshold_factor,step):  
    A, S1, Vh = np.linalg.svd(V, full_matrices=False)
    threshold = S1[0] * threshold_factor
    thresh = S1 > threshold
    S1 = S1[thresh]
    trunc=len(S1)
    A = A[:, thresh] 
    V = Vh[thresh, :]
    S = np.diag(S1)
    if step % 1 == 0:
        V = S @ V 
    else:
        A = A @ S
    if abs(V[0][0]) > 1e6:
        value = 2.5
    else:
        value = 1.3
    return V/value, A*value, trunc

#%%
# M1, Qlist, exp_k0_factor = params.cumulant_generator()
def Compute_dynamics():
    start_time = time.time()
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

    # if params.no_of_QDs==1:
    #     dt=params.factortau*params.tauib/(params.L+1)
    # if params.no_of_QDs==2:
    #     dt= ( (params.r0/params.Vs) + params.factortau*params.tauib)/(params.L+1)
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
         
    start_time = time.time() # time at which function is called
    for step in tqdm(range(step_no),position=0, leave=True):     
        print(' \n STEP NUMBER IS:', step+1)
        values = []
        if step!=0:
            indices = np.roll(indices, -1) #rolling the indices to simulate the remapping i4i3 i2i1 -> i3i2 i1i4
        i2_matrix1 = []
        V_indices2 = []
        V = []
        i1_matrix_pos = None
        i1_pos = None
        matrix_numbers = np.arange(params.matrix_no)
        
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
                Vs_shape_last = V_matrices[i1_matrix_pos].shape[0]         
                for j in range(Vs_shape_last):
                    V_matrices[i1_matrix_pos][j, :] *= QV                        
                V_i_splitting=generate_V_arrays(V_matrices[i1_matrix_pos],true_false_arrays)
            
                #this is where we apply Q_p containing K^(L) cumulant elements, to V, however p can be 0,1,2, so we will split our current V into 3.
                V_ips = [generate_ip_splitting(V_i_splitting[i], params.d) for i in range(params.d)]
                Qlist_last = params.Qlist[-1]
                for i in range(params.d):
                    for p in range(params.d):
                        V_ips[i][p] *= Qlist_last[p,i]
                     
                V_is = restructure_arrays(V_ips,params.d,indices_no[matrix] - i1_pos - 1)
                V_matrices[i1_matrix_pos] = np.vstack([V_is[i] for i in range(len(V_is))])     
            else:
            ############
                Vs = generate_copies(V_matrices[matrix], params.d)
                Q_Vs = generate_arrays(n_perms, params.d)
                for j in range(len(V_indices)):
                    index_val = list(permutations_matrix[:, j])  #values of the index across permutations
                    V_index = int(V_indices[j])
                    V_j = V_index - 2 #if V_index != '1' else None
                    index_np = np.array(index_val)
                    for i in range(params.d):
                        Q_Vs[i] *= np.array(params.Qlist)[V_j][index_val,i]
                Vs_shape_last = Vs[0][0].shape[0]  
                for i in range(params.d):
                    # Multiply all rows at once for each matrix
                    Vs[i] = Vs[i] * Q_Vs[i]    
                V.append(np.vstack([Vs[i] for i in range(len(Vs))]))
                del Vs, Q_Vs
        print(f"\n  Q on all matrices applied at {np.round((time.time() - start_time), 2)} seconds")  
                
            
        for j, i in enumerate(matrix_numbers[np.arange(len(V_matrices)) != i1_matrix_pos]):
            V_matrices[i] = V[j]
    
        del V
        for matrix in range(0, params.matrix_no):
            permutations_matrix = permutations[f'{indices_no[matrix]}']
            V_indices = V_indices2[matrix]
            # i2_matrix = (((step + 1) % L) // indices_no[matrix])
            
            if '2' in V_indices:
                i2_pos = np.where(V_indices == '2')[0][0]
                desired_perm = np.ones([1, indices_no[matrix]])
                desired_perm[:, i2_pos] = 0
                col_pos = np.where(np.all(permutations_matrix == desired_perm, axis=1))[0][0]
                i2_matrix1.append(col_pos)
                values.append(V_matrices[matrix][:, col_pos])
            else:
                i2_matrix1.append(-1)  
                values.append(V_matrices[matrix][:, -1]) 
                             
            V_matrices[matrix], A_matrices[matrix], trunc = optimize_svd(V_matrices[matrix], params.threshold_factor, step)
            print(f"\n SVD calc for matrix{matrix} {np.round((time.time() - start_time), 2)} seconds")

        if step == 0:
            all_ind = "abcdefghijklmnopqrstuvwxyz"
            einsum_ind = all_ind[:params.matrix_no ]      
            einsum_str = "".join("z" + einsum_ind[i] + "," for i in range(params.matrix_no-1)) + "z" + einsum_ind[-1] + "->" + f"{einsum_ind}"
            R =oe.contract(einsum_str, *A_matrices)        
            inputs = [R] + [V_matrices[i][:, i2_matrix1[i]] for i in range(params.matrix_no)]
            einsum_str_result = f"{einsum_ind}, " + ", ".join(  einsum_ind[i] for i in range(params.matrix_no)) + " ->"
            result = params.exp_k0_factor * oe.contract(einsum_str_result, *inputs)
            del inputs
        else:   
            output_indices = all_ind[params.matrix_no : 2 * params.matrix_no]  # Next N indices for output
            V_terms = [f"{einsum_ind[i]}{output_indices[i]}" for i in range(params.matrix_no)]  # Each V has 'xi'
            einsum_str = f"{einsum_ind}, " + ", ".join(V_terms) + f" -> {output_indices}"
            for j in range(params.d):
                if step%2 == 1: # odd step
                    inputs = [R] + [A_matrices[i][j*int(A_matrices[i].shape[0]/params.d):(j+1)*int(A_matrices[i].shape[0]/params.d), :] for i in range(params.matrix_no)]
                    if j==0:
                        R_temp = oe.contract(einsum_str, *inputs)
                    else:
                        R_temp +=  oe.contract(einsum_str, *inputs)
                else:
                    inputs = [R_temp] + [A_matrices[i][j*int(A_matrices[i].shape[0]/params.d):(j+1)*int(A_matrices[i].shape[0]/params.d), :] for i in range(params.matrix_no)]
                    if j==0:
                        R = oe.contract(einsum_str, *inputs)
                    else:
                        R +=  oe.contract(einsum_str, *inputs)
            del inputs
            print(f"\n   R calculation {np.round((time.time() - start_time), 2)} seconds")  
            if step%2 == 1:
                del R
                print('R shape: ', R_temp.shape)
                inputs = [R_temp] + [V_matrices[i][:, i2_matrix1[i]] for i in range(params.matrix_no)]
                result = params.exp_k0_factor*oe.contract(einsum_str_result, *inputs)
                # n,m,k,o=R_temp.shape
                # R_remapped= R_temp.reshape(o*m*k,n)
                # RV=  R_remapped @ V_matrices[0]        
                # V_matrices[0],R_remapped, trunc= optimize_svd(RV,threshold_factor,step)
                # R_temp=R_remapped.reshape(trunc,m,k,o)
                # print('R shape after SVD: ', R_temp.shape)
            else:
                del R_temp
                print('R shape: ', R.shape)
                inputs = [R] + [V_matrices[i][:, i2_matrix1[i]] for i in range(params.matrix_no)]
                result = params.exp_k0_factor*oe.contract(einsum_str_result, *inputs)
                # n,m,k,o=R.shape
                # R_remapped= R.reshape(o*m*k,n)
                # RV=  R_remapped @ V_matrices[0]        
                # V_matrices[0],R_remapped, trunc= optimize_svd(RV,threshold_factor,step)
                # R=R_remapped.reshape(trunc,m,k,o)
                # print('R shape after SVD: ', R.shape)
            del inputs
            print(f"\n   result einsum {np.round((time.time() - start_time), 2)} seconds")  
        P.append(result)
    return np.asarray(P), np.asarray(times)

# tfinal=50
# threshold_factor=1e-6
# matrix_no = 4
# P, times = Compute_dynamics()
# plt.figure(10) 
# plt.plot(times, np.abs(P), label=fr'L={params.L}, g={params.gd}, thresh={params.threshold_factor}')   
# plt.legend(loc='best')
# plt.yscale('log')    

#%%
def fitprocedure( P, times): 
    
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
    if params.no_of_QDs==2 and params.cavity==1:
        matrix = np.array([[params.w_qd1, params.gd_comp, params.g1_comp],
                   [params.gd_comp, params.w_qd2, params.g2_comp],
                   [params.g1_comp, params.g2_comp, params.w_c]])
        eigenvalues = np.linalg.eigvals(matrix)
        eigenvalues=  np.sort(eigenvalues)[::-1]
     
    if params.no_of_QDs==1 and params.cavity==1:   
        matrix = np.array([[params.w_qd1, params.gd_comp],
                           [params.gd_comp, params.w_c]])
        eigenvalues = np.linalg.eigvals(matrix)
        eigenvalues=  np.sort(eigenvalues)[::-1]
        
    if params.no_of_QDs==2 and params.cavity==1 and params.g1_comp!=0:
        A1s.append(0.5 - 0*1j)
        A2s.append(0.5 - 0*1j)  
        A3s.append(0.5- 0*1j)  
        w2s.append(eigenvalues[0]+ 0*1j)            
        if params.gd < params.g1:
            w1s.append(eigenvalues[1] + 0*1j)   
            w3s.append(eigenvalues[2] + 0*1j) 
        else:
            w1s.append(eigenvalues[2] + 0*1j)   
            w3s.append(eigenvalues[1] + 0*1j)
        
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
        LF = LFpop(params.g_comp,  np.real(params.w_qd1), np.real(params.w_qd2), -params.gamma1_comp, -params.gamma2_comp) 
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
    return tmlongt, Pnlongt, fit_sametimes, FitParameters


#%%
params= Parameters()
label=params.generate_label()
P, times = Compute_dynamics()
tmlongt, Pnlongt, fit_sametimes, fitparams = fitprocedure(P,times)
fiterror_ours=(P-fit_sametimes)/P

fig1 = plt.figure( figsize=(4.5,3),dpi=150)
bb = fig1.add_subplot(1, 1, 1)     
bb.plot(np.abs(times),np.abs(P),'r-',markersize='1',linewidth='0.4', label=fr'L={params.L}, g={params.gd}, thresh={params.threshold_factor}')   
# bb.plot(np.abs(times),np.abs(fit_sametimes),'b--',label=f'fit, L={L}') 
bb.plot(times,np.abs(fit_sametimes),'b--',label='fit')   
plt.yscale('log')
print(fitparams)
  

data_to_save = np.column_stack((times, P))
np.save(path+"/data/"+label+".npy",data_to_save)

Ls=np.arange(20,37,1)
fitparameters=[]
for L in Ls:
    params.update(L=L)
    label=params.generate_label()
    P, times = Compute_dynamics()
    tmlongt, Pnlongt, fit_sametimes, fitparams = fitprocedure(P,times)
    fitparameters.append(fitparams)
    if L % 4:
        fig1 = plt.figure( figsize=(4.5,3),dpi=150)
        bb = fig1.add_subplot(1, 1, 1)  
        bb.plot(np.abs(times),np.abs(P),'b-',markersize='1',linewidth='0.8', label=f'L={params.L}')
        bb.plot(np.abs(times),np.abs(fit_sametimes),'g--',label='fit') 

w1s, w2s, w3s, gam1s, gam2s, gam3s, A1res, A2res, A3res, A1ims, A2ims, A3ims = [[] for _ in range(12)]
for i in range(len(Ls)): 
    w1s.append(fitparameters[i]['w1_re']*params.hbar*1e3)
    w2s.append(fitparameters[i]['w2_re']*params.hbar*1e3)
    A1res.append(fitparameters[i]['A1_re'])
    A2res.append(fitparameters[i]['A2_re'])
    if params.g1!= 0 and params.no_of_QDs==2 and params.cavity==1:
        w3s.append(fitparameters[i]['w3_re']*params.hbar*1e3)
        A3res.append(fitparameters[i]['A3_re'])

    gam1s.append(np.abs(fitparameters[i]['w1_im']*params.hbar*1e3))
    gam2s.append(np.abs(fitparameters[i]['w2_im']*params.hbar*1e3))
    A1ims.append(fitparameters[i]['A1_im'])
    A2ims.append(fitparameters[i]['A2_im'])
    if params.g1!= 0 and params.no_of_QDs==2 and params.cavity==1:
        gam3s.append(np.abs(fitparameters[i]['w3_im']*params.hbar*1e3))
        A3ims.append(fitparameters[i]['A3_im'])


fig1 = plt.figure(figsize=(8, 9), dpi=150)  # Increase figure height to accommodate additional plots

# Upper left subplot for w1s
ax2 = fig1.add_subplot(321)  # Changed from 221 to 321
ax2.plot(Ls, w2s, 'r-')
ax2.set_ylabel(r'$\Omega_{+,+}$ ($\mu$eV)', fontsize=12)
# ax2.set_xlabel('$L$ (neighbors)', fontsize=12)
ax2.set_yticklabels(['{:.2f}'.format(x) for x in ax2.get_yticks()])

# Middle left subplot for w2s
ax1 = fig1.add_subplot(323)  # New subplot
ax1.plot(Ls, w1s, 'b-')
ax1.set_ylabel(r'$\Omega_-$ ($\mu$eV)', fontsize=12)
# ax1.set_xlabel('$L$ (neighbors)', fontsize=12)

if params.g1!= 0 and params.no_of_QDs==2 and params.cavity==1:
    # Bottom left subplot for w3s
    ax3 = fig1.add_subplot(325)  # New subplot
    ax3.plot(Ls, w3s, 'g-')  # Assuming w3s is defined
    ax3.set_ylabel(r'$\Omega_{+,-}$ ($\mu$eV)', fontsize=12)
    ax3.set_xlabel('$L$ (neighbors)', fontsize=12)

# Upper right subplot for gam1s
ax5 = fig1.add_subplot(322)  # Changed from 222 to 322
ax5.plot(Ls, gam2s, 'r-')
ax5.set_ylabel(r'$\Gamma_{+,+}$ ($\mu$eV)', fontsize=12)
# ax5.set_xlabel('$L$ (neighbours)', fontsize=12)

# Middle right subplot for gam2s
ax4 = fig1.add_subplot(324)  # New subplot
ax4.plot(Ls, gam1s, 'b-')
ax4.set_ylabel(r'$\Gamma_{-}$ ($\mu$eV)', fontsize=12)
# ax4.set_xlabel('$L$ (neighbours)', fontsize=12)

if params.g1!= 0 and params.no_of_QDs==2 and params.cavity==1:
    # Bottom right subplot for gam3s
    ax6 = fig1.add_subplot(326)  # New subplot
    ax6.plot(Ls, gam3s, 'g-')  # Assuming gam3s is defined
    ax6.set_ylabel(r'$\Gamma_{+,-}$ ($\mu$eV)', fontsize=12)
    ax6.set_xlabel('$L$ (neighbours)', fontsize=12)
    
plt.tight_layout()






































































