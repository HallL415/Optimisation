import numpy as np
from scipy.linalg import expm
from Functions import LFpop, LFpol, forster, LFpol_qdqdcav, DiagM_qdcav, S_inin, S_inim, PolaronShift, PolaronShift_inim, phi_inin, phi_inim, K11_smartie, K12_smartie, Kbb2, analytics_bareg, analytics_modified, QDQD_analytics_smartie, FGR_smartie, FGR_spherical, FGR_qdcav_spherical, FGR_qdcav_spherical_det

class Parameters:
    def __init__(self, **kwargs):
        # System configuration
        self.correlator = kwargs.get('correlator', 'LP')
        self.L = kwargs.get('L', 20) 
        self.threshold_factor = kwargs.get('threshold_factor', 1e-8)
        self.matrix_no = kwargs.get('matrix_no', 4)
        self.tfinal = kwargs.get('tfinal', 200)
        
        self.cavity = kwargs.get('cavity', 1)
        self.no_of_QDs = kwargs.get('no_of_QDs', 1)
        self.d = kwargs.get('d', self.cavity + self.no_of_QDs)
        
       
        self.sharebath = kwargs.get('sharebath', 1)
        self.tf = kwargs.get('tf', 200)
        self.dotshape = kwargs.get('dotshape', 'spherical')
        self.ec = kwargs.get('ec', '1')
        self.mc = kwargs.get('mc', '1')
        if self.correlator=='LP' and self.no_of_QDs==1:
            if self.ec=='1':
                self.exc_channel=0
            if self.ec=='C':
                self.exc_channel= 1
            if self.mc=='1':
                self.measure_channel=0
            if self.mc=='C':
               self.measure_channel=1                
        if self.correlator=='LP' and self.no_of_QDs==2:
            if self.ec=='1':
                self.exc_channel=0
            if  self.ec=='2':
                self.exc_channel=1
            if  self.ec=='C':
                self.exc_channel=2
            if  self.mc=='1':
                self.measure_channel=0
            if  self.mc=='2':
                self.measure_channel=1
            if  self.mc=='C':
                self.measure_channel=2    
        if self.correlator=='NQD':
            if self.ec=='2':
                self.exc_channel=4
            else:
                self.exc_channel=1
            if self.mc=='2':
                self.measure_channel=4
            else:
                self.measure_channel=1

        # Physical parameters
        self.T = kwargs.get('T', 50)  # temperature in K
        self.g = kwargs.get('g', 0)   # exciton-exciton coupling strength in micro eV
        self.gc = kwargs.get('gc', 600)  # exciton-cavity coupling strength
        self.detuning = kwargs.get('detuning', 0)  # detuning in micro eV
        self.g1 = kwargs.get('g1', self.gc)  # exciton 1-cavity coupling strength
        self.g2 = kwargs.get('g2', self.gc)  # exciton 2-cavity coupling strength
        self.r0 = kwargs.get('r0', 10)  # distance between QDs in nm
        self.DvDc = kwargs.get('DvDc', 6.5)  # Dc-Dv in eV
        self.l = kwargs.get('l', 3.3)  # exciton confinement lengths
        self.lp = kwargs.get('lp', 3.3)
        
        # Decay rates
        self.gamma1 = kwargs.get('gamma1', 0)  # phenomenological decay rate of dot 1
        self.gamma2 = kwargs.get('gamma2', 0)  # phenomenological decay rate of dot 2
        self.gammac = kwargs.get('gammac', 0)  # phenomenological decay rate of cavity
        
        # Convergence parameters
        self.factortau = kwargs.get('factortau', 1.2)
        
        # Foerster coupling parameters
        self.dcv = kwargs.get('dcv', 0.6)
        self.incr = kwargs.get('incr', 0.05)
        self.eps = kwargs.get('eps', 12.53)
        self.fact = kwargs.get('fact', 5)
        
        # Initialize derived parameters
        self._init_constants()
        self._init_derived_params()
        
        # Compute cumulants during initialization
        self.cumulant_generator()  # Call this to set M1, Qlist, exp_k0_factor
    def _init_constants(self):
        # Fundamental constants
        self.hbar = 0.6582119569  # [meV ps]
        self.kb = 8.617333262e-2  # [meV ps]
        self.vc = 299792458  # [m s^-1]
        
        # QD material parameters
        self.Vs = 4.6e3 * 1e-12  * 1e9  # [km/s]
        self.dens = 5.65  # [g/cm^3]
    
    def _init_derived_params(self):
        # Apply conditional logic based on system configuration
        if self.cavity == 1 and self.no_of_QDs == 2:
            self.gd = self.g
        elif self.cavity == 1 and self.no_of_QDs == 1:
            self.gd = self.gc
            # self.g1 = 0
            # self.g2 = 0

            
        if self.correlator == 'NQD':
            self.d = 5
            self.gd = 0
        elif self.correlator == 'LP':
            self.d = self.cavity + self.no_of_QDs
            
        # QD size in nm
        if self.dotshape == 'spherical':
            self.lp = self.l
            
        self.lbar = (np.sqrt((self.l**2 + self.lp**2)/2))
        
        # Process Foerster coupling if needed
        if self.correlator == 'NQD':
            self.g = self._calculate_forster_coupling(self.r0)
            r0p = self.r0 + self.incr
            self.gp = self._calculate_forster_coupling(r0p)
            
        # Calculate derived parameters
        self.w0 = (np.sqrt(2) * self.Vs / self.l)  # [ps]
        self.T_ps = self.T * self.kb / self.hbar  # [ps]
        
        # Calculate j0 values
        self._calculate_j0_values()
        
        # Convert parameters to calculation units
        self._convert_units()
        
        # Calculate tauib and dt
        self._calculate_time_params()
        
        # Calculate polarization shifts and SHR
        self._calculate_polaron_params()
        
        # Setup quantum dot frequencies
        self._setup_frequencies()
        
        # Setup polariton states
        self._setup_polariton_states()
        
        # Setup measurement channels
        self._setup_measurement_channels()
    
    def _calculate_forster_coupling(self, r):
        """Calculate Foerster coupling for given distance"""
        return forster(self.l, self.eps, self.dcv, r, 
                      -2*self.fact, 10*self.fact, -4*self.fact, 8*self.fact)
    
    def _calculate_j0_values(self):
        """Calculate j0 values required for calculations"""
        # Base j0 calculation
        self.j0 = ((self.DvDc**2 * (1e3)**2 * (self.vc)**2 * (1e9)**2 * (1e-12)**2) / 
                 ((2*np.pi)**2 * self.dens * (1e-3) * (self.vc)**2 * (6.24e21) * (1e-7)**3 * self.Vs**5))
        self.j0 = self.j0 / self.hbar  # j0 in ps^2
        
        # j0_1 for two QDs
        if self.no_of_QDs == 2:
            self.j0_1 = ((self.DvDc**2 * (1e3)**2 * (self.vc)**2 * (1e9)**2 * (1e-12)**2) / 
                        ((2*np.pi)**2 * self.dens * (1e-3) * (self.vc)**2 * (6.24e21) * (1e-7)**3 * self.Vs**4 * self.r0))
            self.j0_1 = self.j0_1 / self.hbar
        
        # j0_FGR calculation
        self.j0_FGR = ((self.DvDc**2 * (1e3)**2 * (self.vc)**2 * (1e9)**2 * (1e-12)**2) / 
                      (2*np.pi * self.dens * (1e-3) * (self.vc)**2 * (6.24e21) * (1e-7)**3 * self.Vs**5)) / self.hbar
    
    def _convert_units(self):
        """Convert parameters to computational units"""
        self.gamma1_comp = self.gamma1 * 1e-3 / self.hbar
        self.gamma2_comp = self.gamma2 * 1e-3 / self.hbar
        self.gammac_comp = self.gammac * 1e-3 / self.hbar
        
        if self.correlator == 'LP':
            self.g_comp = self.g * 1e-3 / self.hbar
        else:
            self.g_comp = self.g  # For NQD, g is already calculated in computational units
            
        self.g1_comp = self.g1 * 1e-3 / self.hbar
        self.g2_comp = self.g2 * 1e-3 / self.hbar
        
        if self.cavity == 1:
            self.gd_comp = self.gd * 1e-3 / self.hbar
            
        self.detuning_comp = self.detuning * 1e-3 / self.hbar
        self.det = self.detuning_comp
    
    def _calculate_time_params(self):
        """Calculate time-related parameters"""
        if self.dotshape == 'spherical':
            self.tauib = np.sqrt(2) * np.pi * self.l / self.Vs
        elif self.dotshape == 'smartie':
            self.tauib = np.sqrt(2) * np.pi * self.lbar / self.Vs
        
        # Choose correct delay time t0
        if self.no_of_QDs == 2:
            self.t0 = self.r0 / self.Vs
        if self.no_of_QDs == 1:
            self.t0 = 0
        
        if self.sharebath == 1:
            self.dt = (self.t0 + self.factortau * self.tauib) / (self.L + 1)
        elif self.sharebath == 0:
            self.dt = self.factortau * self.tauib / (self.L + 1)
            
        if self.no_of_QDs == 1:
            self.dt = (self.factortau * self.tauib) / (self.L + 1)
    
    def _calculate_polaron_params(self):
        """Calculate polaron shift and SHR values"""
        self.omp = PolaronShift(self.j0, self.w0)
        self.ompinim = PolaronShift_inim(self.j0, self.w0, self.r0, self.l)
        self.SHR = S_inin(self.T_ps, self.j0, self.w0)
        
        if self.no_of_QDs == 2:
            self.SHRinim = S_inim(self.T_ps, self.j0_1, self.w0, self.r0, self.Vs)
            
    def Sanalyt(self, r0):
        """Analytical S function calculator"""
        return S_inin(self.T_ps, self.j0, self.w0) - S_inim(self.T_ps, self.j0_1, self.w0, r0, self.Vs)
    
    def _setup_frequencies(self):
        """Setup quantum dot frequencies"""
        # Set up real parts, QD energies
        if self.cavity == 0:
            self.Om1 = 0
            self.Om2 = self.detuning_comp
        elif self.cavity == 1:
            self.Om1 = 0.0
            self.Om2 = 0.0 + self.detuning_comp
            self.OmC = self.Om1
        
        # Set up complex frequencies with decay rates
        self.w_qd1 = (self.Om1 - 1j * self.gamma1_comp)
        self.w_qd2 = (self.Om2 - 1j * self.gamma2_comp)
        
        if self.cavity == 1:
            self.w_c = self.OmC - 1j * self.gammac_comp
    
    def _setup_polariton_states(self):
        """Setup polariton states in JC model"""
        self.w1 = -np.sqrt(self.g_comp**2 + (0.5 * (self.w_qd1 - self.w_qd2))**2) + (self.w_qd1 + self.w_qd2) / 2
        self.w2 = np.sqrt(self.g_comp**2 + (0.5 * (self.w_qd1 - self.w_qd2))**2) + (self.w_qd1 + self.w_qd2) / 2
        
        self.Delta_xc = np.sqrt((0.5 * (self.w_qd1 - self.w_qd2))**2 + self.g_comp**2) - 0.5 * (self.w_qd1 - self.w_qd2)
        
        # Elements which contribute to diagonalising JC Hamiltonian
        self.alpha_jc = self.Delta_xc / np.sqrt(self.Delta_xc**2 + self.g_comp**2)
        self.beta_jc = self.g_comp / np.sqrt(self.Delta_xc**2 + self.g_comp**2)
    
    def _setup_measurement_channels(self):
        """Setup measurement channels and vectors based on correlator type"""
        if self.correlator == 'LP' and self.cavity == 0:
            self.d = 2
            # QD2 channel
            self.Q2 = np.matrix([[0], [1]])
            self.o2 = self.Q2.T
            
            # QD1 channel
            self.Q1 = np.matrix([[1], [0]])
            self.o1 = self.Q1.T
            
            self.alpha = np.array([1, 0])  # left vector for V1
            self.mu = np.array([0, 1])     # left vector for V2
            self.beta = np.array([0, 0])   # right vector for V1
            self.nu = np.array([0, 0])     # right vector for V2
            
        elif self.correlator == 'NQD':
            self.d = 5
            # QD2 channel
            self.Q2 = np.matrix([[0], [0], [0], [0], [1]])
            self.o2 = self.Q2.T
            
            # QD1 channel
            self.Q1 = np.matrix([[0], [1], [0], [0], [0]])
            self.o1 = self.Q1.T
            
            self.alpha = np.array([0, 1, 1, 0, 0])  # left vector for V1
            self.mu = np.array([0, 0, 0, 1, 1])     # left vector for V2
            self.beta = np.array([0, 1, 0, 1, 0])   # right vector for V1
            self.nu = np.array([0, 0, 1, 0, 1])     # right vector for V2
            
            # For the Bloch sphere
            self.Ox = np.matrix([[0], [0], [2], [0], [0]]).T
            self.Oy = np.matrix([[0], [0], [0], [2], [0]]).T
            self.Oz = np.matrix([[0], [1], [0], [0], [-1]]).T
    
    def update(self, **kwargs):
        """Update parameters and recalculate derived values"""
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Parameter '{key}' does not exist and will be ignored.")
        
        # Recalculate derived parameters
        self._init_derived_params()
        self.cumulant_generator()
    
    def get_cumulant_params(self):
        """Get parameters needed for CumulantGenerator function"""
        return {
            'L': self.L, 
            'r0': self.r0, 
            'g1': self.g1_comp, 
            'g2': self.g2_comp, 
            'gd': self.gd_comp if hasattr(self, 'gd_comp') else 0, 
            'w_qd1': self.w_qd1, 
            'w_qd2': self.w_qd2, 
            'w_c': self.w_c if hasattr(self, 'w_c') else 0, 
            'tauib': self.tauib, 
            'factortau': self.factortau, 
            'ec': self.ec, 
            'mc': self.mc, 
            'omp': self.omp, 
            'shr': self.SHR, 
            'j0': self.j0, 
            'j0_1': self.j0_1 if hasattr(self, 'j0_1') else 0, 
            'w0': self.w0, 
            'T': self.T_ps, 
            'Vs': self.Vs, 
            'l': self.l, 
            'dotshape': self.dotshape, 
            'sharebath': self.sharebath, 
            'lp': self.lp
        }
    
    def generate_label(self):
        """Generate a label for data and plots"""
        if self.cavity == 0:
            label = f"{self.correlator}_{self.dotshape}_{int(self.no_of_QDs)}QDs_{int(self.cavity)}cavs_" \
                    f"T{round(self.T_ps*self.hbar/self.kb)}_g{round(self.g_comp*self.hbar*1e3)}_" \
                    f"R{np.round(np.float64(self.r0),3)}_L{self.L}_l{np.round(np.float64(self.l),1)}_" \
                    f"lp{np.round(np.float64(self.lp),1)}_EC{self.ec}_MC{self.mc}_DvDc{self.DvDc}_" \
                    f"det{np.round(self.detuning_comp*self.hbar*1e3,1)}_sharebath{self.sharebath}_" \
                    f"threshold{self.threshold_factor}_factortau{self.factortau}_tf{self.tf}"
        elif self.cavity == 1 and self.no_of_QDs == 1:
            label = f"{self.correlator}_{self.dotshape}_QDs{int(self.no_of_QDs)}_cav{int(self.cavity)}_" \
                    f"T{round(self.T_ps*self.hbar/self.kb)}_g{round(self.gd_comp*self.hbar*1e3)}_" \
                    f"L{self.L}_l{np.round(np.float64(self.l),1)}_lp{np.round(np.float64(self.lp),1)}_" \
                    f"EC{self.ec}_MC{self.mc}_DvDc{self.DvDc}_det{np.round(self.detuning_comp*self.hbar*1e3,1)}_" \
                    f"sharebath{self.sharebath}_threshold{self.threshold_factor}_factortau{self.factortau}_tf{self.tf}"
        elif self.cavity == 1 and self.no_of_QDs == 2:
            label = f"{self.correlator}_{self.dotshape}_QDs{int(self.no_of_QDs)}_cav{int(self.cavity)}_" \
                    f"T{round(self.T_ps*self.hbar/self.kb)}_1g{round(self.g1_comp*self.hbar*1e3)}_" \
                    f"2g{round(self.g2_comp*self.hbar*1e3)}_g{round(self.gd_comp*self.hbar*1e3)}_" \
                    f"R{np.round(np.float64(self.r0),3)}_L{self.L}_l{np.round(np.float64(self.l),1)}_" \
                    f"lp{np.round(np.float64(self.lp),1)}_EC{self.ec}_MC{self.mc}_DvDc{self.DvDc}_" \
                    f"det{np.round(self.detuning_comp*self.hbar*1e3,1)}_sharebath{self.sharebath}_" \
                    f"factortau{self.factortau}_threshold{self.threshold_factor}_tf{self.tf}"
        return label
    
    # def generate_cumulants(self):
    #     """Generate cumulants using the current parameters"""
    #     # This replaces the external CumulantGenerator function
    #     M1, Qlist, exp_k0_factor, exc_channel, measure_channel = self._cumulant_generator()
    #     return M1, Qlist, exp_k0_factor, exc_channel, measure_channel
    
    def cumulant_generator(self):
        """Internal implementation of CumulantGenerator function"""
        # Get required parameters
        L = self.L
        r0 = self.r0
        g1 = self.g1_comp
        g2 = self.g2_comp
        gd = self.gd_comp if hasattr(self, 'gd_comp') else 0
        w_qd1 = self.w_qd1
        w_qd2 = self.w_qd2
        gamma1=self.gamma1_comp
        gamma2=self.gamma2_comp
        gammac=self.gammac_comp
        detuning=self.detuning_comp
        w_c = self.w_c if hasattr(self, 'w_c') else 0
        tauib = self.tauib
        factortau = self.factortau
        ec = self.ec
        mc = self.mc
        omp = self.omp
        shr = self.SHR
        j0 = self.j0
        j0_1 = self.j0_1 if hasattr(self, 'j0_1') else 0
        w0 = self.w0
        T = self.T_ps
        Vs = self.Vs
        l = self.l
        dotshape = self.dotshape
        sharebath = self.sharebath
        lp = self.lp

        # Implementation for 2 QDs case
        if self.no_of_QDs == 2:
            # Define Cumulants class for QD-QD or QD-QD-Cavity system
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
                    cumulants = [K0_inim]
                    cumulantsinin = [K0]
                    for n in range(1, L + 1):
                        Knn, Knn_inin = self.Kn(cumulants, cumulantsinin, dt, n, r0, j0, j0_1, w0, T, Vs, l, dotshape, sharebath, lp)
                        cumulants.append(Knn)
                        cumulantsinin.append(Knn_inin)
                    return cumulants, cumulantsinin
        
        # Implementation for 1 QD case
        else:  # self.no_of_QDs == 1
            # Define Cumulants class for QD-cavity system
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

        # Handle LP correlator with 2 QDs
        if self.correlator == 'LP' and self.no_of_QDs == 2:
            if ec == '1':
                exc_channel = 0
            if ec == '2':
                exc_channel = 1
            if ec == 'C':
                exc_channel = 2
            if mc == '1':
                measure_channel = 0
            if mc == '2':
                measure_channel = 1
            if mc == 'C':
               measure_channel = 2    
            
            t0 = r0 / Vs
            dt = (t0 + factortau * tauib) / (L + 1)       
            phonon_uncoupled_mode = 2
            phonon_uncoupled_permutation = -1
            
            # Import required functions from your modules
            LF = LFpol_qdqdcav(g1, g2, gd, w_qd1, w_qd2, w_c)
            M1 = expm(-1j * LF * dt)
            
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
                'lp': lp
            }  
            
            cumulants_obj = Cumulants()
            cumulants_obj.update_parameters(**params_cumulant)
            cumulants, cumulants_inin = cumulants_obj.cu(L, dt, **params_cumulant)
            
            Q0 = np.array([
                [M1[0, 0] * np.exp(cumulants_inin[0] + 2 * cumulants_inin[1]), 
                 M1[0, 1] * np.exp(cumulants_inin[0] + 2 * cumulants[1]), 
                 M1[0, 2]],
                [M1[1, 0] * np.exp(cumulants_inin[0] + 2 * cumulants[1]), 
                 M1[1, 1] * np.exp(cumulants_inin[0] + 2 * cumulants_inin[1]), 
                 M1[1, 2]],
                [M1[2, 0] * np.exp(cumulants_inin[0]), 
                 M1[2, 1] * np.exp(cumulants_inin[0]), 
                 M1[2, 2]]
            ])
            
            Qlist = []
            Qlist.append(Q0)
            
            for i in range(int(L - 1)):
                Qlist.append(np.array([
                    [np.exp(2 * cumulants_inin[i + 2]), np.exp(2 * cumulants[i + 2]), 1],
                    [np.exp(2 * cumulants[i + 2]), np.exp(2 * cumulants_inin[i + 2]), 1],
                    [1, 1, 1]
                ]))
                
            exp_k0_factor = np.array([np.exp(cumulants_inin[0]), np.exp(cumulants_inin[0]), 1])
            exp_k0_factor = exp_k0_factor[measure_channel]
            self.M1=M1
            self.Qlist=Qlist
            self.exp_k0_factor=exp_k0_factor
            
            # return M1, Qlist, exp_k0_factor
            
        # Handle LP correlator with 1 QD
        elif self.correlator == 'LP' and self.no_of_QDs == 1:
            if ec == '1':
                exc_channel = 0
            if ec == 'C':
                exc_channel = 1
            if mc == '1':
                measure_channel = 0
            if mc == 'C':
               measure_channel = 1    
               
            dt = (factortau * tauib) / (L + 1)   
            phonon_uncoupled_mode = 1
            phonon_uncoupled_permutation = -1
            
            # Import required functions from your modules
            D, ww, U1, V1 = DiagM_qdcav(gd, 0, 0, 0, detuning, omp)
            DD = np.diag(np.exp(-1j * ww * dt))
            M1 = U1 * DD * V1
            
            params_cumulant = {
                'omp': omp,
                'shr': shr,
                'j0': j0,
                'T': T,
                'w0': w0
            }  
            
            Cumulants_obj = Cumulants_1qd()
            Cumulants_obj.update_parameters(**params_cumulant)
            cumulants_inin = Cumulants_obj.cu(L, dt, **params_cumulant)
            cumulants = cumulants_inin
            
            Q0 = np.array([
                [M1[0, 0] * np.exp(cumulants[0] + 2 * cumulants[1]), M1[0, 1]],
                [M1[1, 0] * np.exp(cumulants[0]), M1[1, 1]]
            ])
            
            Qlist = []
            Qlist.append(Q0)
            
            for i in range(int(L - 1)):
                Qlist.append(np.array([
                    [np.exp(2 * cumulants[i + 2]), 1],
                    [1, 1]
                ]))
            exp_k0_factor=np.array([np.exp(cumulants_inin[0]), 1])
            exp_k0_factor=exp_k0_factor[measure_channel]
            self.M1=M1
            self.Qlist=Qlist
            self.exp_k0_factor=exp_k0_factor
            
            # return M1, Qlist, exp_k0_factor
        
        elif self.correlator == 'NQD' and self.no_of_QDs == 2:
            if ec=='2':
                exc_channel=4
            else:
                exc_channel=1
            if mc=='2':
                measure_channel=4
            else:
                measure_channel=1
            
            t0=r0/Vs
            dt= (t0 + factortau*tauib )/(L+1)          
            phonon_uncoupled_mode=0
            phonon_uncoupled_permutation=0
            LF = LFpop(gd,  np.real(w_qd1), np.real(w_qd2), gamma1, gamma2) 
            M1=expm(-1j * LF * dt)
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
            self.M1=M1
            self.Qlist=Qlist
            self.exp_k0_factor=exp_k0_factor
            