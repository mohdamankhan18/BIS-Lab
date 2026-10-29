import numpy as np
import cv2
import scipy.special as special
from scipy.ndimage import sobel

# ==========================================
# 1. ENHANCEMENT KERNEL & FITNESS FUNCTION
# ==========================================

def calculate_local_stats(image, window_size=3):
    """Calculates local mean and local standard deviation over a window."""
    kernel = np.ones((window_size, window_size), dtype=np.float64) / (window_size ** 2)
    local_mean = cv2.filter2D(image, -1, kernel)
    
    # Local variance: E[X^2] - (E[X])^2
    local_sq_mean = cv2.filter2D(image ** 2, -1, kernel)
    local_var = np.maximum(0, local_sq_mean - local_mean ** 2)
    local_std = np.sqrt(local_var)
    
    return local_mean, local_std

def apply_enhancement_kernel(image, params, local_mean, local_std, global_mean):
    """
    Applies Eq. (5): O(i,j) = (mu_L)^a + k * (I(i,j) - c * mu_L) * mu_G / (sigma_L + b)
    """
    a, b, c, k = params
    
    # First term: brightening/smoothing
    term1 = np.power(np.maximum(0, local_mean), a)
    
    # Second term: local contrast stretching
    numerator = (image - c * local_mean) * global_mean
    denominator = local_std + b
    term2 = k * (numerator / denominator)
    
    # Enhanced output mapped to [0, 255]
    enhanced = term1 + term2
    enhanced = np.clip(enhanced, 0, 255)
    return enhanced

def evaluate_fitness(enhanced_img):
    """
    Evaluates Eq. (6): Fitness = log(E(O_s) + e) * (Ne(O_s) / (M*N)) * exp(H(O))
    """
    M, N = enhanced_img.shape
    
    # Sobel Gradient Computation
    gx = sobel(enhanced_img, axis=1)
    gy = sobel(enhanced_img, axis=0)
    sobel_img = np.sqrt(gx**2 + gy**2)
    
    # Sum of edge intensities: E(O_s)
    E_Os = np.sum(sobel_img)
    
    # Number of edge pixels above a threshold (Ne)
    threshold = np.mean(sobel_img) + np.std(sobel_img)
    Ne_Os = np.sum(sobel_img > threshold)
    
    # Image Entropy: H(O)
    hist, _ = np.histogram(enhanced_img, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    # Combined Fitness Criteria
    e = np.e
    term_edge_sum = np.log(E_Os + e)
    term_edge_cnt = Ne_Os / (M * N)
    term_entropy = np.exp(entropy)
    
    fitness = term_edge_sum * term_edge_cnt * term_entropy
    return fitness

# ==========================================
# 2. ADAPTIVE CUCKOO SEARCH ALGORITHM (ACSEA)
# ==========================================

class ACSEA:
    def __init__(self, img, num_nests=20, max_iter=30, pa=0.25):
        self.img = img.astype(np.float64)
        self.num_nests = num_nests
        self.max_iter = max_iter
        self.pa = pa
        
        # Parameter bounds: a in [0, 1.5], b in [0, 0.5], c in [0, 1], k in [0.5, 1.5]
        self.lb = np.array([0.0, 0.001, 0.0, 0.5])
        self.ub = np.array([1.5, 0.500, 1.0, 1.5])
        self.dim = len(self.lb)
        
        # Precompute image global statistics
        self.global_mean = np.mean(self.img)
        self.local_mean, self.local_std = calculate_local_stats(self.img)

    def chaotic_initialization(self, T=5):
        """Chaotic initialization using Logistic Map (Eq. 9)"""
        nests = np.zeros((self.num_nests, self.dim))
        ch = 0.7  # Initial seed
        K_param = 4.0  # Logistic map parameter
        
        for i in range(self.num_nests):
            for j in range(self.dim):
                for _ in range(T):
                    ch = K_param * ch * (1.0 - ch)
                nests[i, j] = self.lb[j] + (self.ub[j] - self.lb[j]) * ch
        return nests

    def levy_flight(self, nests, best_nest, best_fit, worst_fit, fitness_vals, beta=1.5):
        """Adaptive Lévy Flight Generation (Eq. 10 - Eq. 12)"""
        sigma_u = (special.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                  (special.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        sigma_v = 1.0
        
        new_nests = np.copy(nests)
        fit_diff = max(best_fit - worst_fit, 1e-8)
        
        for i in range(self.num_nests):
            # Adaptive step size scaling alpha
            alpha = (best_fit - fitness_vals[i]) / fit_diff
            
            u = np.random.normal(0, sigma_u, size=self.dim)
            v = np.random.normal(0, sigma_v, size=self.dim)
            step = u / (np.abs(v) ** (1 / beta))
            
            step_size = alpha * step * (nests[i] - best_nest)
            new_nests[i] = nests[i] + step_size * np.random.randn(self.dim)
            # Clip bounds
            new_nests[i] = np.clip(new_nests[i], self.lb, self.ub)
            
        return new_nests

    def mutative_randomization(self, nests, best_nest, current_iter):
        """Discovery and Mutative Randomization Strategy (Eq. 13 - Eq. 14)"""
        new_nests = np.copy(nests)
        prob_threshold = 1.0 - (current_iter / self.max_iter)
        
        for i in range(self.num_nests):
            r1, r2, r3, r4 = np.random.choice(self.num_nests, 4, replace=False)
            
            if np.random.rand() > prob_threshold:
                # CS/best/2 strategy
                mutation = best_nest + self.pa * (nests[r1] - nests[r2] + nests[r3] - nests[r4])
            else:
                # CS/rand/1 strategy
                mutation = nests[r1] + self.pa * (nests[r2] - nests[r3])
                
            new_nests[i] = np.clip(mutation, self.lb, self.ub)
            
        return new_nests

    def get_fitness(self, nest_params):
        """Applies kernel and evaluates fitness for a given parameter set."""
        enhanced = apply_enhancement_kernel(
            self.img, nest_params, self.local_mean, self.local_std, self.global_mean
        )
        return evaluate_fitness(enhanced)

    def optimize(self):
        # 1. Population Initialization
        nests = self.chaotic_initialization()
        fitness_vals = np.array([self.get_fitness(n) for n in nests])
        
        best_idx = np.argmax(fitness_vals)
        best_fitness = fitness_vals[best_idx]
        best_nest = nests[best_idx].copy()
        
        print(f"Initial Best Fitness: {best_fitness:.4f}")
        
        # 2. Main Optimization Loop
        for iter_idx in range(1, self.max_iter + 1):
            worst_fitness = np.min(fitness_vals)
            
            # --- Adaptive Lévy Flight Step ---
            levy_nests = self.levy_flight(nests, best_nest, best_fitness, worst_fitness, fitness_vals)
            for i in range(self.num_nests):
                f_new = self.get_fitness(levy_nests[i])
                if f_new > fitness_vals[i]:
                    fitness_vals[i] = f_new
                    nests[i] = levy_nests[i]
            
            # --- Update Global Best ---
            best_idx = np.argmax(fitness_vals)
            if fitness_vals[best_idx] > best_fitness:
                best_fitness = fitness_vals[best_idx]
                best_nest = nests[best_idx].copy()

            # --- Mutative Randomization Step ---
            mutated_nests = self.mutative_randomization(nests, best_nest, iter_idx)
            for i in range(self.num_nests):
                f_new = self.get_fitness(mutated_nests[i])
                if f_new > fitness_vals[i]:
                    fitness_vals[i] = f_new
                    nests[i] = mutated_nests[i]
            
            # --- Update Global Best ---
            best_idx = np.argmax(fitness_vals)
            if fitness_vals[best_idx] > best_fitness:
                best_fitness = fitness_vals[best_idx]
                best_nest = nests[best_idx].copy()

            if iter_idx % 5 == 0 or iter_idx == 1:
                print(f"Iteration {iter_idx:02d}/{self.max_iter} | Best Fitness: {best_fitness:.4f}")

        # Generate final enhanced image using optimal parameters
        final_enhanced = apply_enhancement_kernel(
            self.img, best_nest, self.local_mean, self.local_std, self.global_mean
        )
        return final_enhanced, best_nest, best_fitness

# ==========================================
# 3. EXECUTION & DEMO
# ==========================================

if __name__ == "__main__":
    np.random.seed(42)  # For reproducible demo results
    
    # Generate a low-contrast synthetic satellite-like image
    grid = np.linspace(0, 4 * np.pi, 256)
    x, y = np.meshgrid(grid, grid)
    synthetic_pattern = (np.sin(x) + np.cos(y)) * 30 + 100
    
    # Add contrast reduction (dark/foggy image simulation)
    low_contrast_img = np.clip(synthetic_pattern * 0.35 + 40, 0, 255).astype(np.uint8)
    
    print("--- Running Adaptive Cuckoo Search Enhancement (ACSEA) ---")
    acsea = ACSEA(low_contrast_img, num_nests=15, max_iter=20)
    enhanced_img, best_params, best_fit = acsea.optimize()
    
    a_opt, b_opt, c_opt, k_opt = best_params
    
    print("\n" + "="*45)
    print("Optimization Complete!")
    print(f"Optimal Parameters Found: a={a_opt:.4f}, b={b_opt:.4f}, c={c_opt:.4f}, k={k_opt:.4f}")
    print(f"Final Image Fitness: {best_fit:.4f}")
    print(f"Input Image Intensity Range: [{low_contrast_img.min()}, {low_contrast_img.max()}]")
    print(f"Enhanced Image Intensity Range: [{int(enhanced_img.min())}, {int(enhanced_img.max())}]")
    print("="*45)