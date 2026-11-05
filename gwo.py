import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PHYSICAL PARAMETERS & INVERSE KINEMATICS
# ==========================================
# Link lengths from Table I (in mm)
L1 = 400.0  # Upper leg length (l1)
L2 = 410.0  # Lower leg length (l2)

def inverse_kinematics(x, y):
    """
    Computes hip angle (Theta_1) and knee angle (Theta_2) for target (x, y).
    Angles are in degrees to match Figures 7 & 8.
    """
    r = np.sqrt(x**2 + y**2)
    # Cosine rule for knee angle
    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arctan2(-np.sqrt(1 - cos_theta2**2), cos_theta2)
    
    # Cosine rule / polar components for hip angle
    beta = np.arctan2(y, x)
    gamma = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    theta1 = beta - gamma
    
    return np.degrees(theta1), np.degrees(theta2)

def forward_kinematics(theta1_deg, theta2_deg):
    """
    Computes (x, y) end-effector position from joint angles (in degrees).
    """
    t1 = np.radians(theta1_deg)
    t2 = np.radians(theta2_deg)
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return x, y

# ==========================================
# 2. REFERENCE TRAJECTORY SETUP (Figure 9)
# ==========================================
# Single footstep semi-circular trajectory in X-Y plane (in mm)
t_sim = np.linspace(0, 10, 100)  # 10 seconds simulation, 100 timesteps
dt = t_sim[1] - t_sim[0]

# Semi-circular step: X spans 0 to 200 mm, Y follows arc centered at Y = -600 mm
x_ref = 100.0 - 100.0 * np.cos(np.pi * t_sim / 10.0)
y_ref = -600.0 + 100.0 * np.sin(np.pi * t_sim / 10.0)

# Target joint trajectories
theta1_ref = np.zeros(len(t_sim))
theta2_ref = np.zeros(len(t_sim))
for i in range(len(t_sim)):
    theta1_ref[i], theta2_ref[i] = inverse_kinematics(x_ref[i], y_ref[i])

# ==========================================
# 3. QUADRUPED LEG SIMULATION & OBJECTIVE FUNCTION
# ==========================================
def simulate_system(gains):
    """
    Simulates the two PID controllers regulating joint angles (Theta_1, Theta_2)
    and calculates actual trajectory (X, Y) and objective function value J.
    gains: [Kp1, Ki1, Kd1, Kp2, Ki2, Kd2]
    """
    Kp1, Ki1, Kd1, Kp2, Ki2, Kd2 = gains
    
    # Joint inertia / damping parameters for realistic simplified dynamics
    J1, B1 = 0.8, 0.5
    J2, B2 = 0.5, 0.3
    
    # Initial state
    th1, th2 = theta1_ref[0], theta2_ref[0]
    w1, w2 = 0.0, 0.0
    int_err1, int_err2 = 0.0, 0.0
    prev_err1, prev_err2 = 0.0, 0.0
    
    x_actual = np.zeros(len(t_sim))
    y_actual = np.zeros(len(t_sim))
    
    for i in range(len(t_sim)):
        # Errors
        err1 = theta1_ref[i] - th1
        err2 = theta2_ref[i] - th2
        
        # Integrals
        int_err1 += err1 * dt
        int_err2 += err2 * dt
        
        # Derivatives
        der1 = (err1 - prev_err1) / dt if i > 0 else 0.0
        der2 = (err2 - prev_err2) / dt if i > 0 else 0.0
        
        # PID control outputs
        u1 = Kp1 * err1 + Ki1 * int_err1 + Kd1 * der1
        u2 = Kp2 * err2 + Ki2 * int_err2 + Kd2 * der2
        
        # Torque saturation limits (+/- 5 Nm as specified in Fig. 5)
        u1 = np.clip(u1, -5.0, 5.0)
        u2 = np.clip(u2, -5.0, 5.0)
        
        # Equations of motion update
        alpha1 = (u1 - B1 * w1) / J1
        alpha2 = (u2 - B2 * w2) / J2
        
        w1 += alpha1 * dt
        w2 += alpha2 * dt
        
        th1 += w1 * dt
        th2 += w2 * dt
        
        prev_err1 = err1
        prev_err2 = err2
        
        # Compute forward position
        x_actual[i], y_actual[i] = forward_kinematics(th1, th2)
        
    # Objective function J = norm(sqrt((Xref - X)^2 + (Yref - Y)^2)) (Eq. 10)
    J = np.linalg.norm(np.sqrt((x_ref - x_actual)**2 + (y_ref - y_actual)**2))
    return J, x_actual, y_actual

# ==========================================
# 4. GREY WOLF OPTIMIZER (GWO) ALGORITHM
# ==========================================
def grey_wolf_optimizer(num_agents=30, max_iter=100, dim=6, lb=0, ub=1000):
    """
    Implements GWO algorithm (Equations 1-7, Table II, Table V)
    """
    # Initialize position of wolves randomly between lb (0) and ub (1000)
    Positions = np.random.uniform(lb, ub, (num_agents, dim))
    
    # Initialize Alpha, Beta, Delta positions & scores
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")
    
    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")
    
    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")
    
    convergence_curve = np.zeros(max_iter)
    
    for l in range(max_iter):
        for i in range(num_agents):
            # Clip positions to bounds [0, 1000]
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            
            # Evaluate fitness
            fitness, _, _ = simulate_system(Positions[i, :])
            
            # Update Alpha, Beta, Delta (Social Hierarchy)
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()
                
            if fitness > Alpha_score and fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()
                
            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()
                
        # Linearly decrease parameter 'a' from 2 to 0
        a = 2.0 - l * (2.0 / max_iter)
        
        # Update positions of search agents including Omegas
        for i in range(num_agents):
            for j in range(dim):
                # Alpha update (Eq. 3, 5, 6)
                r1, r2 = np.random.random(), np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha
                
                # Beta update (Eq. 3, 5, 6)
                r1, r2 = np.random.random(), np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta
                
                # Delta update (Eq. 3, 5, 6)
                r1, r2 = np.random.random(), np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta
                
                # Final position update (Eq. 7)
                Positions[i, j] = (X1 + X2 + X3) / 3.0
                
        convergence_curve[l] = Alpha_score
        print(f"Iteration {l+1}/{max_iter} | Best Cost (J): {Alpha_score:.2f}")
        
    return Alpha_pos, Alpha_score, convergence_curve

# ==========================================
# 5. EXECUTION & RESULTS VISUALIZATION
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)  # For reproducible simulation run
    
    print("--- Running Grey Wolf Optimizer for Quadruped PID Tuning ---")
    best_gains, best_cost, conv_curve = grey_wolf_optimizer(num_agents=30, max_iter=100)
    
    print("\nOptimal Gains Obtained by GWO:")
    print(f"PID 1 (Hip)  : Kp = {best_gains[0]:.2f}, Ki = {best_gains[1]:.4f}, Kd = {best_gains[2]:.2f}")
    print(f"PID 2 (Knee) : Kp = {best_gains[3]:.2f}, Ki = {best_gains[4]:.2f}, Kd = {best_gains[5]:.2f}")
    
    # Run final simulation with tuned gains
    cost, x_actual, y_actual = simulate_system(best_gains)
    
    # Plot Trajectory (Figure 9 in paper)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_ref, y_ref, 'k--', label='Reference', linewidth=2)
    plt.plot(x_actual, y_actual, 'r-', label='GWO Tuned PID', linewidth=2)
    plt.title("Footstep Trajectory Tracking (Fig. 9)")
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.legend()
    plt.grid(True)
    
    # Plot Convergence Curve (Figure 11 in paper)
    plt.subplot(1, 2, 2)
    plt.plot(conv_curve, 'r-', label='GWO Convergence', linewidth=2)
    plt.title("Cost Function Convergence (Fig. 11)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost Function (min J)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()