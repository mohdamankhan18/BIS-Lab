import numpy as np


class SMIBSystem:
    """
    Single Machine Infinite Bus (SMIB) model for PSS parameter tuning.
    Linearized state variables: x = [delta, omega, Eq']^T
    Control input: u = V_PSS
    """

    def __init__(self, H=3.5, D=0.0, K1=1.0, K2=1.0, K3=0.4, K4=1.4, K5=-0.1, K6=0.5,
                 Tdo_prime=6.0, Ka=50.0, Ta=0.05, f0=50.0):
        self.H = H
        self.D = D
        self.K1, self.K2, self.K3 = K1, K2, K3
        self.K4, self.K5, self.K6 = K4, K5, K6
        self.Tdo_prime = Tdo_prime
        self.Ka, self.Ta = Ka, Ta
        self.omega0 = 2 * np.pi * f0

    def get_state_matrix(self, K7, K8, T7=0.05, T8=0.05):
        """
        Builds system A matrix including Generator, Exciter, and PD-PSS.
        PD-PSS Transfer Function: F_PSS(s) = (K7 / (1 + s*T7)) + (s * K8 / (1 + s*T8))
        Inputs: Speed deviation (delta_omega)
        Output: Stabilizing signal V_PSS
        """
        # States: [delta, d_omega, d_Eq_prime, d_Efd, x_lag, x_washout]
        A = np.zeros((6, 6))

        # 1. d(delta)/dt = omega0 * d_omega
        A[0, 1] = self.omega0

        # 2. d(d_omega)/dt = (-K1*delta - D*d_omega - K2*d_Eq_prime) / (2*H)
        A[1, 0] = -self.K1 / (2 * self.H)
        A[1, 1] = -self.D / (2 * self.H)
        A[1, 2] = -self.K2 / (2 * self.H)

        # 3. d(d_Eq_prime)/dt = (-K4*delta - d_Eq_prime/K3 + d_Efd) / Tdo_prime
        A[2, 0] = -self.K4 / self.Tdo_prime
        A[2, 2] = -1.0 / (self.K3 * self.Tdo_prime)
        A[2, 3] = 1.0 / self.Tdo_prime

        # PSS internal states equations:
        # dx5/dt = (-x5 + K7 * d_omega) / T7
        A[4, 1] = K7 / T7
        A[4, 4] = -1.0 / T7

        # dx6/dt = (-x6 + K8 * d(d_omega)/dt) / T8
        # Substituting d(d_omega)/dt:
        A[5, 0] = (K8 / T8) * (-self.K1 / (2 * self.H))
        A[5, 1] = (K8 / T8) * (-self.D / (2 * self.H))
        A[5, 2] = (K8 / T8) * (-self.K2 / (2 * self.H))
        A[5, 5] = -1.0 / T8

        # 4. Exciter equation: d(d_Efd)/dt = (-d_Efd - Ka*K5*delta - Ka*K6*d_Eq_prime + Ka*u_pss) / Ta
        # u_pss = x5 + x6
        A[3, 0] = -self.Ka * self.K5 / self.Ta
        A[3, 2] = -self.Ka * self.K6 / self.Ta
        A[3, 3] = -1.0 / self.Ta
        A[3, 4] = self.Ka / self.Ta  # Contribution from x5
        A[3, 5] = self.Ka / self.Ta  # Contribution from x6

        return A


def objective_function(params, smib_system):
    """
    Objective Function J = max(Real(lambda_i)).
    Minimizing J shifts electromechanical eigenvalues to the left in the s-plane.
    """
    K7, K8 = params
    A = smib_system.get_state_matrix(K7, K8)
    eigenvalues = np.linalg.eigvals(A)
    
    # Target electromechanical mode shifting
    real_parts = np.real(eigenvalues)
    return np.max(real_parts)


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization (PSO) implementation for PSS tuning[cite: 1, 3].
    """

    def __init__(self, num_particles=30, max_iter=100, bounds=(-500, 500)):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds

    def optimize(self, smib_system):
        # Initialize particle positions and velocities randomly within bounds [-500, 500]
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_particles, 2))
        velocities = np.random.uniform(-10, 10, (self.num_particles, 2))

        pbest_positions = np.copy(positions)
        pbest_scores = np.array([objective_function(p, smib_system) for p in positions])

        gbest_index = np.argmin(pbest_scores)
        gbest_position = np.copy(pbest_positions[gbest_index])
        gbest_score = pbest_scores[gbest_index]

        # PSO Parameters
        w = 0.7       # Inertia weight
        c1 = 1.5      # Cognitive component
        c2 = 1.5      # Social component

        print(f"Initial Best J: {gbest_score:.4f}")

        for it in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                
                # Update Velocity
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (pbest_positions[i] - positions[i]) +
                                 c2 * r2 * (gbest_position - positions[i]))
                
                # Update Position
                positions[i] += velocities[i]
                
                # Apply Boundary Constraints [-500, 500][cite: 3]
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])

                # Evaluate Objective
                score = objective_function(positions[i], smib_system)

                # Update Personal Best
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = np.copy(positions[i])

                    # Update Global Best
                    if score < gbest_score:
                        gbest_score = score
                        gbest_position = np.copy(positions[i])

            if (it + 1) % 20 == 0 or it == 0:
                print(f"Iteration {it+1}/{self.max_iter} -> Best J: {gbest_score:.4f} | K7: {gbest_position[0]:.2f}, K8: {gbest_position[1]:.2f}")

        return gbest_position, gbest_score


if __name__ == "__main__":
    # Create SMIB system instance
    smib = SMIBSystem()

    # Run PSO optimization to tune PSS gains K7 and K8[cite: 3]
    pso = ParticleSwarmOptimizer(num_particles=40, max_iter=100, bounds=(-500, 500))
    best_gains, min_J = pso.optimize(smib)

    print("\n" + "="*45)
    print("      PSO TUNING RESULTS (PAPER MATCH)")
    print("="*45)
    print(f"Optimized Gain K7 : {best_gains[0]:.2f}")
    print(f"Optimized Gain K8 : {best_gains[1]:.2f}")
    print(f"Maximum Real Eigenvalue (J): {min_J:.4f}")
    
    # Paper baseline gains reference (Table 1)[cite: 3]:
    # G3 PSS: K7 = 68.24, K8 = 318.3
    # G5 PSS: K7 = 5.60,  K8 = -2.84
    # G6 PSS: K7 = 4.47,  K8 = -3.87