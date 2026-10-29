import math
import random
import numpy as np
import matplotlib.pyplot as plt


class CFMTSP_ACO:
    def __init__(self, adj_matrix, node_coords, speed_options, t_occ, t_op=0.0):
        """
        CFMTSP ACO Implementation using Augmented Graph Representation.
        
        :param adj_matrix: Adjacency matrix B (Nv x Nv)
        :param node_coords: Coordinates of nodes [(x1, y1), (x2, y2), ...]
        :param speed_options: Available speed choices S = [s1, s2, ..., sNs]
        :param t_occ: Minimum allowable arrival time difference (occupancy time)
        :param t_op: Operational time at nodes
        """
        self.B = np.array(adj_matrix)
        self.coords = np.array(node_coords)
        self.Nv = len(adj_matrix)
        self.S = speed_options
        self.Ns = len(speed_options)
        self.t_occ = t_occ
        self.t_op = t_op

        # Compute Euclidean Distance Matrix L_ij
        self.L = np.zeros((self.Nv, self.Nv))
        for i in range(self.Nv):
            for j in range(self.Nv):
                if self.B[i, j] == 1:
                    self.L[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])

        # 1. Build Edge Matrix (eB) [Algorithm 1]
        self.eB, self.edges = self._build_edge_matrix()
        self.Ne = len(self.edges)

        # 2. Build Trajectory Adjacency Matrix (psi_B) [Equation 7]
        self.num_trajectories = self.Ne * self.Ns
        self.psi_map = {}
        for q in range(1, self.Ne + 1):
            for l_idx in range(1, self.Ns + 1):
                p = (q - 1) * self.Ns + l_idx
                self.psi_map[p] = (q, l_idx)

        # 3. Build Augmented Edge Adjacency Matrix (xi_B) [Algorithm 2]
        self.xi_B, self.augmented_edges = self._build_augmented_edge_matrix()
        self.N_xi = len(self.augmented_edges)

    def _build_edge_matrix(self):
        """Algorithm 1: Determine the index of edge matrix"""
        eB = np.zeros((self.Nv, self.Nv), dtype=int)
        edges = {}
        q = 0
        for i in range(self.Nv):
            for j in range(self.Nv):
                if i != j and self.B[i, j] == 1:
                    q += 1
                    eB[i, j] = q
                    edges[q] = (i, j)
        return eB, edges

    def _build_augmented_edge_matrix(self):
        """Algorithm 2: Determining the index of augmented edges"""
        xi_B = np.zeros((self.num_trajectories + 1, self.num_trajectories + 1), dtype=int)
        augmented_edges = {}
        h = 0

        for p1 in range(1, self.num_trajectories + 1):
            q1, l1 = self.psi_map[p1]
            start_node_1, end_node_1 = self.edges[q1]

            for p2 in range(1, self.num_trajectories + 1):
                q2, l2 = self.psi_map[p2]
                start_node_2, end_node_2 = self.edges[q2]

                # Augmented edge exists if successor node matches and edges are distinct
                if q1 != q2 and end_node_1 == start_node_2:
                    h += 1
                    xi_B[p1, p2] = h
                    
                    # Compute uniform acceleration and travel time along augmented edge [Equations 2 & 3]
                    s_i = self.S[l1 - 1]
                    s_j = self.S[l2 - 1]
                    dist = self.L[start_node_2, end_node_2]
                    
                    a = (s_j**2 - s_i**2) / (2.0 * dist) if dist > 0 else 0
                    if abs(a) < 1e-9:
                        t = dist / s_i
                    else:
                        t = (-s_i + math.sqrt(max(0, s_i**2 + 2 * a * dist))) / a

                    augmented_edges[h] = {
                        'p1': p1, 'p2': p2,
                        'edge1': q1, 'edge2': q2,
                        'from_node': start_node_1,
                        'via_node': end_node_1,
                        'to_node': end_node_2,
                        's_start': s_i, 's_end': s_j,
                        'accel': a, 'travel_time': t
                    }

        return xi_B, augmented_edges

    def _is_collided(self, node_vis_info, target_node, arrival_time):
        """Definition 6 & IsCollided procedure check"""
        if target_node not in node_vis_info:
            return False
        
        for k_other, t_arr in node_vis_info[target_node]:
            if abs(arrival_time - t_arr) <= (self.t_occ + self.t_op):
                return True
        return False

    def solve(self, start_nodes, start_speeds, num_iterations=500, alpha=1.0, beta=1.0, rho=0.95, Q=100.0):
        """
        Main ACO Algorithm for CFMTSP.
        """
        Nu = len(start_nodes)  # Number of vehicle species
        
        # Initialize pheromone trails per species on augmented edges
        tau = {k: np.full(self.N_xi + 1, 1e-4) for k in range(1, Nu + 1)}

        best_global_max_time = float('inf')
        best_trajectories = None
        history = []

        for iteration in range(1, num_iterations + 1):
            species_paths = {}
            node_vis_info = {}
            all_species_success = True

            for k in range(1, Nu + 1):
                curr_node = start_nodes[k - 1]
                curr_speed = start_speeds[k - 1]
                
                unvisited = set(range(self.Nv)) - {curr_node}
                curr_time = 0.0

                if curr_node not in node_vis_info:
                    node_vis_info[curr_node] = []
                node_vis_info[curr_node].append((k, curr_time))

                selected_augmented_edges = []
                prev_p = None

                while unvisited:
                    candidate_edges = [q for q, (u, v) in self.edges.items() if u == curr_node and v in unvisited]
                    
                    valid_aug_choices = []
                    probabilities = []

                    for q in candidate_edges:
                        next_node = self.edges[q][1]

                        for l_idx in range(1, self.Ns + 1):
                            s_next = self.S[l_idx - 1]
                            p_cand = (q - 1) * self.Ns + l_idx

                            if prev_p is None:
                                dist = self.L[curr_node, next_node]
                                a = (s_next**2 - curr_speed**2) / (2.0 * dist) if dist > 0 else 0
                                travel_t = dist / curr_speed if abs(a) < 1e-9 else (-curr_speed + math.sqrt(max(0, curr_speed**2 + 2 * a * dist))) / a
                                arr_time = curr_time + travel_t

                                if not self._is_collided(node_vis_info, next_node, arr_time):
                                    valid_aug_choices.append({
                                        'h': None, 'p_next': p_cand, 'from_node': curr_node, 'next_node': next_node,
                                        'travel_time': travel_t, 'dist': dist, 'speed': s_next
                                    })
                                    probabilities.append(1.0)
                            else:
                                h_candidate = self.xi_B[prev_p, p_cand]
                                if h_candidate > 0:
                                    aug_info = self.augmented_edges[h_candidate]
                                    travel_t = aug_info['travel_time']
                                    arr_time = curr_time + travel_t

                                    if not self._is_collided(node_vis_info, next_node, arr_time):
                                        eta = 1.0 / (travel_t + 1e-6)
                                        phero = tau[k][h_candidate]

                                        valid_aug_choices.append({
                                            'h': h_candidate, 'p_next': p_cand, 'from_node': curr_node, 'next_node': next_node,
                                            'travel_time': travel_t, 'dist': self.L[curr_node, next_node], 'speed': s_next
                                        })
                                        # Probability formula [Equation 11]
                                        probabilities.append((eta ** beta) * (phero ** alpha))

                    if not valid_aug_choices:
                        all_species_success = False
                        break

                    prob_sum = sum(probabilities)
                    if prob_sum > 0:
                        norm_probs = [p / prob_sum for p in probabilities]
                        chosen_idx = np.random.choice(len(valid_aug_choices), p=norm_probs)
                    else:
                        chosen_idx = random.randint(0, len(valid_aug_choices) - 1)

                    choice = valid_aug_choices[chosen_idx]
                    
                    curr_node = choice['next_node']
                    curr_time += choice['travel_time']
                    prev_p = choice['p_next']
                    
                    unvisited.remove(curr_node)
                    selected_augmented_edges.append(choice)

                    if curr_node not in node_vis_info:
                        node_vis_info[curr_node] = []
                    node_vis_info[curr_node].append((k, curr_time))

                if not all_species_success:
                    break
                
                species_paths[k] = {
                    'choices': selected_augmented_edges,
                    'total_time': curr_time,
                    'total_dist': sum(c['dist'] for c in selected_augmented_edges)
                }

            if all_species_success:
                # Calculate completion time of slowest vehicle [Equation 1]
                t_max = max(species_paths[k]['total_time'] for k in range(1, Nu + 1))

                if t_max < best_global_max_time:
                    best_global_max_time = t_max
                    best_trajectories = species_paths

                # Evaporation and Pheromone Deposit [Equations 8 & 10]
                for k in range(1, Nu + 1):
                    delta_tau = Q / species_paths[k]['total_dist']
                    for choice in species_paths[k]['choices']:
                        h = choice['h']
                        if h is not None:
                            tau[k][h] = tau[k][h] + delta_tau
            else:
                # Evaporate pheromones on search failure [Equation 9]
                for k in range(1, Nu + 1):
                    tau[k] *= rho

            history.append(best_global_max_time if best_global_max_time != float('inf') else None)

        return best_global_max_time, best_trajectories, history


# ==========================================
# Script Execution & Graph Visualization
# ==========================================
if __name__ == "__main__":
    # 1. Coordinates for 10-node network (Figure 3 in paper)
    coords = [
        (150, -123),  # Node 0
        (-150, 50),   # Node 1
        (110, -30),   # Node 2
        (70, 150),    # Node 3
        (-50, -100),  # Node 4
        (5, 5),       # Node 5
        (15, -10),    # Node 6
        (75, -75),    # Node 7
        (-80, 160),   # Node 8
        (-10, -35)    # Node 9
    ]

    # 2. Adjacency Matrix B for CONFIG 10-1 (Section 5.2.1)
    B_10_1 = [
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # 0
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1],  # 1
        [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],  # 2
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # 3
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 4
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 1],  # 5
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],  # 6
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 7
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],  # 8
        [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]   # 9
    ]

    # Speed Options S (Section 5)
    speed_options = [0.1, 0.5, 1.0, 1.5]
    t_occ = 10.0  # Minimum allowable arrival time difference (s)

    solver = CFMTSP_ACO(
        adj_matrix=B_10_1,
        node_coords=coords,
        speed_options=speed_options,
        t_occ=t_occ
    )

    # 3 Vehicles starting from Nodes 4, 1, and 3 (Section 5)
    start_nodes = [4, 1, 3]
    start_speeds = [0.5, 0.5, 0.5]
    num_iterations = 500

    print("Running ACO for CFMTSP on CONFIG 10-1...")
    best_time, best_trajectories, history = solver.solve(
        start_nodes=start_nodes,
        start_speeds=start_speeds,
        num_iterations=num_iterations,
        alpha=1.0,
        beta=1.0,
        rho=0.95
    )

    # Visualizing Results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graph Plot 1: Node Network and Vehicle Routes
    ax1.set_title("CFMTSP Vehicle Trajectories (CONFIG 10-1)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True)

    # Draw Graph Edges
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if B_10_1[i][j] == 1:
                ax1.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], 'k--', alpha=0.3)

    # Plot Nodes
    coords_arr = np.array(coords)
    ax1.scatter(coords_arr[:, 0], coords_arr[:, 1], c='red', s=100, zorder=5)
    for idx, (x, y) in enumerate(coords):
        ax1.annotate(f"Node {idx}", (x + 3, y + 3), fontsize=9)

    # Draw Vehicle Trajectories
    colors = ['blue', 'green', 'orange']
    if best_trajectories:
        for k, data in best_trajectories.items():
            start_node = start_nodes[k - 1]
            route = [start_node] + [c['next_node'] for c in data['choices']]
            
            rx = [coords[n][0] for n in route]
            ry = [coords[n][1] for n in route]
            ax1.plot(rx, ry, color=colors[k - 1], linewidth=2.5, label=f"Vehicle {k} (t={data['total_time']:.1f}s)")

    ax1.legend()

    # Graph Plot 2: Convergence Plot (Min of Max Times)
    ax2.set_title("ACO Optimization Convergence Curve")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Completion Time of Slowest Vehicle (s)")
    ax2.grid(True)
    ax2.plot(range(1, num_iterations + 1), history, color='purple', linewidth=2)

    plt.tight_layout()
    plt.savefig('cfmtsp_results.png')
    print("Optimization Complete! Plot saved as 'cfmtsp_results.png'")
    plt.show()