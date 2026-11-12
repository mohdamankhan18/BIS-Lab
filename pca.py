import time
import threading
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. THREAD-SAFE READ-WRITE LOCK (Section 3.2: Concurrent Shared-Memory Access)
# =============================================================================
class ReadWriteLock:
    """
    Allows concurrent reads from multiple threads (neighborhood selection & crossover)
    while serializing writes (individual replacement), matching the POSIX rwlock
    described in Section 3.2.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._read_ready = threading.Condition(self._lock)
        self._readers = 0

    def acquire_read(self):
        with self._lock:
            self._readers += 1

    def release_read(self):
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        self._lock.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._lock.release()


# =============================================================================
# 2. BENCHMARK INSTANCE GENERATOR (Braun et al. ETC Model - Section 4.1)
# =============================================================================
def generate_etc_matrix(nb_tasks=512, nb_machines=16, consistency="inconsistent",
                        task_het="high", machine_het="high", seed=42):
    """
    Generates an Expected Time to Compute (ETC) matrix following Braun et al. [9]
    and Ali et al. [17] as described in Section 4.1 (e.g., u_i_hihi, u_c_hihi).
    """
    rng = np.random.default_rng(seed)
    tau_task = 1000.0 if task_het == "high" else 100.0
    tau_mach = 1000.0 if machine_het == "high" else 10.0

    # Baseline task workloads
    task_base = rng.uniform(1.0, tau_task, size=(nb_tasks, 1))
    # Machine scaling factors
    mach_scale = rng.uniform(1.0, tau_mach, size=(nb_tasks, nb_machines))
    etc = task_base * mach_scale

    if consistency == "consistent":
        etc = np.sort(etc, axis=1)
    elif consistency == "semi-consistent":
        # Half of the machines form a consistent sub-matrix
        half = nb_machines // 2
        etc[:, :half] = np.sort(etc[:, :half], axis=1)

    return etc.astype(np.float64)


# =============================================================================
# 3. INDIVIDUAL REPRESENTATION & MIN-MIN HEURISTIC (Sections 3.3 & 4.1)
# =============================================================================
class Individual:
    """
    Solution representation (Fig. 3):
    - S: array of size nb_tasks where S[t] = m (task t assigned to machine m)
    - CT: array of size nb_machines tracking completion time of each machine
    """
    def __init__(self, schedule, etc_T, ready_times):
        self.S = np.array(schedule, dtype=np.int32)
        self.CT = self._compute_completion_times(self.S, etc_T, ready_times)
        self.makespan = float(np.max(self.CT))

    @staticmethod
    def _compute_completion_times(S, etc_T, ready_times):
        nb_machines = etc_T.shape[0]
        ct = ready_times.copy()
        for t, m in enumerate(S):
            ct[m] += etc_T[m, t]
        return ct

    def copy(self):
        clone = Individual.__new__(Individual)
        clone.S = self.S.copy()
        clone.CT = self.CT.copy()
        clone.makespan = self.makespan
        return clone


def min_min_heuristic(etc_T, ready_times):
    """
    Min-Min heuristic (Section 4.1 [16]) used to seed 1 individual in the population.
    Iteratively assigns the task with the minimum earliest completion time.
    """
    nb_machines, nb_tasks = etc_T.shape
    ct = ready_times.copy()
    schedule = np.full(nb_tasks, -1, dtype=np.int32)
    unassigned = set(range(nb_tasks))

    while unassigned:
        best_task = -1
        best_machine = -1
        min_completion = float("inf")

        for t in unassigned:
            completion_times = ct + etc_T[:, t]
            m = int(np.argmin(completion_times))
            if completion_times[m] < min_completion:
                min_completion = completion_times[m]
                best_task = t
                best_machine = m

        schedule[best_task] = best_machine
        ct[best_machine] += etc_T[best_machine, best_task]
        unassigned.remove(best_task)

    return schedule


# =============================================================================
# 4. PARALLEL ASYNCHRONOUS CELLULAR GENETIC ALGORITHM (Algorithms 2, 3, & 4)
# =============================================================================
class ParallelAsynchronousCGA:
    def __init__(self, etc_matrix, ready_times=None, grid_rows=16, grid_cols=16,
                 num_threads=3, crossover_type="tpx", p_comb=1.0, p_mut=1.0,
                 p_ls=1.0, ls_iter=10, max_time_sec=5.0, seed=42):
        # Store transposed ETC matrix (nb_machines x nb_tasks) for cache efficiency (Sec 3.3)
        self.etc_T = np.ascontiguousarray(etc_matrix.T, dtype=np.float64)
        self.nb_machines, self.nb_tasks = self.etc_T.shape
        self.ready_times = (np.zeros(self.nb_machines, dtype=np.float64)
                            if ready_times is None else np.array(ready_times, dtype=np.float64))

        # Grid & Parallelism Parameters (Table 1)
        self.rows = grid_rows
        self.cols = grid_cols
        self.pop_size = grid_rows * grid_cols
        self.num_threads = num_threads

        # Evolutionary & Local Search Parameters (Table 1)
        self.crossover_type = crossover_type  # 'tpx' (two-point) or 'opx' (one-point)
        self.p_comb = p_comb
        self.p_mut = p_mut
        self.p_ls = p_ls
        self.ls_iter = ls_iter
        self.max_time_sec = max_time_sec
        self.rng_seed = seed

        # Shared population grid & per-cell Read-Write locks
        self.population = [None] * self.pop_size
        self.locks = [ReadWriteLock() for _ in range(self.pop_size)]

        # Tracking metrics
        self.eval_counts = [0] * self.num_threads
        self.history = []
        self.history_lock = threading.Lock()

    def _setup_population(self):
        """
        Initializes 16x16 population: 1 individual via Min-Min heuristic,
        remaining 255 individuals randomly (Section 4.1).
        """
        rng = np.random.default_rng(self.rng_seed)
        min_min_sched = min_min_heuristic(self.etc_T, self.ready_times)
        self.population[0] = Individual(min_min_sched, self.etc_T, self.ready_times)

        for idx in range(1, self.pop_size):
            rand_sched = rng.integers(0, self.nb_machines, size=self.nb_tasks, dtype=np.int32)
            self.population[idx] = Individual(rand_sched, self.etc_T, self.ready_times)

    def _get_l5_neighborhood(self, idx):
        """
        Returns indices of the Linear 5 (Von Neumann) neighborhood on a 2D toroidal grid:
        center cell + North, South, East, West neighbors (Section 4.1).
        """
        r, c = divmod(idx, self.cols)
        north = ((r - 1) % self.rows) * self.cols + c
        south = ((r + 1) % self.rows) * self.cols + c
        west = r * self.cols + ((c - 1) % self.cols)
        east = r * self.cols + ((c + 1) % self.cols)
        return [idx, north, south, west, east]

    def _recombine(self, parent1, parent2, rng):
        """
        One-point (opx) or two-point (tpx) crossover (Section 4.1).
        """
        child_S = parent1.S.copy()
        if rng.random() < self.p_comb:
            if self.crossover_type == "opx":
                pt = int(rng.integers(1, self.nb_tasks))
                child_S[pt:] = parent2.S[pt:]
            else:  # two-point crossover ('tpx')
                pt1, pt2 = sorted(rng.choice(self.nb_tasks, size=2, replace=False))
                child_S[pt1:pt2] = parent2.S[pt1:pt2]
        return Individual(child_S, self.etc_T, self.ready_times)

    def _mutate(self, ind, rng):
        """
        Move mutation: moves one randomly chosen task to a randomly chosen machine,
        incrementally updating completion times CT (Section 3.3 & 4.1).
        """
        if rng.random() < self.p_mut:
            task = int(rng.integers(0, self.nb_tasks))
            old_mac = ind.S[task]
            new_mac = int(rng.integers(0, self.nb_machines))
            if new_mac != old_mac:
                ind.S[task] = new_mac
                ind.CT[old_mac] -= self.etc_T[old_mac, task]
                ind.CT[new_mac] += self.etc_T[new_mac, task]
                ind.makespan = float(np.max(ind.CT))

    def _h2ll_local_search(self, ind, rng):
        """
        Algorithm 4: H2LL Local Search Operator (Section 3.3).
        Moves a randomly chosen task from the most loaded machine (makespan)
        to the best candidate among the least loaded machines.
        """
        if rng.random() >= self.p_ls:
            return

        num_candidates = max(1, self.nb_machines // 2)

        for _ in range(self.ls_iter):
            # Step 2: Sort machines in ascending order of completion time
            sorted_macs = np.argsort(ind.CT)
            most_loaded_mac = int(sorted_macs[-1])

            # Step 3: Pick a random task assigned to the most loaded machine
            assigned_tasks = np.where(ind.S == most_loaded_mac)[0]
            if len(assigned_tasks) == 0:
                break
            task = int(rng.choice(assigned_tasks))

            # Step 4: Current makespan acts as threshold score
            best_score = ind.CT[most_loaded_mac]
            best_mac = -1

            # Steps 5-11: Evaluate least-loaded candidate machines
            for mac in sorted_macs[:num_candidates]:
                mac = int(mac)
                if mac == most_loaded_mac:
                    continue
                new_score = ind.CT[mac] + self.etc_T[mac, task]
                if new_score < best_score:
                    best_mac = mac
                    best_score = new_score

            # Step 12: Move task to best_mac if an improving move was found
            if best_mac != -1:
                ind.S[task] = best_mac
                ind.CT[most_loaded_mac] -= self.etc_T[most_loaded_mac, task]
                ind.CT[best_mac] += self.etc_T[best_mac, task]
                ind.makespan = float(np.max(ind.CT))

    def _evolve_worker(self, thread_id, start_idx, end_idx, t0):
        """
        Algorithm 3: Asynchronous evolution loop executed independently by each thread
        over its contiguous block of cells (Section 3.2, Fig. 2).
        """
        rng = np.random.default_rng(self.rng_seed + thread_id + 100)
        local_evals = 0
        generation = 0

        while (time.perf_counter() - t0) < self.max_time_sec:
            # Fixed line sweep across the thread's assigned block
            for idx in range(start_idx, end_idx):
                neigh_indices = self._get_l5_neighborhood(idx)

                # Acquire read locks on unique neighbors in sorted order to prevent deadlocks
                unique_neigh = sorted(set(neigh_indices))
                for n_idx in unique_neigh:
                    self.locks[n_idx].acquire_read()
                try:
                    # Select best 2 neighbors as parents (Table 1)
                    candidates = [self.population[n_idx] for n_idx in neigh_indices]
                    candidates.sort(key=lambda ind: ind.makespan)
                    parent1 = candidates[0].copy()
                    parent2 = candidates[1].copy()
                finally:
                    for n_idx in reversed(unique_neigh):
                        self.locks[n_idx].release_read()

                # Variation & Local Search (executed outside lock for maximum parallelism)
                offspring = self._recombine(parent1, parent2, rng)
                self._mutate(offspring, rng)
                self._h2ll_local_search(offspring, rng)
                local_evals += 1

                # Replace current individual if offspring improves fitness (makespan)
                self.locks[idx].acquire_write()
                try:
                    if offspring.makespan < self.population[idx].makespan:
                        self.population[idx] = offspring
                finally:
                    self.locks[idx].release_write()

            generation += 1
            if thread_id == 0 and generation % 5 == 0:
                elapsed = time.perf_counter() - t0
                best_now = min(ind.makespan for ind in self.population if ind is not None)
                with self.history_lock:
                    self.history.append((elapsed, best_now))

        self.eval_counts[thread_id] = local_evals

    def run(self):
        """
        Algorithm 2: Master execution routine for PA-CGA.
        """
        t0 = time.perf_counter()
        self._setup_population()
        initial_best = min(ind.makespan for ind in self.population)

        # Partition 16x16 population into contiguous blocks across threads (Fig. 2)
        block_size = self.pop_size // self.num_threads
        threads = []
        for tid in range(self.num_threads):
            s_idx = tid * block_size
            e_idx = self.pop_size if tid == self.num_threads - 1 else (tid + 1) * block_size
            t = threading.Thread(target=self._evolve_worker, args=(tid, s_idx, e_idx, t0))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        elapsed = time.perf_counter() - t0
        best_ind = min(self.population, key=lambda ind: ind.makespan)
        mean_makespan = float(np.mean([ind.makespan for ind in self.population]))
        total_evals = sum(self.eval_counts)

        return {
            "initial_min_min_makespan": initial_best,
            "final_best_makespan": best_ind.makespan,
            "final_mean_makespan": mean_makespan,
            "best_schedule": best_ind.S,
            "machine_loads": best_ind.CT,
            "total_evaluations": total_evals,
            "elapsed_sec": elapsed,
            "history": self.history
        }


# =============================================================================
# 5. EXECUTION & DEMONSTRATION ON BENCHMARK INSTANCE (u_i_hihi)
# =============================================================================
if __name__ == "__main__":
    # Create standard benchmark instance: 512 tasks x 16 heterogeneous machines
    etc_matrix = generate_etc_matrix(
        nb_tasks=512,
        nb_machines=16,
        consistency="inconsistent",
        task_het="high",
        machine_het="high",
        seed=42
    )

    print("Running PA-CGA (3 Threads, Two-Point Crossover 'tpx', H2LL iter=10)...")
    pacga = ParallelAsynchronousCGA(
        etc_matrix=etc_matrix,
        grid_rows=16,
        grid_cols=16,
        num_threads=3,          # Optimal thread count reported in Section 4.2
        crossover_type="tpx",   # Best crossover operator reported in Section 4.2
        p_comb=1.0,
        p_mut=1.0,
        p_ls=1.0,
        ls_iter=10,             # Best H2LL iteration depth reported in Section 4.2
        max_time_sec=5.0,       # Wall-clock stopping criterion (set to 10s or 90s for full runs)
        seed=42
    )

    results = pacga.run()

    print("\n=== PA-CGA RESOURCE ALLOCATION RESULTS ===")
    print(f"Initial Min-Min Heuristic Makespan : {results['initial_min_min_makespan']:,.2f}")
    print(f"Final Best PA-CGA Makespan         : {results['final_best_makespan']:,.2f}")
    print(f"Final Population Mean Makespan     : {results['final_mean_makespan']:,.2f}")
    improvement = ((results['initial_min_min_makespan'] - results['final_best_makespan'])
                   / results['initial_min_min_makespan']) * 100
    print(f"Makespan Reduction over Min-Min    : {improvement:.2f}%")
    print(f"Total Offspring Evaluations        : {results['total_evaluations']:,}")
    print(f"Elapsed Wall Time                  : {results['elapsed_sec']:.2f} s")

    # Visualize convergence and final resource load balance across the 16 machines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    if results["history"]:
        times, bests = zip(*results["history"])
        ax1.plot(times, bests, color="#1f77b4", linewidth=2)
        ax1.set_title("PA-CGA Makespan Convergence")
        ax1.set_xlabel("Wall-Clock Time (seconds)")
        ax1.set_ylabel("Best Makespan")
        ax1.grid(True, linestyle="--", alpha=0.6)

    machines = np.arange(1, len(results["machine_loads"]) + 1)
    ax2.bar(machines, results["machine_loads"], color="#2ca02c", alpha=0.85)
    ax2.axhline(results["final_best_makespan"], color="red", linestyle="--", label="Makespan (Max Load)")
    ax2.set_title("Final Resource Load Across 16 Machines")
    ax2.set_xlabel("Machine ID")
    ax2.set_ylabel("Completion Time (CT[m])")
    ax2.set_xticks(machines)
    ax2.legend()
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()