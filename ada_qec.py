import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt

# ==============================================================================
# 1. PARAMETERS AND QEC MODEL DEFINITION
# ==============================================================================

# --- Simulation and Cost Parameters ---
T_HORIZON = 5000    # Time horizon (number of logical qubits to transmit)
WINDOW_SIZE = 10    # Window size for the heuristic/MLE (W)
N_TRIALS = 50       # Number of independent trials to average results (reduces noise)

# --- Parameter Sweep Combinations ---
# Define combinations of (True_P, Lambda) to demonstrate behavior
PARAM_COMBINATIONS = [
    (0.02, 0.05), # Low Noise (Favors 5-qubit code)
    (0.05, 0.05), # Medium Noise (Favors 7-qubit code)
    (0.10, 0.05), # High Noise (Favors 9-qubit code)
    (0.05, 0.1)  # Medium Noise, Very High Cost Lambda (Favors 5-qubit/not robust)
]

# --- QEC Code Definitions (Arms) ---
QEC_CODES = {
    5: {'qubits': 5, 'name': '5-qubit'},
    7: {'qubits': 7, 'name': '7-qubit'},
    9: {'qubits': 9, 'name': '9-qubit'}
}
CODE_LIST = list(QEC_CODES.keys())


# --- QEC Error Rate Function ---
def logical_error_rate(p, code_size):
    """Approximation of the logical error rate L_k(p) for d=3 codes."""
    if code_size == 5:
        A = 100.0  # Least robust
    elif code_size == 7:
        A = 50.0   # Medium robustness
    else: # code_size == 9
        A = 25.0   # Most robust
        
    L_k = A * (p**2) 
    return min(1.0, max(0.0, L_k))

def expected_reward(p, code_size, cost_weight):
    """Calculates the expected reward mu_k(p) = F_k(p) - lambda * C_k."""
    L_k = logical_error_rate(p, code_size)
    F_k = 1.0 - L_k
    C_k = QEC_CODES[code_size]['qubits']
    
    return F_k - cost_weight * C_k

# --- Oracle Calculation (Best fixed code in hindsight) ---
def find_oracle(p_true, lambda_cost):
    """Finds the maximum possible expected reward and the corresponding code."""
    max_reward = -np.inf
    best_code = None
    
    for k in QEC_CODES.keys():
        mu_k = expected_reward(p_true, k, lambda_cost)
        if mu_k > max_reward:
            max_reward = mu_k
            best_code = k
            
    return max_reward, best_code


# ==============================================================================
# 2. ADAPTIVE QEC POLICY CLASS (MAB Implementation)
# ==============================================================================

class AdaptiveQECPolicy:
    def __init__(self, p_true, lambda_cost, T, window_size):
        self.p_true = p_true
        self.lambda_cost = lambda_cost
        self.T = T
        self.window_size = window_size
        
        # History
        self.history = []  # Stores (k, logical_error)
        self.N = {k: 0 for k in CODE_LIST} # Pull count
        self.S = {k: 0 for k in CODE_LIST} # Sum of rewards
        self.regret_history = []
        self.total_phys_qubits = 0 # New counter for physical qubits
        
        # Get Oracle for regret calculation
        self.ORACLE_REWARD, _ = find_oracle(self.p_true, self.lambda_cost)

    def estimate_p_mle(self):
        """Estimates the physical error rate p using history (simplified MLE)."""
        if not self.history:
            return 0.5
            
        recent_history = self.history[-self.window_size:]
        if not recent_history:
            return 0.5
            
        logical_errors = [le for _, le in recent_history]
        mean_L_k = np.mean(logical_errors)
        A_avg = 58.33
        p_hat = sqrt(mean_L_k / A_avg) if mean_L_k > 0 else 0.0
        
        return min(0.5, p_hat) 

    def choose_code(self, method, t):
        """Chooses the next QEC code based on the strategy."""
        
        if method in CODE_LIST: # Fixed Code Policy
            return method 
            
        if t <= len(CODE_LIST) and method != 'Fixed':
            return CODE_LIST[t - 1] # Initial Exploration for adaptive methods
            
        if method == 'Greedy':
            p_hat = self.estimate_p_mle()
            best_code = None
            max_mu = -np.inf
            
            for k in CODE_LIST:
                mu_k = expected_reward(p_hat, k, self.lambda_cost)
                if mu_k > max_mu:
                    max_mu = mu_k
                    best_code = k
            return best_code
            
        elif method == 'UCB1':
            best_code = None
            max_ucb = -np.inf
            
            for k in CODE_LIST:
                mu_hat = self.S[k] / self.N[k]
                exploration_term = sqrt(2 * log(t) / self.N[k])
                ucb_value = mu_hat + exploration_term
                
                if ucb_value > max_ucb:
                    max_ucb = ucb_value
                    best_code = k
            return best_code
        
        raise ValueError("Unknown method")

    def simulate_step(self, method, t):
        """Simulates one logical qubit transmission."""
        k_t = self.choose_code(method, t)
        
        # True expected logical error rate
        L_k_p = logical_error_rate(self.p_true, k_t)
        
        # Observation (binary reward)
        logical_error = int(np.random.rand() < L_k_p) 
        
        # Compute fidelity and reward
        fidelity = 1.0 - logical_error
        cost = QEC_CODES[k_t]['qubits']
        r_t = fidelity - self.lambda_cost * cost
        
        # Update history and stats
        self.history.append((k_t, logical_error))
        self.N[k_t] += 1
        self.S[k_t] += r_t
        
        # Track total physical qubits used (NEW FEATURE)
        self.total_phys_qubits += cost

        # Calculate instantaneous regret
        mu_k_t = expected_reward(self.p_true, k_t, self.lambda_cost)
        inst_regret = self.ORACLE_REWARD - mu_k_t
        
        current_cumulative_regret = (self.regret_history[-1] if self.regret_history else 0) + inst_regret
        self.regret_history.append(current_cumulative_regret)
        
    def run_simulation(self, method):
        """Runs the simulation for the specified method."""
        # Reset stats
        self.history = []
        self.regret_history = []
        self.N = {k: 0 for k in CODE_LIST} 
        self.S = {k: 0 for k in CODE_LIST} 
        self.total_phys_qubits = 0 # Reset the counter

        # Initialize steps for adaptive policies
        if method not in CODE_LIST:
            for t in range(1, len(CODE_LIST) + 1):
                self.simulate_step(method, t)
        
        # Main loop
        start_t = len(CODE_LIST) + 1 if method not in CODE_LIST else 1
        for t in range(start_t, self.T + 1):
            self.simulate_step(method, t)
            
        # Return the new metric (total_phys_qubits)
        return np.array(self.regret_history), self.N, self.total_phys_qubits


# ==============================================================================
# 3. MAIN EXECUTION AND ANALYSIS
# ==============================================================================

if __name__ == "__main__":
    
    np.random.seed(42) 
    
    # Policies to test: Fixed codes (5, 7, 9), Greedy, UCB1
    POLICIES = CODE_LIST + ['Greedy', 'UCB1']
    POLICY_COLORS = {5: 'C0', 7: 'C1', 9: 'C2', 'Greedy': 'red', 'UCB1': 'blue'}
    ADAPTIVE_POLICIES = ['Greedy', 'UCB1']

    print("--- Adaptive QEC MAB Simulation Summary ---")
    print(f"Time Horizon (T): {T_HORIZON}, Trials: {N_TRIALS}")
    
    # Set up figures for the sweep
    num_combinations = len(PARAM_COMBINATIONS)
    
    # Plot 1: Cumulative Regret (Multiple Subplots)
    fig_regret, axs_regret = plt.subplots(2, num_combinations // 2, figsize=(16, 9))
    axs_regret = axs_regret.flatten()
    
    # Plot 2: Code Usage Distribution (Multiple Subplots)
    fig_usage, axs_usage = plt.subplots(2, num_combinations // 2, figsize=(16, 9))
    axs_usage = axs_usage.flatten()
    
    # Initialize Terminal Output Table
    terminal_output = []
    
    # Regret Table Header
    terminal_output.append("=====================================================================================")
    terminal_output.append("|                         Total Cumulative Regret (Adaptive vs. Fixed)                      |")
    terminal_output.append("=====================================================================================")
    terminal_output.append(f"| P | λ | Optimal | Oracle Reward | {' | '.join([f'{p:>11}' for p in POLICIES])} |")
    terminal_output.append("=====================================================================================")


    all_qubits_results = [] # To store the new qubit count data

    for i, (p_true, lambda_cost) in enumerate(PARAM_COMBINATIONS):
        
        # Calculate Oracle for this combination
        oracle_reward, oracle_code = find_oracle(p_true, lambda_cost)
        
        # Store results for all policies
        policy_results = {}
        qubit_results = {'P': p_true, 'λ': lambda_cost} # New dictionary for qubit counts
        
        for method in POLICIES:
            all_regret = []
            all_pulls = {k: 0 for k in CODE_LIST}
            all_qubit_counts = [] # New list for total qubits per trial
            
            # Run multiple trials
            for trial in range(N_TRIALS):
                np.random.seed(42 + trial) # Ensures new channel realization for each trial
                sim = AdaptiveQECPolicy(p_true, lambda_cost, T_HORIZON, WINDOW_SIZE)
                regret, pulls, phys_qubits = sim.run_simulation(method)
                
                all_regret.append(regret)
                all_qubit_counts.append(phys_qubits) # Store new metric
                
                for k in CODE_LIST:
                    all_pulls[k] += pulls[k]

            # Average Regret, Pulls, and Qubits
            avg_regret = np.mean(all_regret, axis=0)
            avg_pulls = {k: v / N_TRIALS for k, v in all_pulls.items()}
            avg_qubit_count = np.mean(all_qubit_counts) # Average new metric
            
            policy_results[method] = {
                'regret': avg_regret, 
                'final_regret': avg_regret[-1] if len(avg_regret)>0 else 0,
                'pulls': avg_pulls
            }
            
            if method in ADAPTIVE_POLICIES:
                qubit_results[method] = avg_qubit_count

            
            # Plot Regret Curve
            axs_regret[i].plot(range(1, T_HORIZON + 1), avg_regret, 
                               label=f'{method}', color=POLICY_COLORS[method], 
                               linestyle='--' if method in CODE_LIST else '-')

        all_qubits_results.append(qubit_results)

        # --- Terminal Output Row (Regret) ---
        regret_values = [f'{policy_results[p]["final_regret"]:^11.2f}' for p in POLICIES]
        terminal_output.append(f"|{p_true:<3}|{lambda_cost:<3}| {oracle_code}-qubit |   {oracle_reward:^8.4f}    | {' | '.join(regret_values)} |")

        # --- Plot 1: Cumulative Regret ---
        axs_regret[i].set_title(f'Cumulative Regret: $p$={p_true}, $\lambda$={lambda_cost} (Oracle: {oracle_code})')
        axs_regret[i].set_xlabel('Time (t, Logarithmic)')
        axs_regret[i].set_ylabel('Average Regret ($R_t$)')
        axs_regret[i].set_xscale('log')
        axs_regret[i].legend(loc='upper left', fontsize='small')
        axs_regret[i].grid(True, linestyle='--', alpha=0.5)

        # --- Plot 2: Code Usage Distribution ---
        usage_data = []
        usage_labels = []
        for method in POLICIES:
            # We only show usage for adaptive methods as fixed codes are trivial (100% on one code)
            if method not in CODE_LIST:
                total_pulls = sum(policy_results[method]['pulls'].values())
                if total_pulls == 0: continue
                usage_data.append([policy_results[method]['pulls'][k] / total_pulls for k in CODE_LIST])
                usage_labels.append(method)

        if usage_data:
            usage_matrix = np.array(usage_data).T
            bar_width = 0.25
            r = np.arange(len(usage_labels))
            
            for j, code in enumerate(CODE_LIST):
                axs_usage[i].bar(r + j * bar_width, usage_matrix[j], 
                                 width=bar_width, label=f'{code}-qubit', 
                                 color=POLICY_COLORS[code])

            axs_usage[i].set_title(f'Code Usage Distribution: $p$={p_true}, $\lambda$={lambda_cost}')
            axs_usage[i].set_xticks(r + bar_width)
            axs_usage[i].set_xticklabels(usage_labels)
            axs_usage[i].set_xlabel('Adaptive Policy')
            axs_usage[i].set_ylabel('Proportion of Total Pulls')
            axs_usage[i].legend(loc='upper right', fontsize='small')
            axs_usage[i].grid(axis='y', linestyle='--', alpha=0.5)
            
    # Final display cleanup
    fig_regret.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_usage.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- New Table: Total Physical Qubits ---
    
    # Calculate Fixed Code Qubit Counts (T_HORIZON * C_k)
    fixed_qubit_counts = {k: T_HORIZON * QEC_CODES[k]['qubits'] for k in CODE_LIST}
    
    # Add Fixed Code Qubit Counts to the results for display consistency
    for res in all_qubits_results:
        # Fixed policies always use T*C_k qubits
        res[5] = fixed_qubit_counts[5] 
        res[7] = fixed_qubit_counts[7] 
        res[9] = fixed_qubit_counts[9]
            
    terminal_output.append("\n")
    terminal_output.append("=====================================================================================")
    terminal_output.append("|               Total Physical Qubits Transmitted (Adaptive vs. Fixed)                |")
    terminal_output.append("=====================================================================================")
    terminal_output.append(f"| P | λ | Optimal | {' | '.join([f'{p:>11}' for p in POLICIES])} |")
    terminal_output.append("=====================================================================================")

    for res in all_qubits_results:
        # The Fixed columns (5, 7, 9) use the deterministic cost.
        # The Adaptive columns use the average cost from the simulation.
        qubit_values = [f'{res[p]:^11.0f}' if p in CODE_LIST else f'{res[p]:^11.0f}' for p in POLICIES]
        
        _, oracle_code = find_oracle(res['P'], res['λ'])
        terminal_output.append(f"|{res['P']:<3}|{res['λ']:<3}| {oracle_code}-qubit | {' | '.join(qubit_values)} |")

    terminal_output.append("=====================================================================================")

    print("\n".join(terminal_output))
    
    # In a real environment, this would save the plots:
    # fig_regret.savefig('cumulative_regret_comparison.png')
    # fig_usage.savefig('code_usage_distribution.png')

    # print("Plots saved: cumulative_regret_comparison.png and code_usage_distribution.png")