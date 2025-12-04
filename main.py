import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

# --- Theoretical fidelities ---

def unencoded_fidelity_theoretical(epsilon):
    return np.cos(epsilon/2)**2  # Correct Rx rotation fidelity for |+>

def encoded_fidelity_no_error_theoretical(epsilon):
    c = np.cos(epsilon/2)
    s = np.sin(epsilon/2)
    return c**6 / (c**6 + s**6 + 1e-12)

def encoded_fidelity_error_detected_theoretical(epsilon):
    return np.cos(epsilon/2)**2

# --- Simulation functions ---

def simulate_unencoded_fidelity(epsilon_range):
    fidelities = []
    psi_plus = Statevector.from_label('+')
    for eps in epsilon_range:
        rx = Operator([[np.cos(eps/2), -1j*np.sin(eps/2)],
                       [-1j*np.sin(eps/2), np.cos(eps/2)]])
        state_after = psi_plus.evolve(rx)
        fidelities.append(np.abs(psi_plus.inner(state_after))**2)
    return fidelities

def simulate_encoded_fidelity(epsilon_range):
    fid_no_error = []
    fid_error_detected = []

    ideal_state = Statevector((Statevector.from_label('000').data +
                               Statevector.from_label('111').data)/np.sqrt(2))

    for eps in epsilon_range:
        # Rx coherent error on each qubit
        rx = Operator([[np.cos(eps/2), -1j*np.sin(eps/2)],
                       [-1j*np.sin(eps/2), np.cos(eps/2)]])
        error_op_3q = rx.tensor(rx).tensor(rx)

        # Encode |+> -> (|000> + |111>)/√2
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        state_encoded = Statevector(qc)

        # Apply error
        state_after_error = state_encoded.evolve(error_op_3q)

        # --- No error detected (|000> + |111>) ---
        proj_no_error = np.zeros(8, dtype=complex)
        proj_no_error[0] = state_after_error.data[0]
        proj_no_error[7] = state_after_error.data[7]
        norm = np.linalg.norm(proj_no_error)
        if norm < 1e-12:
            fid_no_error.append(1.0)
        else:
            proj_no_error = Statevector(proj_no_error / norm)
            fid_no_error.append(np.abs(ideal_state.inner(proj_no_error))**2)

        # --- Single error detected: one-qubit flip states ---
        single_error_indices = [1, 2, 4]  # |001>, |010>, |100>
        f_err_list = []
        for idx in single_error_indices:
            if np.isclose(np.abs(state_after_error.data[idx]), 0):
                f_err_list.append(1.0)
            else:
                proj = np.zeros(8, dtype=complex)
                proj[idx] = state_after_error.data[idx]
                proj_state = Statevector(proj / np.linalg.norm(proj))
                # Apply correction X on corresponding qubit
                qc_corr = QuantumCircuit(3)
                qc_corr.x(int(np.log2(idx)))
                proj_state = proj_state.evolve(Operator(qc_corr))
                f_err_list.append(np.abs(ideal_state.inner(proj_state))**2)
        fid_error_detected.append(np.mean(f_err_list))

    return fid_no_error, fid_error_detected

# --- Main Execution ---

if __name__ == '__main__':
    epsilon_range = np.linspace(0, 0.5, 25)

    # Unencoded
    unencoded_sim = simulate_unencoded_fidelity(epsilon_range)
    unencoded_th = unencoded_fidelity_theoretical(epsilon_range)

    # Encoded
    enc_no_err_sim, enc_err_sim = simulate_encoded_fidelity(epsilon_range)
    enc_no_err_th = encoded_fidelity_no_error_theoretical(epsilon_range)
    enc_err_th = encoded_fidelity_error_detected_theoretical(epsilon_range)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Unencoded
    axs[0].plot(epsilon_range, unencoded_th, 'r-', label='Unencoded (Theoretical)')
    axs[0].plot(epsilon_range, unencoded_sim, 'ro', markersize=5, label='Unencoded (Simulated)')
    axs[0].set_title('Unencoded Qubit Fidelity', fontsize=14)
    axs[0].set_xlabel(r'Error Rotation Angle $\epsilon$', fontsize=12)
    axs[0].set_ylabel('Fidelity', fontsize=12)
    axs[0].legend(fontsize=10)
    axs[0].grid(True)

    # Encoded
    axs[1].plot(epsilon_range, enc_no_err_th, 'g-', label='No Error Detected (Theoretical)')
    axs[1].plot(epsilon_range, enc_no_err_sim, 'go', markersize=5, label='No Error Detected (Simulated)')
    axs[1].plot(epsilon_range, enc_err_th, 'm-', label='Error Detected (Theoretical)')
    axs[1].plot(epsilon_range, enc_err_sim, 'mo', markersize=5, label='Error Detected (Simulated)')
    axs[1].set_title('Encoded 3-Qubit Bit-Flip Code Fidelity', fontsize=14)
    axs[1].set_xlabel(r'Error Rotation Angle $\epsilon$', fontsize=12)
    axs[1].set_ylabel('Fidelity', fontsize=12)
    axs[1].legend(fontsize=10)
    axs[1].grid(True)

    plt.suptitle('Qubit Fidelity under Coherent X Error', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
