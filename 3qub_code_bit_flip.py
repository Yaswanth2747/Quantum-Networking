import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity

# ============================
# 3-Qubit Bit-Flip Code Simulation
# ============================

# ------------------ ENCODING + COHERENT ERROR ------------------
def create_3qubit_code_circuit(alpha, beta, epsilon):
    data = QuantumRegister(3, 'data')
    anc = QuantumRegister(2, 'ancilla')
    qc = QuantumCircuit(data, anc)

    # Encode logical qubit
    qc.initialize([alpha, beta], data[0])
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])

    # Apply coherent RX(2*epsilon) error to each data qubit
    for i in range(3):
        qc.rx(2 * epsilon, data[i])

    # Syndrome extraction (ancillas, no measurement)
    qc.cx(data[0], anc[0])
    qc.cx(data[2], anc[0])
    qc.cx(data[1], anc[1])
    qc.cx(data[2], anc[1])

    return qc

# ------------------ FIDELITY CALCULATION ------------------
def compute_fidelity_coherent_error(alpha, beta, epsilon):
    # Build circuit and simulate statevector
    qc = create_3qubit_code_circuit(alpha, beta, epsilon)
    full_state = Statevector(qc)

    # Trace out ancilla qubits (indices 3 and 4)
    rho_data = partial_trace(DensityMatrix(full_state), [3,4])

    # Decode (reverse encoding) using a unitary on 3 qubits
    decode_qc = QuantumCircuit(3)
    decode_qc.cx(0,2)
    decode_qc.cx(0,1)
    rho_decoded = rho_data.evolve(decode_qc)

    # Trace out qubits 1 and 2 to isolate logical qubit 0
    rho_logical = partial_trace(rho_decoded, [1,2])

    # Ideal logical qubit density matrix
    ideal_logical = DensityMatrix([[abs(alpha)**2, alpha*np.conj(beta)],
                                   [np.conj(alpha)*beta, abs(beta)**2]])

    # Compute fidelity
    fid = state_fidelity(ideal_logical, rho_logical)
    return np.real(fid)

# ------------------ UNENCODED FIDELITY ------------------
def compute_unencoded_fidelity(alpha, beta, epsilon):
    U = np.array([[np.cos(epsilon), -1j*np.sin(epsilon)],
                  [-1j*np.sin(epsilon), np.cos(epsilon)]])
    psi = np.array([alpha, beta])
    psi_err = U @ psi
    return np.abs(np.vdot(psi, psi_err))**2

# ------------------ SINGLE CASE ANALYSIS ------------------
def analyze_single_case():
    alpha = np.sqrt(0.7)
    beta = np.sqrt(0.3)
    eps = 0.2
    print(f"\nSingle-case analysis: ε = {eps}")
    enc = compute_fidelity_coherent_error(alpha, beta, eps)
    unenc = compute_unencoded_fidelity(alpha, beta, eps)
    print(f"Encoded fidelity   : {enc:.6f}")
    print(f"Unencoded fidelity : {unenc:.6f}")
    print(f"Improvement factor : {enc/unenc:.3f}")

# ------------------ SIMULATION OVER RANGE OF EPS ------------------
def simulate_coherent_errors():
    alpha = np.sqrt(0.7)
    beta = np.sqrt(0.3)
    eps_vals = np.linspace(0, 0.5, 15)

    enc_sim, unenc_sim, enc_theo, unenc_theo = [], [], [], []

    print("\nSimulating coherent errors across ε values...")
    for eps in eps_vals:
        print(f"  ε = {eps:.3f}")
        f_enc = compute_fidelity_coherent_error(alpha, beta, eps)
        f_unenc = compute_unencoded_fidelity(alpha, beta, eps)
        enc_sim.append(f_enc)
        unenc_sim.append(f_unenc)

        # Theoretical curves for comparison (stochastic approx)
        cos, sin = np.cos(eps), np.sin(eps)
        enc_theo.append((cos**6) / (cos**6 + sin**6))
        unenc_theo.append(cos**2)

    plt.figure(figsize=(10,6))
    plt.plot(eps_vals, enc_sim, 'bo-', label='Encoded (Simulated)')
    plt.plot(eps_vals, enc_theo, 'b--', label='Encoded (Theoretical)')
    plt.plot(eps_vals, unenc_sim, 'ro-', label='Unencoded (Simulated)')
    plt.plot(eps_vals, unenc_theo, 'r--', label='Unencoded (Theoretical)')
    plt.xlabel('Coherent Error Strength ε', fontsize=12)
    plt.ylabel('Fidelity', fontsize=12)
    plt.title('Fidelity vs ε for 3-Qubit Bit-Flip Code', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    analyze_single_case()
    print("\n" + "="*60 + "\n")
    simulate_coherent_errors()
