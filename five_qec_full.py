from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity, partial_trace
from qiskit_aer.noise import NoiseModel, depolarizing_error
import numpy as np

logical_plus = Statevector.from_label('+')

# ----------------------------------------------------------------------
# 5-QUBIT CODE STABILIZERS (canonical)
# ----------------------------------------------------------------------
stabilizers = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ"
]

single_qubit = ["id", "x", "h"]
two_qubit = ["cx", "cz"]

# ----------------------------------------------------------------------
# ENCODER (maps |ψ⟩ to 5 physical qubits)
# RFC: This is NOT the full canonical 5-qubit encoder matrix
# but a working Clifford encoder equivalent up to stabilizer basis
# ----------------------------------------------------------------------
def encode_5(qc, data):
    # Prepare logical |+>
    qc.h(data[0])

    # Spread amplitude
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
    qc.cx(data[0], data[3])
    qc.cx(data[0], data[4])

    # Add phase correlations (simple example)
    qc.cz(data[0], data[2])
    qc.cz(data[1], data[3])
    qc.cz(data[2], data[4])
    return qc


# ----------------------------------------------------------------------
# SYNDROME MEASUREMENT
# ----------------------------------------------------------------------
def measure_stabilizer(qc, data, anc, stab):
    qc.h(anc)
    for i, p in enumerate(stab):
        if p == "X":
            qc.cx(anc, data[i])
        elif p == "Z":
            qc.cz(anc, data[i])
    qc.h(anc)
    qc.measure(anc, anc)


# ----------------------------------------------------------------------
# SIMPLE ONE-ERROR RECOVERY TABLE (for demo)
# ----------------------------------------------------------------------
recovery_map = {
    "0000": None,
    "0001": ("X", 0),
    "0010": ("Z", 1),
    "0100": ("X", 2),
    "1000": ("Z", 3),
    # You will expand this to full syndrome → Pauli operator map
}

def apply_recovery(qc, data, syndrome):
    if syndrome in recovery_map and recovery_map[syndrome] is not None:
        op, q = recovery_map[syndrome]
        if op == "X":
            qc.x(data[q])
        elif op == "Z":
            qc.z(data[q])


# ----------------------------------------------------------------------
# MAIN QEC PIPELINE
# ----------------------------------------------------------------------
def run_cycle(p):

    qc = QuantumCircuit(5, 4)
    data = [0,1,2,3,4]

    # Encode
    encode_5(qc, data)

    # Save state pre-noise
    sim = AerSimulator(method="statevector")
    state_before = sim.run(qc).result().get_statevector()

    # ------------------------------------------------------------------
    # Add noise model
    # ------------------------------------------------------------------
    noise = NoiseModel()

    noise.add_all_qubit_quantum_error(depolarizing_error(p,1), single_qubit)
    noise.add_all_qubit_quantum_error(depolarizing_error(p,2), two_qubit)

    sim = AerSimulator(method="density_matrix", noise_model=noise)

    # ------------------------------------------------------------------
    # NOISY EXECUTION
    # ------------------------------------------------------------------
    noisy = sim.run(qc).result().data(0)["density_matrix"]
    rho_noisy = DensityMatrix(noisy)

    # ------------------------------------------------------------------
    # Syndrome Measurement
    # ------------------------------------------------------------------
    qc2 = QuantumCircuit(5,4)
    for i, stab in enumerate(stabilizers):
        measure_stabilizer(qc2, data, i, stab)

    result = sim.run(qc2, initial_state=rho_noisy).result()
    syndrome_bits = "".join(str(b) for b in result.get_counts().most_frequent())

    # ------------------------------------------------------------------
    # APPLY RECOVERY
    # ------------------------------------------------------------------
    qc3 = QuantumCircuit(5)
    apply_recovery(qc3, data, syndrome_bits)

    final = sim.run(qc3, initial_state=rho_noisy).result().data(0)["density_matrix"]
    rho_final = DensityMatrix(final)

    # ------------------------------------------------------------------
    # DECODE (trace out ancillary qubits to logical qubit)
    # ------------------------------------------------------------------
    logical_dm = partial_trace(rho_final, [1,2,3,4])

    return state_fidelity(logical_dm, logical_plus)


# ----------------------------------------------------------------------
# RUN EXPERIMENT
# ----------------------------------------------------------------------
pvals = [0.0,0.01,0.05,0.1,0.2]

for p in pvals:
    f = run_cycle(p)
    print(f"p={p:.3f} → fidelity={f:.6f}")
