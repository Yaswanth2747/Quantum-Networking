from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli, state_fidelity, partial_trace
import numpy as np


def encode_5():
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(0,2)
    qc.cx(0,3)
    qc.cx(0,4)
    return qc

def decode_5():
    return encode_5().inverse()


def apply_pauli_errors(state, p):
    for q in range(5):
        if np.random.rand() < p:
            op = np.random.choice(["X","Y","Z"])
            state = state.evolve(Pauli(op), [q])
    return state


STABILIZERS = [
    Pauli("XZZXI"),
    Pauli("IXZZX"),
    Pauli("XIXZZ"),
    Pauli("ZXIXZ")
]

SYNDROME_TO_CORRECTION = {
    "0000": ("I",None),
    "1000": ("X",0),
    "0100": ("Z",1),
    "0010": ("Y",2),
    "0001": ("X",3),
    "1111": ("X",4),
}

def compute_syndrome(state):
    syn = []
    for stab in STABILIZERS:
        val = np.real(state.expectation_value(stab))
        syn.append("0" if val > 0 else "1")
    return "".join(syn)


def send_logical_plus(p):
    ideal = Statevector.from_label('+')

    # initial |+0000⟩
    state = Statevector.from_label('+').tensor(Statevector.from_label('0'*4))

    # encode
    state = state.evolve(encode_5())

    # noise
    state = apply_pauli_errors(state, p)

    # syndrome + correction
    syndrome = compute_syndrome(state)
    op, q = SYNDROME_TO_CORRECTION.get(syndrome, ("I",None))
    if op != "I":
        state = state.evolve(Pauli(op), [q])

    # decode
    state = state.evolve(decode_5())

    # extract logical via partial trace
    rho = partial_trace(state, list(range(1,5)))  # keep qubit 0 only

    return state_fidelity(rho, ideal)


# ===== RUN =====

p_list = [0.0, 0.01, 0.05, 0.1, 0.2]

for p in p_list:
    print(f"p={p:.3f} → fidelity = {send_logical_plus(p):.6f}")
