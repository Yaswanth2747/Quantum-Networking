#!/usr/bin/env python3

import numpy as np
from random import choices
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Statevector, Operator, partial_trace, state_fidelity, DensityMatrix
)

# ============================================================
# CONFIGURATION
# ============================================================

P_VALUES = [0.001, 0.2,0.6]
NUM_LOGICAL = 100

# Physical qubits per encoding strategy (conceptual)
PHYSICAL_PER_LOGICAL = {5: 5, 7: 7, 9: 9}

# Feedback thresholds measured in physical qubits
# (you can tune these later)
BATCH_PHYSICAL = {
    5: 5,  # feedback when 10 physical qubits transmitted
    7: 7,
    9: 9
}

ADAPT_THRESHOLDS = {
    "low": 0.03,
    "medium": 0.10
}

# Cost model
ENCODER_COST = {5: 5, 7: 7, 9: 9}

# Noise model
PAULIS = {
    "I": np.array([[1,0],[0,1]], dtype=complex),
    "X": np.array([[0,1],[1,0]], dtype=complex),
    "Y": np.array([[0,-1j],[1j,0]], dtype=complex),
    "Z": np.array([[1,0],[0,-1]], dtype=complex),
}

def sample_pauli_error(p, n):
    return choices(["I","X","Y","Z"], [1-p, p/3, p/3, p/3], k=n)

# ============================================================
# 5-QUBIT PERFECT CODE
# ============================================================

def encode_circuit():
    qc = QuantumCircuit(5)
    qc.cx(0,1); qc.cx(0,2); qc.cx(0,3); qc.cx(0,4)
    qc.h(0)

    qc.cx(0,1); qc.tdg(1); qc.cx(2,1); qc.s(1)
    qc.cx(0,2); qc.tdg(2); qc.cx(3,2); qc.s(2)
    qc.cx(0,3); qc.tdg(3); qc.cx(4,3); qc.s(3)
    qc.cx(0,4); qc.tdg(4); qc.cx(1,4); qc.s(4)
    return qc

def decode_circuit():
    return encode_circuit().inverse()

# ============================================================
# SINGLE LOGICAL SEND (always 5-qubit physically)
# ============================================================

def run_once(p):
    n = 5
    psi = Statevector.from_label("00000")

    prep = QuantumCircuit(n)
    prep.h(0)
    psi = psi.evolve(Operator(prep))

    psi = psi.evolve(Operator(encode_circuit()))

    errors = sample_pauli_error(p, n)
    for i, pauli in enumerate(errors):
        op = 1
        for q in range(n):
            op = np.kron(op, PAULIS[pauli] if q == i else PAULIS["I"])
        psi = psi.evolve(Operator(op))

    psi = psi.evolve(Operator(decode_circuit()))

    rho = DensityMatrix(psi)
    rho_logical = partial_trace(rho, list(range(1, n)))

    ideal = Statevector.from_label("+")
    f = state_fidelity(rho_logical, ideal)

    return f, (f < 0.9999)

# ============================================================
# ADAPTIVE SIMULATION WITH PHYSICAL BATCHING
# ============================================================

def choose_next_code(freq):
    if freq < ADAPT_THRESHOLDS["low"]:
        return 5
    elif freq < ADAPT_THRESHOLDS["medium"]:
        return 7
    return 9

def simulate_hybrid(p, runs=NUM_LOGICAL):
    current = 5
    fidelities = []
    switches = 0
    usage_counts = {5:0, 7:0, 9:0}
    total_time_cost = 0

    physical_since_feedback = 0
    logical_errors = 0
    logical_count = 0

    for _ in range(runs):
        # accumulate usage and cost
        usage_counts[current] += 1
        total_time_cost += ENCODER_COST[current]

        f, err = run_once(p)
        fidelities.append(f)

        logical_errors += int(err)
        logical_count += 1

        # add physical qubits sent
        physical_since_feedback += PHYSICAL_PER_LOGICAL[current]

        # check batch condition based on physical usage
        if physical_since_feedback >= BATCH_PHYSICAL[current]:
            freq = logical_errors / logical_count

            next_code = choose_next_code(freq)
            if next_code != current:
                switches += 1
                current = next_code

            # reset counters for next batch
            physical_since_feedback = 0
            logical_errors = 0
            logical_count = 0

    return {
        "fidelity": np.mean(fidelities),
        "switches": switches,
        "usage": usage_counts,
        "time_cost": total_time_cost
    }

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=== Hybrid Adaptive QEC (Physical-Batch Mode) ===\n")

    for p in P_VALUES:
        result = simulate_hybrid(p)
        print(f"p = {p:<6} | fidelity = {result['fidelity']:.6f} | switches = {result['switches']}")
        print("Usage:", result["usage"])
        print("Time cost:", result["time_cost"])
        print("--------------------------------------")

if __name__ == "__main__":
    main()
