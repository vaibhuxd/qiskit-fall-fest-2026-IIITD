
"""
QAOA Portfolio Optimizer — Hamiltonian validation
Six synthetic assets; select exactly two.
"""

import numpy as np
from itertools import product
from qiskit.quantum_info import SparsePauliOp, Statevector

# --------------------------------------------------
# Problem data (same as the validated baseline)
# --------------------------------------------------

ASSETS = ["A", "B", "C", "D", "E", "F"]

MU = np.array([0.12, 0.10, 0.09, 0.07, 0.06, 0.05])

COV = np.array([
    [0.040, 0.006, 0.008, 0.004, 0.003, 0.002],
    [0.006, 0.030, 0.007, 0.005, 0.004, 0.003],
    [0.008, 0.007, 0.025, 0.006, 0.004, 0.003],
    [0.004, 0.005, 0.006, 0.020, 0.005, 0.004],
    [0.003, 0.004, 0.004, 0.005, 0.018, 0.005],
    [0.002, 0.003, 0.003, 0.004, 0.005, 0.015],
])

K = 2
LAMBDA = 0.5
PENALTY = 1.0
N = len(ASSETS)


def objective(x):
    x = np.asarray(x, dtype=float)
    return float(MU @ x - LAMBDA * (x @ COV @ x))


def qubo_energy(x):
    x = np.asarray(x, dtype=float)
    return -objective(x) + PENALTY * (x.sum() - K) ** 2


# --------------------------------------------------
# Convert binary variables to Pauli Z operators
#
# x_i = (1 - Z_i) / 2
#
# Build the diagonal Hamiltonian term by term.
# Qubit i corresponds to asset i.
# --------------------------------------------------

def pauli_label(z_qubits):
    """
    Qiskit Pauli labels are ordered from highest-index
    qubit on the left to qubit 0 on the right.
    """
    label = ["I"] * N
    for qubit in z_qubits:
        label[N - 1 - qubit] = "Z"
    return "".join(label)


terms = {}


def add_term(z_qubits, coefficient):
    label = pauli_label(z_qubits)
    terms[label] = terms.get(label, 0.0) + float(coefficient)


def add_binary_linear(i, coefficient):
    # coefficient * x_i = coefficient/2 - coefficient/2 * Z_i
    add_term((), coefficient / 2)
    add_term((i,), -coefficient / 2)


def add_binary_product(i, j, coefficient):
    # coefficient * x_i*x_j
    # = coefficient/4 * (I - Z_i - Z_j + Z_i Z_j)
    add_term((), coefficient / 4)
    add_term((i,), -coefficient / 4)
    add_term((j,), -coefficient / 4)
    add_term((i, j), coefficient / 4)


# Expand:
# E(x) = -mu.x + lambda*x^T COV x
#        + PENALTY*(sum(x)-K)^2
#
# Since x_i^2 = x_i for binary variables:
# (sum(x)-K)^2 = (1-2K)*sum(x_i)
#                + 2*sum_{i<j}(x_i*x_j) + K^2

for i in range(N):
    linear = -MU[i] + LAMBDA * COV[i, i]
    linear += PENALTY * (1 - 2 * K)
    add_binary_linear(i, linear)

    for j in range(i + 1, N):
        # x^T COV x includes both COV[i,j] and COV[j,i].
        quadratic = (
            2 * LAMBDA * COV[i, j] + 2 * PENALTY
        )
        add_binary_product(i, j, quadratic)

# Constant penalty term.
add_term((), PENALTY * K**2)

# Remove numerically negligible terms.
operator_terms = [
    (label, coefficient)
    for label, coefficient in terms.items()
    if abs(coefficient) > 1e-12
]

COST_HAMILTONIAN = SparsePauliOp.from_list(operator_terms)


# --------------------------------------------------
# Validate all 64 basis states
# --------------------------------------------------

energies = []

for bits in product([0, 1], repeat=N):
    # Qubit 0 represents asset A, qubit 1 asset B, etc.
    # Qiskit computational-basis integer uses qubit 0 as LSB.
    basis_index = sum(bit << i for i, bit in enumerate(bits))

    state = Statevector.from_int(basis_index, 2**N)
    quantum_energy = float(
        np.real(state.expectation_value(COST_HAMILTONIAN))
    )
    classical_energy = qubo_energy(bits)

    if not np.isclose(
        quantum_energy, classical_energy, atol=1e-9
    ):
        raise AssertionError(
            f"Mismatch for {bits}: "
            f"quantum={quantum_energy}, "
            f"classical={classical_energy}"
        )

    energies.append((bits, quantum_energy))

best_bits, best_energy = min(energies, key=lambda item: item[1])
best_assets = [
    ASSETS[i] for i, bit in enumerate(best_bits) if bit
]

print("=" * 58)
print("QISKIT HAMILTONIAN VALIDATION")
print("=" * 58)
print("Number of qubits:", N)
print("Pauli terms:", len(operator_terms))
print("Basis states checked:", len(energies))
print("Best portfolio:", best_assets)
print(f"Minimum energy: {best_energy:.6f}")
print()
print("PASS: all 64 Hamiltonian energies match the QUBO.")
print("PASS: Hamiltonian minimum matches the classical optimum.")
