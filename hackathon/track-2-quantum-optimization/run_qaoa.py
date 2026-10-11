
"""
QAOA portfolio experiment.
Requires qiskit, scipy, numpy.
Uses the same six-asset problem as qaoa_portfolio.py.
"""

from itertools import product
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorSampler

# --------------------------------------------------
# 1. Problem definition
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
SHOTS = 4096
P = 1  # QAOA depth; begin with one layer


def objective(x):
    x = np.asarray(x, dtype=float)
    return float(MU @ x - LAMBDA * (x @ COV @ x))


def qubo_energy(x):
    x = np.asarray(x, dtype=float)
    return -objective(x) + PENALTY * (x.sum() - K) ** 2


# --------------------------------------------------
# 2. Construct the cost Hamiltonian
# x_i = (1 - Z_i) / 2
# --------------------------------------------------

terms = {}


def add_term(z_qubits, coefficient):
    label = ["I"] * N
    for q in z_qubits:
        label[N - 1 - q] = "Z"
    label = "".join(label)
    terms[label] = terms.get(label, 0.0) + float(coefficient)


def add_linear(i, coefficient):
    add_term((), coefficient / 2)
    add_term((i,), -coefficient / 2)


def add_quadratic(i, j, coefficient):
    add_term((), coefficient / 4)
    add_term((i,), -coefficient / 4)
    add_term((j,), -coefficient / 4)
    add_term((i, j), coefficient / 4)


for i in range(N):
    linear = -MU[i] + LAMBDA * COV[i, i]
    linear += PENALTY * (1 - 2 * K)
    add_linear(i, linear)

    for j in range(i + 1, N):
        quadratic = 2 * LAMBDA * COV[i, j] + 2 * PENALTY
        add_quadratic(i, j, quadratic)

add_term((), PENALTY * K**2)

cost_operator = SparsePauliOp.from_list([
    (label, coeff)
    for label, coeff in terms.items()
    if abs(coeff) > 1e-12
])


# --------------------------------------------------
# 3. Classical reference optimum
# --------------------------------------------------

valid = []

for bits in product([0, 1], repeat=N):
    if sum(bits) == K:
        valid.append((bits, objective(bits)))

best_bits, best_score = max(valid, key=lambda item: item[1])

print("Classical optimum:", [
    ASSETS[i] for i, bit in enumerate(best_bits) if bit
])
print(f"Classical objective: {best_score:.6f}")


# --------------------------------------------------
# 4. Build QAOA ansatz and objective estimator
# --------------------------------------------------

ansatz = QAOAAnsatz(cost_operator=cost_operator, reps=P)
ansatz.measure_all()

# The optimizer evaluates the circuit's statevector energy
# without measurements. Measurement is used later for sampling.
energy_circuit = QAOAAnsatz(cost_operator=cost_operator, reps=P)

def expected_energy(parameters):
    bound = energy_circuit.assign_parameters(parameters)
    state = Statevector.from_instruction(bound)
    return float(np.real(state.expectation_value(cost_operator)))


# --------------------------------------------------
# 5. Optimize parameters
# --------------------------------------------------

rng = np.random.default_rng(42)
initial = rng.uniform(0, 2 * np.pi, size=2 * P)

result = minimize(
    expected_energy,
    initial,
    method="COBYLA",
    options={"maxiter": 250, "rhobeg": 0.5},
)

print("\nQAOA optimization")
print("Optimizer success:", result.success)
print("Optimizer message:", result.message)
print(f"Best expected energy: {result.fun:.6f}")
print("Parameters:", result.x)


# --------------------------------------------------
# 6. Sample measured bitstrings
# --------------------------------------------------

bound_measured = ansatz.assign_parameters(result.x)
sampler = StatevectorSampler(seed=42)

job = sampler.run([bound_measured], shots=SHOTS)
sample_result = job.result()[0]
counts = sample_result.data.meas.get_counts()

# Since measure_all() maps qubit i to classical bit i,
# reverse displayed bitstrings to restore asset order.
portfolio_counts = {}

for bitstring, count in counts.items():
    bits = tuple(int(bit) for bit in bitstring[::-1])
    portfolio_counts[bits] = portfolio_counts.get(bits, 0) + count


# --------------------------------------------------
# 7. Score samples using the original objective
# --------------------------------------------------

total = sum(portfolio_counts.values())
feasible_count = 0
feasible_scores = []
candidate_rows = []

for bits, count in portfolio_counts.items():
    selected = [ASSETS[i] for i, bit in enumerate(bits) if bit]
    feasible = sum(bits) == K
    score = objective(bits)

    if feasible:
        feasible_count += count
        feasible_scores.append((bits, count, score))

    candidate_rows.append({
        "bits": bits,
        "assets": selected,
        "count": count,
        "feasible": feasible,
        "objective": score,
        "energy": qubo_energy(bits),
    })

candidate_rows.sort(key=lambda row: row["count"], reverse=True)

print(f"\nSampling results ({SHOTS} shots)")
print(f"Unique bitstrings observed: {len(portfolio_counts)}")
print(f"Feasible shots: {feasible_count}/{SHOTS}")
print(f"Feasible-shot rate: {feasible_count / SHOTS:.2%}")

if feasible_scores:
    best_sample = max(feasible_scores, key=lambda item: item[2])
    sampled_bits, sampled_count, sampled_score = best_sample
    sampled_assets = [
        ASSETS[i] for i, bit in enumerate(sampled_bits) if bit
    ]

    print("\nBest feasible sampled portfolio:", sampled_assets)
    print(f"Sampled objective: {sampled_score:.6f}")
    print(f"Classical objective: {best_score:.6f}")
    print(f"Objective gap: {best_score - sampled_score:.6f}")
    print(f"Best sampled portfolio count: {sampled_count}")

    exact_optimum_count = sum(
        count for bits, count, _ in feasible_scores
        if np.isclose(objective(bits), best_score)
    )
    print(
        "Shots matching the classical optimum:",
        f"{exact_optimum_count}/{SHOTS}",
    )
else:
    print("No feasible portfolio sampled. Try a stronger penalty or a different run.")

print("\nMost frequently sampled candidates:")
for row in candidate_rows[:10]:
    print(
        f"{row['assets']} | count={row['count']} | "
        f"feasible={row['feasible']} | "
        f"objective={row['objective']:.6f}"
    )

print("\nExperiment complete.")
