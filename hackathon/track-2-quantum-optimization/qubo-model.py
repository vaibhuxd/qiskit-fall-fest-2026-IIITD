
"""
QUBO model for the six-asset portfolio optimization experiment.
Uses the same synthetic data and objective as portfolio_optimizer.py.
"""

from itertools import product
import numpy as np

# Same data as portfolio_optimizer.py
ASSET_NAMES = ["A", "B", "C", "D", "E", "F"]

RETURNS = np.array([0.12, 0.10, 0.09, 0.07, 0.06, 0.05])

COVARIANCE = np.array([
    [0.040, 0.006, 0.008, 0.004, 0.003, 0.002],
    [0.006, 0.030, 0.007, 0.005, 0.004, 0.003],
    [0.008, 0.007, 0.025, 0.006, 0.004, 0.003],
    [0.004, 0.005, 0.006, 0.020, 0.005, 0.004],
    [0.003, 0.004, 0.004, 0.005, 0.018, 0.005],
    [0.002, 0.003, 0.003, 0.004, 0.005, 0.015],
])

K = 2
RISK_AVERSION = 0.5
PENALTY = 1.0

N = len(ASSET_NAMES)


def objective(x):
    """Return the original objective: return minus risk penalty."""
    x = np.asarray(x, dtype=float)
    return float(RETURNS @ x - RISK_AVERSION * (x @ COVARIANCE @ x))


def qubo_energy(x):
    """
    Energy to MINIMIZE:
      -objective(x) + PENALTY * (sum(x) - K)^2

    Lower energy is better. The penalty discourages invalid selections.
    """
    x = np.asarray(x, dtype=float)
    constraint_error = x.sum() - K
    return -objective(x) + PENALTY * constraint_error**2


# Enumerate all 2^6 binary strings, including infeasible selections.
results = []

for bits in product([0, 1], repeat=N):
    x = np.array(bits, dtype=int)
    score = objective(x)
    energy = qubo_energy(x)
    feasible = int(x.sum()) == K

    results.append({
        "bits": bits,
        "assets": [
            ASSET_NAMES[i] for i, bit in enumerate(bits) if bit
        ],
        "selected": int(x.sum()),
        "objective": score,
        "energy": energy,
        "feasible": feasible,
    })


# Find the lowest-energy feasible solution.
feasible_results = [r for r in results if r["feasible"]]
best_feasible = min(feasible_results, key=lambda r: r["energy"])

# Find the overall lowest-energy solution, including infeasible candidates.
best_overall = min(results, key=lambda r: r["energy"])

print("=" * 58)
print("QUBO PORTFOLIO MODEL — VALIDATION")
print("=" * 58)
print(f"Total binary candidates: {len(results)}")
print(f"Feasible candidates: {len(feasible_results)}")
print(f"Penalty strength: {PENALTY}")
print()
print("Best feasible portfolio:", best_feasible["assets"])
print(f"Original objective: {best_feasible['objective']:.6f}")
print(f"QUBO energy: {best_feasible['energy']:.6f}")
print()
print("Best overall binary candidate:", best_overall["assets"])
print("Assets selected:", best_overall["selected"])
print(f"QUBO energy: {best_overall['energy']:.6f}")
print("Feasible:", best_overall["feasible"])

assert len(results) == 64
assert len(feasible_results) == 15
assert best_feasible["assets"] == ["A", "B"]
assert best_overall["feasible"], (
    "Penalty is not strong enough to make the global minimum feasible."
)

print()
print("PASS: all 64 binary candidates evaluated.")
print("PASS: 15 feasible portfolios identified.")
print("PASS: QUBO minimum matches the classical optimum.")
