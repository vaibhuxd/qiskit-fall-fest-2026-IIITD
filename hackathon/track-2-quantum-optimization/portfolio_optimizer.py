
"""
Quantum Portfolio Optimizer — Experiment 1
Classical baseline for Qiskit Fall Fest Track 2.

Synthetic demonstration data only; not investment advice.
Select exactly K assets while balancing expected return and risk.
"""

from itertools import combinations
import numpy as np

# --------------------------------------------------
# 1. Synthetic dataset: six assets
# --------------------------------------------------

ASSET_NAMES = ["A", "B", "C", "D", "E", "F"]

# Illustrative expected returns (decimal values)
RETURNS = np.array([
    0.12, 0.10, 0.09, 0.07, 0.06, 0.05
])

# Illustrative covariance matrix.
# Diagonal = individual variance; off-diagonal = covariance.
COVARIANCE = np.array([
    [0.040, 0.006, 0.008, 0.004, 0.003, 0.002],
    [0.006, 0.030, 0.007, 0.005, 0.004, 0.003],
    [0.008, 0.007, 0.025, 0.006, 0.004, 0.003],
    [0.004, 0.005, 0.006, 0.020, 0.005, 0.004],
    [0.003, 0.004, 0.004, 0.005, 0.018, 0.005],
    [0.002, 0.003, 0.003, 0.004, 0.005, 0.015],
])

K = 2                  # Select exactly two assets
RISK_AVERSION = 0.5    # Illustrative parameter, not calibrated


# --------------------------------------------------
# 2. Validate the input data
# --------------------------------------------------

n = len(ASSET_NAMES)

assert RETURNS.shape == (n,)
assert COVARIANCE.shape == (n, n)
assert np.allclose(COVARIANCE, COVARIANCE.T), (
    "Covariance matrix must be symmetric."
)
assert np.linalg.eigvalsh(COVARIANCE).min() >= -1e-10, (
    "Covariance matrix must be positive semidefinite."
)
assert 0 < K <= n


# --------------------------------------------------
# 3. Define the objective
# --------------------------------------------------

def portfolio_objective(selection):
    """
    selection: binary vector; 1 means selected, 0 means excluded.

    Objective = expected return - lambda * portfolio variance.
    """
    x = np.asarray(selection, dtype=float)

    expected_return = float(RETURNS @ x)
    variance = float(x @ COVARIANCE @ x)
    score = expected_return - RISK_AVERSION * variance

    return score, expected_return, variance


# --------------------------------------------------
# 4. Exact classical baseline: enumerate feasible sets
# --------------------------------------------------

results = []

for chosen_indices in combinations(range(n), K):
    x = np.zeros(n, dtype=int)
    x[list(chosen_indices)] = 1

    score, expected_return, variance = portfolio_objective(x)

    results.append({
        "assets": [ASSET_NAMES[i] for i in chosen_indices],
        "selection": x.tolist(),
        "score": score,
        "expected_return": expected_return,
        "variance": variance,
    })

results.sort(key=lambda item: item["score"], reverse=True)
best = results[0]


# --------------------------------------------------
# 5. Display and verify the answer
# --------------------------------------------------

print("=" * 55)
print("CLASSICAL PORTFOLIO OPTIMIZATION BASELINE")
print("=" * 55)
print(f"Number of assets: {n}")
print(f"Assets selected: {K}")
print(f"Feasible portfolios evaluated: {len(results)}")
print(f"Risk-aversion parameter: {RISK_AVERSION}")
print()
print("Best portfolio:", ", ".join(best["assets"]))
print("Binary selection:", best["selection"])
print(f"Expected return (illustrative): {best['expected_return']:.4f}")
print(f"Portfolio variance: {best['variance']:.6f}")
print(f"Objective score: {best['score']:.6f}")
print()

print("TOP 5 FEASIBLE PORTFOLIOS")
for rank, result in enumerate(results[:5], start=1):
    print(
        f"{rank}. {result['assets']} | "
        f"score={result['score']:.6f} | "
        f"return={result['expected_return']:.4f} | "
        f"variance={result['variance']:.6f}"
    )

assert len(results) == 15
assert all(sum(result["selection"]) == K for result in results)

print()
print("PASS: all 15 feasible portfolios were evaluated.")
print("PASS: every evaluated portfolio selects exactly K assets.")
