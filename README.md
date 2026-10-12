# Smarter Choices at Scale — Quantum Portfolio Optimization

## Project Overview
This project explores QAOA (Quantum Approximate Optimization Algorithm) for a constrained portfolio-selection problem using Qiskit.

The beginner challenge selects exactly 3 stocks from a set of 6 using synthetic expected returns and a covariance matrix.

## Method
- Generate reproducible synthetic portfolio data using random seed 2026.
- Formulate the portfolio objective with a penalty for violating the stock-selection constraint.
- Convert the objective into a Pauli Hamiltonian.
- Verify the Hamiltonian against all 64 possible binary selections.
- Optimize QAOA parameters and sample candidate portfolios.
- Compare QAOA results against exhaustive classical search over all 20 feasible portfolios.
- Investigate QAOA depth and penalty strength.

## Preliminary Results
- Best feasible portfolio found by exhaustive search: stocks 2, 3, and 6.
- Best feasible objective value: -0.50301469.
- QAOA exact-optimum sampling rate in the recorded run: 1.56% (32 of 2,048 shots).
- QAOA samples satisfying the exactly-three-stocks constraint: 53.47% in the recorded run.

These results are preliminary simulator results. Penalty and depth experiments used limited optimization budgets and should not be interpreted as proof of globally optimal QAOA settings.

## Files
- `Copy_of_challenge_2_tutorial.ipynb` — notebook containing the implementation, experiments, and outputs.
- `qaoa_portfolio_results.csv` — experiment summary.

## Reproducibility
Open the notebook in Google Colab, install any missing dependencies listed in its setup cells, and run the cells in order.

## Limitations
The experiments use a classical statevector simulator and a small six-stock instance. They do not demonstrate a quantum advantage over classical optimization. Larger instances and more extensive optimization runs are future work.
