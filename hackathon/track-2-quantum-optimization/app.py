
import time
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorSampler


# ============================================================
# QUANTUM PORTFOLIO OPTIMIZER
# Research prototype using synthetic data
# ============================================================

st.set_page_config(
    page_title="Quantum Portfolio Optimizer",
    page_icon="⚛️",
    layout="wide",
)

ASSETS = ["A", "B", "C", "D", "E", "F"]

DEFAULT_RETURNS = np.array([
    0.12, 0.10, 0.09, 0.07, 0.06, 0.05
])

DEFAULT_COV = np.array([
    [0.040, 0.006, 0.008, 0.004, 0.003, 0.002],
    [0.006, 0.030, 0.007, 0.005, 0.004, 0.003],
    [0.008, 0.007, 0.025, 0.006, 0.004, 0.003],
    [0.004, 0.005, 0.006, 0.020, 0.005, 0.004],
    [0.003, 0.004, 0.004, 0.005, 0.018, 0.005],
    [0.002, 0.003, 0.003, 0.004, 0.005, 0.015],
])


# ============================================================
# OBJECTIVE AND QUBO HAMILTONIAN
# ============================================================

def objective(x, mu, cov, risk_aversion):
    x = np.asarray(x, dtype=float)
    return float(mu @ x - risk_aversion * (x @ cov @ x))


def build_cost_hamiltonian(mu, cov, risk_aversion, k, penalty):
    """
    Minimize:
        -objective(x) + penalty * (sum(x) - k)^2

    Binary encoding:
        x_i = (1 - Z_i) / 2
    """
    n = len(mu)
    terms = {}

    def add_term(qubits, coefficient):
        label = ["I"] * n

        for q in qubits:
            label[n - 1 - q] = "Z"

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

    for i in range(n):
        linear = -mu[i] + risk_aversion * cov[i, i]
        linear += penalty * (1 - 2 * k)
        add_linear(i, linear)

        for j in range(i + 1, n):
            quadratic = (
                2 * risk_aversion * cov[i, j] + 2 * penalty
            )
            add_quadratic(i, j, quadratic)

    add_term((), penalty * k**2)

    return SparsePauliOp.from_list([
        (label, coefficient)
        for label, coefficient in terms.items()
        if abs(coefficient) > 1e-12
    ])


# ============================================================
# CLASSICAL BASELINE AND QAOA
# ============================================================

def run_experiment(
    mu, cov, risk_aversion, k, penalty, depth, shots, seed
):
    n = len(mu)

    # Exact classical baseline: all feasible portfolios.
    candidates = []

    for chosen in combinations(range(n), k):
        x = np.zeros(n, dtype=int)
        x[list(chosen)] = 1
        candidates.append(
            (x, objective(x, mu, cov, risk_aversion))
        )

    best_x, best_score = max(
        candidates, key=lambda item: item[1]
    )

    cost = build_cost_hamiltonian(
        mu, cov, risk_aversion, k, penalty
    )

    # Validate every computational-basis energy.
    for integer in range(2**n):
        bits = np.array([
            (integer >> i) & 1 for i in range(n)
        ])

        state = Statevector.from_int(integer, 2**n)
        quantum_energy = float(
            np.real(state.expectation_value(cost))
        )

        expected_energy = (
            -objective(bits, mu, cov, risk_aversion)
            + penalty * (bits.sum() - k)**2
        )

        if not np.isclose(
            quantum_energy, expected_energy, atol=1e-8
        ):
            raise ValueError(
                f"Hamiltonian validation failed for {bits.tolist()}"
            )

    # Parameterized QAOA circuit.
    circuit = QAOAAnsatz(
        cost_operator=cost,
        reps=depth,
    )

    def expected_energy(parameters):
        bound = circuit.assign_parameters(parameters)
        state = Statevector.from_instruction(bound)
        return float(
            np.real(state.expectation_value(cost))
        )

    rng = np.random.default_rng(seed)
    initial = rng.uniform(0, 2 * np.pi, size=2 * depth)

    optimization = minimize(
        expected_energy,
        initial,
        method="COBYLA",
        options={"maxiter": 250, "rhobeg": 0.5},
    )

    # Sample optimized circuit.
    measured_circuit = circuit.assign_parameters(
        optimization.x
    )
    measured_circuit.measure_all()

    sampler = StatevectorSampler(seed=seed)
    job = sampler.run(
        [measured_circuit],
        shots=shots,
    )
    result = job.result()[0]
    raw_counts = result.data.meas.get_counts()

    # Reverse displayed strings to restore asset-index order.
    counts = {}

    for bitstring, count in raw_counts.items():
        bits = tuple(
            int(bit) for bit in bitstring[::-1]
        )
        counts[bits] = counts.get(bits, 0) + count

    rows = []
    feasible_shots = 0
    optimum_shots = 0

    for bits, count in counts.items():
        x = np.array(bits, dtype=int)
        feasible = int(x.sum()) == k
        score = objective(x, mu, cov, risk_aversion)

        selected = [
            ASSETS[i]
            for i, bit in enumerate(bits)
            if bit
        ]

        if feasible:
            feasible_shots += count

            if np.isclose(score, best_score, atol=1e-10):
                optimum_shots += count

        rows.append({
            "Portfolio": ", ".join(selected) if selected else "(none)",
            "Bitstring": "".join(map(str, bits)),
            "Count": count,
            "Feasible": feasible,
            "Objective": score,
        })

    rows.sort(
        key=lambda row: row["Count"],
        reverse=True,
    )

    feasible_rows = [
        row for row in rows if row["Feasible"]
    ]

    best_sample = (
        max(feasible_rows, key=lambda row: row["Objective"])
        if feasible_rows else None
    )

    return {
        "classical_portfolio": [
            ASSETS[i]
            for i, bit in enumerate(best_x)
            if bit
        ],
        "classical_score": best_score,
        "sampled_best": best_sample,
        "feasible_rate": feasible_shots / shots,
        "optimum_rate": optimum_shots / shots,
        "counts": rows,
        "optimizer_success": bool(optimization.success),
        "optimizer_message": str(optimization.message),
        "expected_energy": float(optimization.fun),
    }


# ============================================================
# DASHBOARD INTERFACE
# ============================================================

st.title("⚛️ Quantum Portfolio Optimizer")
st.caption(
    "QAOA portfolio selection | Qiskit statevector simulator"
)

st.warning(
    "Research prototype using synthetic data. "
    "This is not investment advice and does not establish "
    "quantum advantage."
)

with st.sidebar:
    st.header("Portfolio settings")

    k = st.slider(
        "Assets to select",
        min_value=1,
        max_value=len(ASSETS),
        value=2,
    )

    risk_aversion = st.slider(
        "Risk aversion",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
    )

    penalty = st.select_slider(
        "Constraint penalty",
        options=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        value=1.0,
    )

    depth = st.slider(
        "QAOA depth (p)",
        min_value=1,
        max_value=3,
        value=1,
    )

    shots = st.select_slider(
        "Measurement shots",
        options=[512, 1024, 2048, 4096, 8192],
        value=4096,
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=42,
        step=1,
    )


# Editable asset returns.
st.subheader("1. Asset inputs")
st.write("These are illustrative expected returns, not market forecasts.")

columns = st.columns(3)
returns = []

for i, asset in enumerate(ASSETS):
    with columns[i % 3]:
        value = st.number_input(
            f"Asset {asset} expected return",
            min_value=-1.0,
            max_value=1.0,
            value=float(DEFAULT_RETURNS[i]),
            step=0.01,
            format="%.3f",
            key=f"return_{asset}",
        )
        returns.append(value)


st.subheader("2. Risk model")
st.write(
    "The covariance matrix is fixed synthetic data in this version. "
    "Changing expected returns does not update the covariance matrix."
)

with st.expander("View covariance matrix"):
    st.dataframe(
        pd.DataFrame(
            DEFAULT_COV,
            index=ASSETS,
            columns=ASSETS,
        ),
        use_container_width=True,
    )


# Run optimization.
if st.button(
    "Run portfolio optimization",
    type="primary",
    use_container_width=True,
):
    mu = np.array(returns, dtype=float)

    try:
        start = time.perf_counter()

        with st.spinner(
            "Validating Hamiltonian, optimizing QAOA and sampling..."
        ):
            output = run_experiment(
                mu=mu,
                cov=DEFAULT_COV,
                risk_aversion=float(risk_aversion),
                k=int(k),
                penalty=float(penalty),
                depth=int(depth),
                shots=int(shots),
                seed=int(seed),
            )

        elapsed = time.perf_counter() - start

        st.session_state["portfolio_output"] = output
        st.session_state["portfolio_runtime"] = elapsed

    except Exception as exc:
        st.error(f"Optimization failed: {exc}")
        st.exception(exc)


# ============================================================
# RESULTS
# ============================================================

if "portfolio_output" in st.session_state:
    output = st.session_state["portfolio_output"]
    elapsed = st.session_state["portfolio_runtime"]

    st.divider()
    st.subheader("3. Optimization results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Classical optimum",
            ", ".join(output["classical_portfolio"]),
            f"Objective: {output['classical_score']:.6f}",
        )

    with col2:
        best = output["sampled_best"]

        st.metric(
            "Best feasible QAOA sample",
            best["Portfolio"] if best else "None found",
            (
                f"Objective: {best['Objective']:.6f}"
                if best else None
            ),
        )

    with col3:
        st.metric(
            "Feasible shots",
            f"{output['feasible_rate']:.1%}",
        )

    st.metric(
        "Shots matching the classical optimum",
        f"{output['optimum_rate']:.1%}",
    )

    st.metric(
        "Experiment runtime",
        f"{elapsed:.2f} seconds",
    )

    st.write(
        f"Optimizer status: {output['optimizer_message']}"
    )
    st.write(
        f"Optimized expected energy: "
        f"{output['expected_energy']:.6f}"
    )

    st.subheader("4. Sampled portfolios")

    table = pd.DataFrame(output["counts"])
    st.dataframe(
        table.head(15),
        use_container_width=True,
    )

    if not table.empty:
        st.bar_chart(
            table.head(10).set_index("Portfolio")[["Count"]]
        )

    st.download_button(
        "Download results as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="qaoa_portfolio_results.csv",
        mime="text/csv",
    )

    st.info(
        "The classical baseline exactly enumerates feasible portfolios. "
        "QAOA is evaluated on an ideal simulator. These results do not "
        "demonstrate quantum advantage."
    )
