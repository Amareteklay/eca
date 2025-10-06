import numpy as np
import pandas as pd


def _binary_series(series: pd.Series) -> np.ndarray:
    """Convert any series to a 0/1 numpy array."""
    return pd.Series(series).fillna(0).astype(int).to_numpy()


def compute_coincidence_summary(df: pd.DataFrame, event_a: str = "Outbreak", event_b: str = "Climatic") -> dict:
    """Return observed coincidence counts and rates for two binary event columns."""
    a = _binary_series(df[event_a])
    b = _binary_series(df[event_b])
    coincidences = int(np.dot(a, b))
    totals = {event_a: int(a.sum()), event_b: int(b.sum())}
    rate_a = np.divide(coincidences, totals[event_a], out=np.full(1, np.nan), where=totals[event_a] > 0)[0]
    rate_b = np.divide(coincidences, totals[event_b], out=np.full(1, np.nan), where=totals[event_b] > 0)[0]
    return {
        "coincidences": coincidences,
        "totals": totals,
        "rates": {
            f"{event_a} with {event_b}": float(rate_a),
            f"{event_b} with {event_a}": float(rate_b),
        },
    }


def monte_carlo_coincidences(
    df: pd.DataFrame,
    event_a: str = "Outbreak",
    event_b: str = "Climatic",
    simulations: int = 1000,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate coincidence counts under random alignment of events."""
    a = _binary_series(df[event_a])
    b = _binary_series(df[event_b])
    rng = np.random.default_rng(random_state)
    draws = np.fromiter(
        (np.dot(a, rng.permutation(b)) for _ in range(simulations)),
        dtype=float,
        count=simulations,
    )
    return draws


def run_event_coincidence_analysis(
    df: pd.DataFrame,
    event_a: str = "Outbreak",
    event_b: str = "Climatic",
    simulations: int = 1000,
    random_state: int | None = None,
) -> dict:
    """Compute observed metrics, simulation outcomes, and a p-value."""
    summary = compute_coincidence_summary(df, event_a, event_b)
    sims = monte_carlo_coincidences(df, event_a, event_b, simulations, random_state)
    observed = summary["coincidences"]
    p_value = float((np.count_nonzero(sims >= observed) + 1) / (simulations + 1))
    return {
        "summary": summary,
        "simulations": sims,
        "p_value": p_value,
        "expected_random_coincidences": float(np.mean(sims)),
    }
