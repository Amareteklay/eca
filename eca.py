import numpy as np
import pandas as pd


def _binary_series(series: pd.Series) -> np.ndarray:
    return pd.Series(series).fillna(0).astype(int).to_numpy()


def compute_coincidence_summary(
    df: pd.DataFrame,
    event_a: str = "Outbreak",
    event_b: str = "Climatic",
) -> dict:
    a = _binary_series(df[event_a])
    b = _binary_series(df[event_b])
    coincidences = int(np.dot(a, b))
    totals = {event_a: int(a.sum()), event_b: int(b.sum())}
    rate_a = np.divide(coincidences, totals[event_a], out=np.full(1, np.nan), where=totals[event_a] > 0)[0]
    rate_b = np.divide(coincidences, totals[event_b], out=np.full(1, np.nan), where=totals[event_b] > 0)[0]
    return {
        "coincidences": coincidences,
        "totals": totals,
        "conditional_rates": {event_a: float(rate_a), event_b: float(rate_b)},
    }


def monte_carlo_coincidences(
    df: pd.DataFrame,
    event_a: str = "Outbreak",
    event_b: str = "Climatic",
    simulations: int = 1000,
    random_state: int | None = None,
) -> np.ndarray:
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


def batch_event_coincidence(
    df: pd.DataFrame,
    base_event: str = "Outbreak",
    other_events: list[str] | None = None,
    simulations: int = 1000,
    random_state: int | None = None,
) -> dict:
    events = other_events or [c for c in df.columns if c != base_event]
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**32 - 1, size=len(events), dtype=np.uint32) if events else np.array([], dtype=np.uint32)
    results = {}
    for index, event in enumerate(events):
        seed = int(seeds[index]) if index < len(seeds) else None
        results[event] = run_event_coincidence_analysis(
            df,
            event_a=base_event,
            event_b=event,
            simulations=simulations,
            random_state=seed,
        )
    return results
