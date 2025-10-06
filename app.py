import streamlit as st
import pandas as pd

from eca import batch_event_coincidence, run_event_coincidence_analysis


st.set_page_config(page_title="ECA", layout="wide")

st.title("Event Coincidence Analysis")
st.write(
    "Explore how frequently outbreak shocks coincide with other shock categories, how that compares to random alignments, and whether lagged patterns emerge."
)

raw = pd.read_csv("data/01_raw/Shocks.csv")
raw.loc[raw["Shock type"] == "Infectious disease", "Shock category"] = "Outbreak"
raw["Year"] = raw["Year"].astype(int)
raw = raw.rename(columns={"Country name": "Country", "Shock category": "category", "Shock type": "type"})
raw = raw.drop(columns=[c for c in ["type", "count"] if c in raw.columns])
raw = raw[raw["category"].isin(["Outbreak", "CLIMATIC", "CONFLICTS", "ECOLOGICAL", "ECONOMIC", "GEOPHYSICAL", "TECHNOLOGICAL"])]

df_wide = (
    raw.assign(value=1)
    .pivot_table(
        index=["Country", "Year"],
        columns="category",
        values="value",
        aggfunc="max",
        fill_value=0,
    )
    .reset_index()
)

df_wide = df_wide.rename(columns=lambda col: col.title() if isinstance(col, str) and col.isupper() else col)
shock_columns = [col for col in df_wide.columns if col not in ["Country", "Year", "Outbreak"]]
analysis_columns = ["Outbreak"] + shock_columns
df_wide[analysis_columns] = df_wide[analysis_columns].fillna(0).astype(int)

st.sidebar.title("Analysis controls")
simulations = st.sidebar.slider("Monte Carlo simulations", 500, 5000, 2000, step=500)
random_seed = int(st.sidebar.number_input("Random seed", value=42))

st.subheader("Step 1 - Prepare the event series")
st.write("We reshape the shock records into yearly indicators per country for outbreaks and other categories.")
st.dataframe(df_wide)

same_year_results = batch_event_coincidence(
    df_wide[analysis_columns],
    base_event="Outbreak",
    other_events=shock_columns,
    simulations=int(simulations),
    random_state=random_seed,
)

same_year_rows = []
for variable in shock_columns:
    result = same_year_results[variable]
    summary = result["summary"]
    conditional = summary["conditional_rates"]
    same_year_rows.append(
        {
            "Variable": variable,
            "Coincidences": summary["coincidences"],
            "Expected coincidences": result["expected_random_coincidences"],
            "P-value": result["p_value"],
            "Outbreak events": summary["totals"]["Outbreak"],
            f"{variable} events": summary["totals"][variable],
            "Share of outbreak years with event": conditional["Outbreak"],
            "Share of event years with outbreak": conditional[variable],
            "Lift": summary["coincidences"] - result["expected_random_coincidences"],
        }
    )

same_year_df = pd.DataFrame(same_year_rows).sort_values("P-value")

st.subheader("Step 2 - Compare same-year coincidences")
st.write(
    "For each shock category we contrast observed overlaps with outbreaks against Monte Carlo expectations."
)
st.dataframe(same_year_df)

chart_df = same_year_df.set_index("Variable")[
    ["Coincidences", "Expected coincidences"]
]
st.subheader("Step 3 - Random expectation snapshot")
st.write("Observed coincidences are benchmarked against the Monte Carlo average for every category.")
st.bar_chart(chart_df)

positive_variables = [row["Variable"] for row in same_year_rows if row["P-value"] < 0.05 and row["Lift"] > 0]
weak_variables = [row["Variable"] for row in same_year_rows if row["P-value"] >= 0.05]
interpretation_text = ", ".join(positive_variables) or "None of the categories"
weak_text = ", ".join(weak_variables) or "No categories"

st.subheader("Step 4 - Interpret associations")
st.write(f"Notable overlaps (p < 0.05) include: {interpretation_text}.")
st.write(f"Categories resembling random alignment: {weak_text}.")

lag_df = df_wide.copy()
lag_specs = [(variable, lag) for variable in shock_columns for lag in range(1, 6)]
lag_records = []
for variable, lag in lag_specs:
    col_name = f"{variable}_lag{lag}"
    lag_df[col_name] = lag_df.groupby("Country")[variable].shift(lag).fillna(0).astype(int)
    lag_result = run_event_coincidence_analysis(
        lag_df[["Outbreak", col_name]],
        event_a="Outbreak",
        event_b=col_name,
        simulations=int(simulations),
        random_state=random_seed,
    )
    lag_summary = lag_result["summary"]
    lag_records.append(
        {
            "Variable": variable,
            "Lag": lag,
            "Coincidences": lag_summary["coincidences"],
            "Expected coincidences": lag_result["expected_random_coincidences"],
            "P-value": lag_result["p_value"],
            "Share of outbreak years with lagged event": lag_summary["conditional_rates"]["Outbreak"],
            "Lift": lag_summary["coincidences"] - lag_result["expected_random_coincidences"],
        }
    )

lag_results_df = pd.DataFrame(lag_records)
lag_results_df = lag_results_df.sort_values(["Variable", "Lag"])
st.dataframe(lag_results_df)
lag_pivot = lag_results_df.pivot(index="Variable", columns="Lag", values="P-value")

best_lags = (
    lag_results_df.sort_values(["Variable", "P-value", "Lift"], ascending=[True, True, False])
    .groupby("Variable", as_index=False)
    .first()
)
lag_comment = ", ".join(
    f"{row.Variable} (lag {int(row.Lag)})"
    for _, row in best_lags.iterrows()
    if row["P-value"] < 0.05
) or "No lagged comparison crossed the significance threshold."

st.subheader("Step 5 - Explore lagged coincidences")
st.write(
    "We lag each shock category by one to five years and test whether past shocks align with current outbreaks."
)
st.table(lag_pivot)
st.write(f"Lagged associations with p < 0.05: {lag_comment}.")

st.caption(
    "Monte Carlo results vary slightly with the random seed. Consider testing alternative seeds or restricting to specific regions/time periods for targeted insights."
)
