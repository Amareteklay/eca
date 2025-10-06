import streamlit as st
import pandas as pd
import numpy as np

from eca import run_event_coincidence_analysis


st.set_page_config(page_title="ECA", layout="wide")

st.title("Event Coincidence Analysis")
st.write("Explore how frequently outbreak and climatic shocks coincide and whether the overlap is stronger than random chance.")

# Load and reshape the data.
df = pd.read_csv("data/01_raw/Shocks.csv")
df.loc[df["Shock type"] == "Infectious disease", "Shock category"] = "Outbreak"
df["Year"] = df["Year"].astype(int)
df = df.rename(columns={"Country name": "Country", "Shock category": "category", "Shock type": "type"})
df = df.drop(columns=["type", "count"])
#df = df[(df["category"] == "Outbreak") | (df["category"] == "CLIMATIC")]
st.dataframe(df.head())
df_wide = (
    df.assign(value=1)
    .pivot_table(
        index=["Country", "Year"],
        columns="category",
        values="value",
        aggfunc="max",
        fill_value=0,
    )
    .reset_index()
)

df_wide = df_wide.rename(columns={"CLIMATIC": "Climatic", "CONFLICTS": "Conflicts", "ECOLOGICAL": "Ecological", "ECONOMIC": "Economic", "GEOPHYSICAL": "Geophysical", "TECHNOLOGICAL": "Technological"})

st.sidebar.title("Analysis controls")
simulations = st.sidebar.slider("Monte Carlo simulations", 1000, 10000, 5000, step=1000)
random_seed = st.sidebar.number_input("Random seed", value=42)

analysis = run_event_coincidence_analysis(
    df_wide,
    event_a="Outbreak",
    event_b="Economic",
    simulations=int(simulations),
    random_state=int(random_seed),
)

summary = analysis["summary"]
coincidences = summary["coincidences"]
totals = summary["totals"]
rates = summary["rates"]
simulation_array = analysis["simulations"]
expected_random = analysis["expected_random_coincidences"]
p_value = analysis["p_value"]
sorted_simulations = np.sort(simulation_array)

st.subheader("Step 1 - Prepare the event series")
st.write("We harmonise the data to yearly country-level indicators for outbreak and climatic shocks.")
st.dataframe(df_wide)

st.subheader("Step 2 - Measure observed coincidences")
st.write(
    "Outbreaks appear in {0} records and climatic shocks in {1} records. They align in {2} country-year combinations.".format(
        totals["Outbreak"], totals["Economic"], coincidences
    )
)
counts_df = pd.DataFrame(
    {
        "Metric": ["Outbreak occurrences", "Economic occurrences", "Joint coincidences"],
        "Count": [totals["Outbreak"], totals["Economic"], coincidences],
    }
)
st.table(counts_df)

rates_df = pd.DataFrame(
    {
        "Relationship": list(rates.keys()),
        "Coincidence rate": [round(value, 3) for value in rates.values()],
    }
)
st.table(rates_df)

st.subheader("Step 3 - Simulate random alignments")
st.write(
    "We shuffle the climatic shock series {0} times to estimate how many coincidences would arise by chance alone. The random runs deliver an average of {1:.2f} coincidences.".format(
        simulations, expected_random
    )
)
frequency_table = pd.Series(simulation_array).describe()
st.table(frequency_table.to_frame(name="Simulated coincidences"))

trend_df = pd.DataFrame({"Simulated coincidences": sorted_simulations})
st.line_chart(trend_df)

st.subheader("Step 4 - Test statistical significance")
st.write(
    "The Monte Carlo test compares the observed {0} coincidences with the simulated distribution. The resulting p-value is {1:.3f}."
    .format(coincidences, p_value)
)
interpretation = (
    "Observed overlap exceeds most random trials, hinting at a meaningful linkage between outbreak and climatic shocks." if p_value < 0.05 else "The overlap is comparable to random alignments, so the linkage could be due to chance."
)
st.info(interpretation)

commentary = (
    "Consider exploring specific regions or time windows to see whether the relationship strengthens under particular conditions. "
    "Further variables such as preparedness or socioeconomic indicators may clarify whether climatic shocks act as a trigger for outbreaks."
)
st.caption(commentary)
