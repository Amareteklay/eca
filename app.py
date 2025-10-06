import streamlit as st
import pandas as pd
import numpy as np  



st.set_page_config(page_title="ECA", layout="wide")

st.title("ECA")
st.write("Event Coincidence Analysis")

df = pd.read_csv("data/01_raw/Shocks.csv")
df.loc[df['Shock type'] == "Infectious disease", 'Shock category'] = "Outbreak"
df['Year'] = df['Year'].astype(int)
df = df.rename(columns={"Country name": "Country", "Shock category": "category", "Shock type": "type"})
df = df.drop (columns=["type"])
df = df[(df["category"] == "Outbreak") | (df["category"] == "CLIMATIC")]

st.dataframe(df)

st.sidebar.title("ECA")
st.sidebar.write("Event Coincidence Analysis")