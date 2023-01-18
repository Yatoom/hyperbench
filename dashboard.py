import streamlit as st
from hyperbench.visualization import aggregate

st.subheader("Average loss")

df = aggregate.get_all_trajectories("results")
df = aggregate.filter_on(df, stage="search")
df = aggregate.aggregate_over_seeds(df)
df = aggregate.aggregate_over_datasets(df)
f = aggregate.visualize(df, target="RandomForest")
st.plotly_chart(f)

