import streamlit as st
from hyperbench.visualization import aggregate

st.header("Search")

st.subheader("Average loss")
df = aggregate.get_all_trajectories("results")
df = aggregate.filter_on(df, stage="search")
df = aggregate.aggregate_over_seeds(df)
df = aggregate.aggregate_over_datasets(df)
f = aggregate.visualize(df, target="RandomForest")
st.plotly_chart(f)

st.subheader("Ranked")
df = aggregate.get_all_trajectories("results")
df = aggregate.filter_on(df, stage="search")
df = aggregate.rank(df)
df = aggregate.aggregate_over_seeds(df)
df = aggregate.aggregate_over_datasets(df)
f = aggregate.visualize(df, target="RandomForest")
st.plotly_chart(f)

st.header("Validation")

st.subheader("Average loss")
df = aggregate.get_all_trajectories("results")
df = aggregate.filter_on(df, stage="eval")
df = aggregate.aggregate_over_seeds(df)
df = aggregate.aggregate_over_datasets(df)
f = aggregate.visualize(df, target="RandomForest")
st.plotly_chart(f)

st.subheader("Ranked")
df = aggregate.get_all_trajectories("results")
df = aggregate.filter_on(df, stage="eval")
df = aggregate.rank(df)
df = aggregate.aggregate_over_seeds(df)
df = aggregate.aggregate_over_datasets(df)
f = aggregate.visualize(df, target="RandomForest")
st.plotly_chart(f)