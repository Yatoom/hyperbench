import streamlit as st
from hyperbench.visualization import aggregate

all_trajectories = aggregate.get_all_trajectories("results")

st.header("Included experiments")
target = st.selectbox(
    "Choose the target algorithm",
    aggregate.get_target_algorithms(all_trajectories))
view = st.radio(
    "Choose your view",
    ('Live view', 'Static view', 'Global view'))
st.caption(
    """
    - **Live view** filters the results by the datasets that are completed by all experiments.
    - **Static view** only includes experiments that have completed searching and evaluating on all datasets.
    - **Global view** includes everything, but might not accurately reflect the relative performance of the uncompleted 
      experiments. Especially the average loss graph might be off.
    """
)

filtered = aggregate.filter_on(all_trajectories, target=target)

if view == "Live view":
    filtered = aggregate.intersect(filtered)
elif view == "Static view":
    filtered = aggregate.union(filtered)

st.table(aggregate.overview(filtered))

# Split by search and eval
search_trajectories = aggregate.filter_on(filtered, stage="search")
eval_trajectories = aggregate.filter_on(filtered, stage="eval")

for i, j in [("Search set", search_trajectories), ("Evaluation set", eval_trajectories)]:
    st.header(i)

    st.subheader("Average loss")
    df = aggregate.aggregate_over_seeds(j)
    df = aggregate.aggregate_over_datasets(df)
    f = aggregate.visualize(df)
    st.plotly_chart(f)

    st.subheader("Ranked")
    df = aggregate.rank(j)
    df = aggregate.aggregate_over_seeds(df)
    df = aggregate.aggregate_over_datasets(df)
    f = aggregate.visualize(df)
    st.plotly_chart(f)

    st.subheader("Normalized average loss")
    df = aggregate.normalize(j)
    df = aggregate.aggregate_over_seeds(df)
    df = aggregate.aggregate_over_datasets(df)
    f = aggregate.visualize(df)
    st.plotly_chart(f)
