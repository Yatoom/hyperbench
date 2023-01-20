import streamlit as st
from hyperbench.visualization import aggregate

all_trajectories = aggregate.get_all_trajectories("results", time_based=True)

with st.sidebar:
    target = st.selectbox(
        "Choose the target algorithm",
        aggregate.get_target_algorithms(all_trajectories))
    view = st.radio(
        "Choose your view",
        ('Live view', 'Static view', 'Global view'))
    with st.expander("About views"):
        st.write(
            """
            - **Live view** filters the results by the datasets that are completed by all experiments.
            - **Static view** only includes experiments that have completed searching and evaluating on all datasets.
            - **Global view** includes everything, but might not accurately reflect the relative performance of the uncompleted
              experiments. Especially the average loss graph might be off.
            """
        )


filtered = aggregate.filter_on(all_trajectories, target=target)

if view == "Live view":
    filtered = aggregate.live_view(filtered)
elif view == "Static view":
    filtered = aggregate.static_view(filtered)

with st.expander("Datasets", expanded=True):
    list_datasets = aggregate.get_datasets(filtered)
    datasets = st.multiselect("Included", list_datasets, default=list_datasets)
    filtered = filtered[filtered.dataset.isin(datasets)]

with st.expander("Experiments", expanded=True):
    st.table(aggregate.overview(filtered))

# Split by search and eval
search_trajectories = aggregate.filter_on(filtered, stage="search")
eval_trajectories = aggregate.filter_on(filtered, stage="eval")

tab1, tab2 = st.tabs(["Search results", "Evaluation results"])
for i, j, k in [("Search set", search_trajectories, tab1), ("Evaluation set", eval_trajectories, tab2)]:
    with k:
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
