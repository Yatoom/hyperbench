import pandas as pd
import streamlit as st
from hyperbench.visualization import aggregate
from run_benchmark import benchmark

directory = "results"

with st.sidebar:
    st.subheader("Settings")
    with st.expander("Budget", expanded=True):
        budget = st.radio("Budget type", ('Iterations', 'Time'), index=1 if benchmark.time_based else 0)
        iterations = st.number_input('Budget size', value=benchmark.budget, step=1 if budget == "Iterations" else None)
        all_trajectories = aggregate.get_all_trajectories(directory, iterations=iterations, time_based=budget == 'Time')

    with st.expander("Options", expanded=True):
        target = st.selectbox(
            "Choose the target algorithm",
            aggregate.get_target_algorithms(all_trajectories))
        view = st.radio(
            "Choose your view",
            ('Live view', 'Static view', 'Global view'))

        filtered = aggregate.filter_on(all_trajectories, target=target)

        if view == "Live view":
            filtered = aggregate.live_view(filtered)
        elif view == "Static view":
            filtered = aggregate.static_view(filtered)

    st.subheader("Info")
    with st.expander("About budget"):
        st.write(
            """
            There are two types of budget: 
            - A **time** budget limits the available time (in seconds) an optimizer has to search for a good configuration.
            - An **iteration** budget limits the number of configurations an optimizer can try out.
            For an accurate result, you should pick the budget type you have used in your benchmarks.
            """
        )
    with st.expander("About views"):
        st.write(
            """
            - **Live view** filters the results by the datasets that are completed by all experiments.
            - **Static view** only includes experiments that have completed searching and evaluating on all datasets.
            - **Global view** includes everything, but might not accurately reflect the relative performance of the uncompleted
              experiments. Especially the average loss graph might be off.
            """
        )

main_tab, explorer_tab = st.tabs(["🔥 Benchmark results", "🧭 Explore datasets"])
with main_tab:
    with st.expander("Datasets included in results", expanded=False):
        list_datasets = aggregate.get_datasets(filtered)
        datasets = st.multiselect("Select datasets to include", list_datasets, default=list_datasets, label_visibility="collapsed")
        filtered = filtered[filtered.dataset.isin(datasets)]
    with st.expander("Experiments included in results", expanded=True):
        st.dataframe(aggregate.overview(filtered), use_container_width=True)

    # Split by search and eval
    search_trajectories = aggregate.filter_on(filtered, stage="search")
    eval_trajectories = aggregate.filter_on(filtered, stage="eval")

    tab1, tab2, tab3 = st.tabs(["🧪 Search results", "️🤔 Evaluation results", "📊 Statistics"])
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

    with tab3:
        stats = aggregate.load_stats(directory, filtered, target=target)
        with st.expander("Average runtime statistics", expanded=True):
            st.dataframe(aggregate.get_run_stats(stats), use_container_width=True)
        with st.expander("Average runtimes per dataset", expanded=True):
            optimizer = st.selectbox("Optimizer", stats.optimizer.unique())
            filtered_on_optimizer = stats[stats.optimizer == optimizer]
            st.dataframe(aggregate.get_dataset_stats(filtered_on_optimizer), use_container_width=True)
        with st.expander("Miscellaneous statistics", expanded=True):
            st.dataframe(aggregate.get_other_stats(stats), use_container_width=True)
with explorer_tab:
    enabled = st.checkbox("Enable explorer")
    if enabled:
        table = pd.DataFrame([ds.stats for ds in benchmark.datasets])
        table['dimension'] = table.n_rows * table.n_columns
        st.dataframe(table)


