import os

import pandas as pd
import streamlit as st
from hyperbench.visualization import aggregate, dashboard
from hyperbench.visualization import dashboard
from hyperbench.visualization.dashboard import BenchmarkResults
from hyperbench.visualization.smac_history import load_leaderboard
from run_benchmark import benchmark

directory = "results"

with st.sidebar:
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

main_tab, explorer_tab, leader_tab = st.tabs(["🔥 Benchmark results", "🧭 Explore datasets", "🥇 Leaderboard"])
with main_tab:
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Budget", expanded=False):
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
    with col2:
        if view == "Live view":
            filtered = aggregate.live_view(filtered)
        elif view == "Static view":
            filtered = aggregate.static_view(filtered)
        with st.expander("Datasets included in results", expanded=False):
            list_datasets = aggregate.get_datasets(filtered)
            datasets = st.multiselect("Select datasets to include", list_datasets, default=list_datasets, label_visibility="collapsed")
            filtered = filtered[filtered.dataset.isin(datasets)]
        with st.expander("Experiments included in results", expanded=True):
            st.dataframe(aggregate.overview(filtered), use_container_width=True)

    # Split by search and eval
    search_trajectories = aggregate.filter_on(filtered, stage="search").drop("virtual", axis=1)
    eval_trajectories = aggregate.filter_on(filtered, stage="eval").drop("virtual", axis=1)

    tab1, tab2, tab3 = st.tabs(["🧪 Search results", "️🤔 Evaluation results", "📊 Statistics"])
    for i, j, k in [("Search set", search_trajectories, tab1), ("Evaluation set", eval_trajectories, tab2)]:
        with k:
            BenchmarkResults(j).display_graphs()

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
        table['dimension'] = table.n_rows * (table.n_columns + table.n_classes)
        st.dataframe(table)

with leader_tab:
    root_folder = "smac_output"
    target = st.selectbox("target", os.listdir(root_folder))
    target_folder = os.path.join(root_folder, target)
    optimizer = st.selectbox("optimizer", os.listdir(target_folder))
    optimizer_folder = os.path.join(root_folder, target, optimizer)
    seed = st.selectbox("seed", os.listdir(optimizer_folder))
    seed_folder = os.path.join(root_folder, target, optimizer, seed)
    data = st.selectbox("data", os.listdir(seed_folder))
    data_folder = os.path.join(root_folder, target, optimizer, seed, data)
    st.dataframe(load_leaderboard("smac_output", target, optimizer, seed, data))