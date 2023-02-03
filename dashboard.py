import streamlit as st
from hyperbench.dashboard import info, leaderboard, statistics, trajectories, explorer, aggregate
from hyperbench.dashboard.options import Options
from run_benchmark import benchmark

# Settings
options = Options(directory="results", time_based=False)

with st.sidebar:
    options.display()
    info.display()
    filtered = aggregate.load_trajectories(options)

main_tab, explorer_tab, leader_tab = st.tabs(["🔥 Benchmark results", "🧭 Explore datasets", "🥇 Leaderboard"])
with main_tab:
    with st.expander("Datasets included in results", expanded=False):
        list_datasets = aggregate.get_datasets(filtered)
        datasets = st.multiselect("Select datasets to include", list_datasets, default=list_datasets,
                                  label_visibility="collapsed")
        filtered = filtered[filtered.dataset.isin(datasets)]

    with st.expander("Experiments included in results", expanded=True):
        st.dataframe(aggregate.overview(filtered), use_container_width=True)

    # Split by search and eval
    search_trajectories = aggregate.filter_on(filtered, stage="search").drop("virtual", axis=1)
    eval_trajectories = aggregate.filter_on(filtered, stage="eval").drop("virtual", axis=1)

    search_tab, eval_tab, stats_tab = st.tabs(["🧪 Search results", "️🤔 Evaluation results", "📊 Statistics"])

    with search_tab:
        trajectories.display(search_trajectories)

    with eval_tab:
        trajectories.display(eval_trajectories)

    with stats_tab:
        statistics.display(options.directory, filtered, options.target)

with explorer_tab:
    explorer.display(benchmark.datasets)

with leader_tab:
    leaderboard.display("smac_output")
