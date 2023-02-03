from hyperbench.dashboard import aggregate
import streamlit as st


def display(directory, filtered, target):
    stats = aggregate.load_stats(directory, filtered, target=target)

    with st.expander("Average runtime statistics", expanded=True):
        st.dataframe(aggregate.get_run_stats(stats), use_container_width=True)

    with st.expander("Average runtimes per dataset", expanded=True):
        optimizer = st.selectbox("Optimizer", stats.optimizer.unique())
        filtered_on_optimizer = stats[stats.optimizer == optimizer]
        st.dataframe(aggregate.get_dataset_stats(filtered_on_optimizer), use_container_width=True)

    with st.expander("Miscellaneous statistics", expanded=True):
        st.dataframe(aggregate.get_other_stats(stats), use_container_width=True)
