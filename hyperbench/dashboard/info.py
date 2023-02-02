import streamlit as st


def display():
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
