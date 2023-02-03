import streamlit as st
import pandas as pd

from hyperbench.provider import Provider


def display(datasets: list[Provider]):
    if datasets is None:
        st.write("The datasets attribute is left empty. It needs to be specified in the Options object.")
        return

    enabled = st.checkbox("Enable explorer")
    if enabled:
        table = pd.DataFrame([ds.stats for ds in datasets])
        table['dimension'] = table.n_rows * (table.n_columns + table.n_classes)
        st.dataframe(table)
