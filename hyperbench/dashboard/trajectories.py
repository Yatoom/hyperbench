import pandas as pd
import streamlit as st
from hyperbench.dashboard import aggregate


def display(dataframe):
    tab1, tab2, tab3 = st.tabs(["Average loss", "Ranked", "Normalized average loss"])

    default = dataframe
    normalized = aggregate.normalize(dataframe)
    ranked = aggregate.rank(dataframe)

    with tab1:
        display_average_loss(default)
    with tab2:
        display_average_loss(ranked)
    with tab3:
        display_average_loss(normalized)


def display_average_loss(dataframe: pd.DataFrame):
    df = aggregate.aggregate_over_seeds(dataframe)
    df = aggregate.aggregate_over_datasets(df)
    f = aggregate.visualize(df)
    st.plotly_chart(f)
