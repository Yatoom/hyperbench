from hyperbench.visualization import aggregate
import streamlit as st
import pandas as pd


class BenchmarkResults:

    def __init__(self, dataframe: pd.DataFrame):
        # Columns = [optimizer, seed, dataset, virtual, 0, 1, 2, ..., n]
        self.dataframe = dataframe

    def display_graphs(self):
        tab1, tab2, tab3 = st.tabs(["Average loss", "Ranked", "Normalized average loss"])

        default = self.dataframe
        normalized = aggregate.normalize(self.dataframe)
        ranked = aggregate.rank(self.dataframe)

        with tab1:
            self.display_average_loss(default)
        with tab2:
            self.display_average_loss(normalized)
        with tab3:
            self.display_average_loss(ranked)

    @staticmethod
    def display_average_loss(dataframe: pd.DataFrame):
        # Columns = [optimizer, seed, dataset, 0, 1, 2, ..., n]
        df = aggregate.aggregate_over_seeds(dataframe)
        df = aggregate.aggregate_over_datasets(df)
        f = aggregate.visualize(df)
        st.plotly_chart(f)
