import enum

import streamlit as st

from hyperbench.dashboard import aggregate


class Options:

    def __init__(self, directory, time_based, smac_directory=None, datasets=None):
        self.directory = directory
        self.time_based = time_based
        self.target = None
        self.view = None
        self.smac_directory = smac_directory
        self.datasets = datasets

    def display(self):
        st.subheader("Options")
        self.set_target()
        self.set_budget()
        self.set_view()

    def set_budget(self):
        budget = st.radio("Budget type", ('Iterations', 'Time'), index=1 if self.time_based else 0)
        self.time_based = budget == "Time"

    def set_target(self):
        targets = aggregate.get_target_algorithms(self.directory)
        self.target = st.selectbox("Choose the target algorithm", targets)

    def set_view(self):
        views = Views.LIVE, Views.STATIC, Views.GLOBAL
        choices = ('Live view', 'Static view', 'Global view')
        view = st.radio("Choose your view", choices)
        self.view = views[choices.index(view)]


class Views(enum.Enum):
    LIVE = 1
    STATIC = 2
    GLOBAL = 3


if __name__ == "__main__":
    options = Options(directory="results", time_based=True)
    options.display()
