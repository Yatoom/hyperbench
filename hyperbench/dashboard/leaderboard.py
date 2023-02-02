import os
import streamlit as st
from hyperbench.visualization.smac_history import load_leaderboard


def display(smac_folder):
    target = st.selectbox("target", os.listdir(smac_folder))
    target_folder = os.path.join(smac_folder, target)
    optimizer = st.selectbox("optimizer", os.listdir(target_folder))
    optimizer_folder = os.path.join(smac_folder, target, optimizer)
    seed = st.selectbox("seed", os.listdir(optimizer_folder))
    seed_folder = os.path.join(smac_folder, target, optimizer, seed)
    data = st.selectbox("data", os.listdir(seed_folder))
    st.dataframe(load_leaderboard("smac_output", target, optimizer, seed, data))
