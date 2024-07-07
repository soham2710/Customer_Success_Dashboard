# app.py

import streamlit as st
import pandas as pd
from model import load_dataset, perform_statistical_tests, monte_carlo_simulation, run_pycaret
from autoviz.AutoViz_Class import AutoViz_Class

# Streamlit App
st.title("A/B Testing Framework with AI Analysis")

# Sidebar for data upload
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = load_dataset(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())

    # AutoViz Visualization
    st.write("### Data Visualization")
    AV = AutoViz_Class()
    df_viz = AV.AutoViz(filename="", dfte=df)

    # Statistical Tests
    st.write("### Statistical Tests")
    stat_results = perform_statistical_tests(df)
    for test, result in stat_results.items():
        if isinstance(result, dict):
            st.write(f"**{test}**: Statistic = {result['Statistic']}, p-value = {result['p-value']}")
        else:
            st.write(f"**{test}**: {result}")

    # Monte Carlo Simulation
    st.write("### Monte Carlo Simulation")
    monte_carlo_results = monte_carlo_simulation(df)
    st.line_chart(pd.DataFrame(monte_carlo_results))

    # PyCaret Model
    st.write("### PyCaret Model")
    target_column = st.selectbox("Select Target Column", df.columns)
    if target_column:
        best_model, pycaret_results = run_pycaret(df, target_column)
        st.write(pycaret_results)
        st.write(f"Best Model: {best_model}")
