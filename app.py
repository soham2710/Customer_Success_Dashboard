import streamlit as st
import pandas as pd
import model

# Load and preprocess the data
df = model.load_and_preprocess_data()
X_train, X_test, y_train, y_test = model.preprocess_data(df)

# Streamlit app
st.set_page_config(page_title="A/B Testing Framework", layout="wide")

# Sidebar
st.sidebar.title("A/B Testing Framework")
st.sidebar.write("An A/B testing framework with AI analysis.")
st.sidebar.image("https://path_to_your_image.jpg", caption="Your Name", use_column_width=True)

# Page selection
page = st.sidebar.selectbox("Choose a page", ["Visualization", "Statistical Tests", "Model Analysis"])

# Visualization page
if page == "Visualization":
    st.title("Auto Visualization")
    st.write("Auto visualization of the dataset using AutoViz.")
    model.auto_visualize(df)

# Statistical tests page
elif page == "Statistical Tests":
    st.title("Statistical Tests")
    st.write("Results of various statistical tests.")
    stats_results = model.run_statistical_tests(df)
    for test, result in stats_results.items():
        st.write(f"{test}: {result}")

# Model analysis page
elif page == "Model Analysis":
    st.title("Model Analysis")
    st.write("Running various machine learning models.")
    model_results = model.run_models(X_train, X_test, y_train, y_test)
    for model_name, accuracy in model_results.items():
        st.write(f"{model_name} Accuracy: {accuracy:.2f}")

# Run the app
if __name__ == "__main__":
    st.run()
