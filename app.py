import streamlit as st
import tensorflow as tf
from model import load_and_preprocess_data, preprocess_data, train_and_evaluate_models, perform_statistical_tests
import matplotlib.pyplot as plt

# Load and preprocess data
df = load_and_preprocess_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

# Streamlit app
st.set_page_config(page_title="AI Model Evaluation Dashboard", layout="wide")

# Sidebar for user input
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Naive Bayes", "SVM", "KNN", "PCA", "KMeans", "ANN", "CNN", "RNN"])

# Sidebar for Statistical Tests
st.sidebar.title("Statistical Tests")
show_stats = st.sidebar.checkbox("Show Statistical Tests")

# Train and evaluate models
models_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# Perform statistical tests
if show_stats:
    stats_results = perform_statistical_tests(X_train, y_train, X_test, y_test)
    st.write("### Statistical Test Results")
    for test, result in stats_results.items():
        st.write(f"**{test}:** {result}")

# Display model results
st.title("AI Model Evaluation Results")
st.write(f"**Selected Model:** {model_choice}")
if model_choice in models_results:
    st.write(f"**Accuracy/Score:** {models_results[model_choice]}")
else:
    st.write("Model not found. Please select a valid model.")

# Display data visualization
st.sidebar.title("Data Visualization")
if st.sidebar.checkbox("Show Data Visualization"):
    auto_viz = AutoViz_Class()
    av = auto_viz.AutoViz("")
    av.AutoViz("", df, sep=",")
    st.pyplot(plt)

# Display the dataset
st.sidebar.title("Dataset")
if st.sidebar.checkbox("Show Dataset"):
    st.write(df)

# Additional functionalities
# You can add more interactive elements here for deeper insights
