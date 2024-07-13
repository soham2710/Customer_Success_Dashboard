import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
import io
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, wilcoxon, mannwhitneyu, kruskal, friedmanchisquare, zscore

# Load and preprocess data
def load_data():
    # This function should be replaced with your actual data loading logic
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris

X, y, iris = load_data()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load pre-trained models
cnn_model = load_model('model.h5')
best_model = joblib.load('best_model.pkl')  # Replace with the correct model loading logic

# Perform statistical tests
def perform_statistical_tests(X, y):
    results = {}

    # Perform t-test
    t_stat, t_p = ttest_ind(X[y == 0], X[y == 1], axis=0)
    results['t-test'] = {'t_stat': t_stat, 'p_value': t_p}

    # Perform chi-squared test
    chi2_stat, chi2_p, _, _ = chi2_contingency(pd.crosstab(y, X[:, 0]))
    results['chi-squared test'] = {'chi2_stat': chi2_stat, 'p_value': chi2_p}

    # Perform ANOVA
    anova_stat, anova_p = f_oneway(X[y == 0], X[y == 1], X[y == 2])
    results['ANOVA'] = {'anova_stat': anova_stat, 'p_value': anova_p}

    # Perform Wilcoxon test
    wilcoxon_stat, wilcoxon_p = wilcoxon(X[y == 0, 0], X[y == 1, 0])
    results['Wilcoxon test'] = {'wilcoxon_stat': wilcoxon_stat, 'p_value': wilcoxon_p}

    # Perform Mann-Whitney U test
    mannwhitney_stat, mannwhitney_p = mannwhitneyu(X[y == 0, 0], X[y == 1, 0])
    results['Mann-Whitney U test'] = {'mannwhitney_stat': mannwhitney_stat, 'p_value': mannwhitney_p}

    # Perform Kruskal-Wallis test
    kruskal_stat, kruskal_p = kruskal(X[y == 0], X[y == 1], X[y == 2])
    results['Kruskal-Wallis test'] = {'kruskal_stat': kruskal_stat, 'p_value': kruskal_p}

    # Perform Friedman test
    friedman_stat, friedman_p = friedmanchisquare(X[y == 0, 0], X[y == 1, 0], X[y == 2, 0])
    results['Friedman test'] = {'friedman_stat': friedman_stat, 'p_value': friedman_p}

    # Calculate z-scores
    z_scores = zscore(X, axis=0)
    results['Z-scores'] = z_scores

    return results

# Streamlit UI
st.title("Iris Species Prediction and Statistical Analysis")

st.sidebar.header("User Input Parameters")
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', float(iris.data[:, 0].min()), float(iris.data[:, 0].max()), float(iris.data[:, 0].mean()))
    sepal_width = st.sidebar.slider('Sepal width', float(iris.data[:, 1].min()), float(iris.data[:, 1].max()), float(iris.data[:, 1].mean()))
    petal_length = st.sidebar.slider('Petal length', float(iris.data[:, 2].min()), float(iris.data[:, 2].max()), float(iris.data[:, 2].mean()))
    petal_width = st.sidebar.slider('Petal width', float(iris.data[:, 3].min()), float(iris.data[:, 3].max()), float(iris.data[:, 3].mean()))
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocess user input
input_scaled = scaler.transform(input_df)

# Display CNN model predictions
st.subheader('CNN Model Prediction')
uploaded_image = st.file_uploader("Upload an image of an iris flower", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    image = image.resize((64, 64))  # Resize to the size expected by your CNN
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    cnn_prediction = cnn_model.predict(image_array)
    cnn_class = np.argmax(cnn_prediction, axis=1)
    st.write(f"CNN Model Prediction: {iris.target_names[cnn_class[0]]}")

# Display best model predictions
st.subheader('Best Model Predictions')
best_prediction = best_model.predict(input_scaled)
best_prediction_proba = best_model.predict_proba(input_scaled)
st.write(f"Best Model Prediction: {iris.target_names[best_prediction][0]}")
st.write(f"Best Model Prediction Probability: {best_prediction_proba}")

# Display statistical test results
st.subheader('Statistical Test Results')
if st.button('Show Statistical Tests'):
    test_results = perform_statistical_tests(X, y)
    st.write(test_results)

# To run the app, use the command: streamlit run app.py
