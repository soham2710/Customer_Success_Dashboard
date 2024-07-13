import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, wilcoxon, mannwhitneyu, kruskal, friedmanchisquare, zscore
import joblib

# Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models (if not already trained, otherwise load from file)
def train_models():
    models = {
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_scaled, y)
        joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")
        trained_models[name] = model
    return trained_models

try:
    logistic_regression_model = joblib.load("logistic_regression_model.pkl")
    naive_bayes_model = joblib.load("naive_bayes_model.pkl")
    svm_model = joblib.load("support_vector_machine_model.pkl")
    knn_model = joblib.load("k_nearest_neighbors_model.pkl")
    trained_models = {
        'Logistic Regression': logistic_regression_model,
        'Naive Bayes': naive_bayes_model,
        'Support Vector Machine': svm_model,
        'K-Nearest Neighbors': knn_model
    }
except:
    trained_models = train_models()

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

# Display predictions
st.subheader('Predictions')
for name, model in trained_models.items():
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    st.write(f"{name} Prediction: {iris.target_names[prediction][0]}")
    st.write(f"{name} Prediction Probability: {prediction_proba}")

# Display statistical test results
st.subheader('Statistical Test Results')
test_results = perform_statistical_tests(X, y)
st.write(test_results)
