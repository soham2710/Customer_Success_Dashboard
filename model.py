import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, SimpleRNN

# Load and preprocess data
def load_and_preprocess_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

def build_ann():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(4,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(2, 2, 1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_rnn():
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(4, 1), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate models
def train_and_evaluate_models(model_type, X_train, y_train, X_test, y_test):
    if model_type in models:
        model = models[model_type]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        return accuracy, report, confusion
    elif model_type == 'Artificial Neural Network':
        model = build_ann()
        model.fit(X_train, y_train, epochs=10, verbose=0)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, y_pred_classes)
        confusion = confusion_matrix(y_test, y_pred_classes)
        return accuracy, report, confusion
    elif model_type == 'Convolutional Neural Network':
        X_train_cnn = X_train.reshape(-1, 2, 2, 1)
        X_test_cnn = X_test.reshape(-1, 2, 2, 1)
        model = build_cnn()
        model.fit(X_train_cnn, y_train, epochs=10, verbose=0)
        loss, accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
        y_pred = model.predict(X_test_cnn)
        y_pred_classes = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, y_pred_classes)
        confusion = confusion_matrix(y_test, y_pred_classes)
        return accuracy, report, confusion
    elif model_type == 'Recurrent Neural Network':
        X_train_rnn = X_train.reshape(-1, 4, 1)
        X_test_rnn = X_test.reshape(-1, 4, 1)
        model = build_rnn()
        model.fit(X_train_rnn, y_train, epochs=10, verbose=0)
        loss, accuracy = model.evaluate(X_test_rnn, y_test, verbose=0)
        y_pred = model.predict(X_test_rnn)
        y_pred_classes = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, y_pred_classes)
        confusion = confusion_matrix(y_test, y_pred_classes)
        return accuracy, report, confusion

# Perform statistical tests
def perform_statistical_tests(data):
    from scipy.stats import ttest_ind, chi2_contingency, f_oneway, wilcoxon, mannwhitneyu, kruskal, friedmanchisquare, zscore
    results = {}
    X, y = data

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

    # Perform Kruskal-Wallis H test
    kruskal_stat, kruskal_p = kruskal(X[y == 0], X[y == 1], X[y == 2])
    results['Kruskal-Wallis H test'] = {'kruskal_stat': kruskal_stat, 'p_value': kruskal_p}

    # Perform Friedman test
    friedman_stat, friedman_p = friedmanchisquare(X[y == 0, 0], X[y == 1, 0], X[y == 2, 0])
    results['Friedman test'] = {'friedman_stat': friedman_stat, 'p_value': friedman_p}

    # Perform z-score test
    z_scores = zscore(X)
    results['z-score test'] = {'z_scores': z_scores}

    return results

# Streamlit App
def main():
    st.title("Iris Dataset - Model Training and Evaluation")

    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    st.sidebar.title("Select Model")
    model_type = st.sidebar.selectbox("Model", ["Logistic Regression", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbors", "Artificial Neural Network", "Convolutional Neural Network", "Recurrent Neural Network"])

    if st.sidebar.button("Train and Evaluate"):
        st.write(f"### {model_type}")
        accuracy, report, confusion = train_and_evaluate_models(model_type, X_train, y_train, X_test, y_test)
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write("**Classification Report:**")
        st.text(report)
        st.write("**Confusion Matrix:**")
        st.write(confusion)

    if st.sidebar.button("Perform Statistical Tests"):
        st.write("### Statistical Tests")
        results = perform_statistical_tests((X, y))
        for test, result in results.items():
            st.write(f"**{test}:**")
            st.write(result)

if __name__ == "__main__":
    main()
