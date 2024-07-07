import pandas as pd
import numpy as np
from autoviz.AutoViz_Class import AutoViz_Class
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, SimpleRNN, LSTM
import random

# Load and preprocess the Iris dataset
def load_and_preprocess_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Statistical Tests
def perform_statistical_tests(X_train, y_train, X_test, y_test):
    results = {}

    # F-test
    f_stat, p_val = stats.f_oneway(X_train[:, 0], X_train[:, 1], X_train[:, 2], X_train[:, 3])
    results['F-test'] = (f_stat, p_val)

    # Z-score
    z_scores = np.abs(stats.zscore(X_train, axis=0))
    results['Z-score'] = np.mean(z_scores)

    # T-test
    t_stat, p_val = stats.ttest_ind(X_train[:, 0], X_train[:, 1])
    results['T-test'] = (t_stat, p_val)

    # Chi-squared test
    chi2_stat, p_val = stats.chisquare(np.sum(X_train, axis=0))
    results['Chi-squared'] = (chi2_stat, p_val)

    # Mann-Whitney U test
    u_stat, p_val = stats.mannwhitneyu(X_train[:, 0], X_train[:, 1])
    results['Mann-Whitney U'] = (u_stat, p_val)

    # Kruskal-Wallis H test
    h_stat, p_val = stats.kruskal(X_train[:, 0], X_train[:, 1], X_train[:, 2], X_train[:, 3])
    results['Kruskal-Wallis H'] = (h_stat, p_val)

    # Shapiro-Wilk test
    w_stat, p_val = stats.shapiro(X_train[:, 0])
    results['Shapiro-Wilk'] = (w_stat, p_val)

    # Anderson-Darling test
    ad_stat, critical_values, sig_level = stats.anderson(X_train[:, 0])
    results['Anderson-Darling'] = (ad_stat, critical_values, sig_level)

    # Kolmogorov-Smirnov test
    ks_stat, p_val = stats.ks_2samp(X_train[:, 0], X_train[:, 1])
    results['Kolmogorov-Smirnov'] = (ks_stat, p_val)

    # Monte Carlo simulation
    def monte_carlo_simulation(n_simulations=1000):
        results = []
        for _ in range(n_simulations):
            sample = np.random.choice(X_train[:, 0], size=10, replace=False)
            mean = np.mean(sample)
            results.append(mean)
        return np.mean(results), np.std(results)

    mc_mean, mc_std = monte_carlo_simulation()
    results['Monte Carlo Simulation'] = (mc_mean, mc_std)

    return results

# Model Definitions
def build_ann_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(input_shape[0], 1)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(64, input_shape=(input_shape[0], 1), return_sequences=True),
        LSTM(32),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {}
    
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train.argmax(axis=1))
    log_reg_pred = log_reg.predict(X_test)
    models['Logistic Regression'] = accuracy_score(y_test.argmax(axis=1), log_reg_pred)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train.argmax(axis=1))
    nb_pred = nb.predict(X_test)
    models['Naive Bayes'] = accuracy_score(y_test.argmax(axis=1), nb_pred)

    # SVM
    svm = SVC()
    svm.fit(X_train, y_train.argmax(axis=1))
    svm_pred = svm.predict(X_test)
    models['SVM'] = accuracy_score(y_test.argmax(axis=1), svm_pred)

    # KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train.argmax(axis=1))
    knn_pred = knn.predict(X_test)
    models['KNN'] = accuracy_score(y_test.argmax(axis=1), knn_pred)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    models['PCA'] = pca.explained_variance_ratio_.sum()

    # KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)
    kmeans_pred = kmeans.predict(X_test)
    models['KMeans'] = accuracy_score(y_test.argmax(axis=1), kmeans_pred)

    # ANN
    ann_model = build_ann_model(X_train.shape[1])
    ann_model.fit(X_train, y_train, epochs=10, verbose=0)
    ann_pred = np.argmax(ann_model.predict(X_test), axis=1)
    models['ANN'] = accuracy_score(np.argmax(y_test, axis=1), ann_pred)

    # CNN
    cnn_model = build_cnn_model((X_train.shape[1], 1))
    cnn_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=10, verbose=0)
    cnn_pred = np.argmax(cnn_model.predict(np.expand_dims(X_test, axis=-1)), axis=1)
    models['CNN'] = accuracy_score(np.argmax(y_test, axis=1), cnn_pred)

    # RNN
    rnn_model = build_rnn_model((X_train.shape[1], 1))
    rnn_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=10, verbose=0)
    rnn_pred = np.argmax(rnn_model.predict(np.expand_dims(X_test, axis=-1)), axis=1)
    models['RNN'] = accuracy_score(np.argmax(y_test, axis=1), rnn_pred)

    return models
