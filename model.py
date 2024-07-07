import pandas as pd
import numpy as np
from autoviz.AutoViz_Class import AutoViz_Class
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, SimpleRNN, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

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

# Auto visualization
def auto_visualize(df):
    AV = AutoViz_Class()
    AV.AutoViz(df)

# Statistical tests
def run_statistical_tests(df):
    results = {}
    control = df[df['target'] == 0]['sepal length (cm)']
    test = df[df['target'] != 0]['sepal length (cm)']

    # t-test
    t_stat, t_p = stats.ttest_ind(control, test)
    results['t-test'] = (t_stat, t_p)

    # z-score
    z_score = (np.mean(test) - np.mean(control)) / np.std(df['sepal length (cm)'])
    results['z-score'] = z_score

    # F-test
    f_stat, f_p = stats.f_oneway(control, test)
    results['f-test'] = (f_stat, f_p)

    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(control, test)
    results['mann-whitney'] = (u_stat, u_p)

    # Chi-square test
    chi2_stat, chi2_p = stats.chisquare(df['target'])
    results['chi-square'] = (chi2_stat, chi2_p)

    # Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(df['sepal length (cm)'], df['sepal width (cm)'])
    results['pearson'] = (pearson_corr, pearson_p)

    # Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(df['sepal length (cm)'], df['sepal width (cm)'])
    results['spearman'] = (spearman_corr, spearman_p)

    # ANOVA
    anova_stat, anova_p = stats.f_oneway(control, test)
    results['anova'] = (anova_stat, anova_p)

    # Monte Carlo simulation (example)
    monte_carlo = np.mean([np.mean(np.random.choice(df['sepal length (cm)'], size=len(control))) for _ in range(1000)])
    results['monte-carlo'] = monte_carlo

    return results

# Build and train an ANN model
def build_ann_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train a CNN model
def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=2, activation='relu', input_shape=(input_shape, 1)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train an RNN model
def build_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=(input_shape, 1)),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def run_models(X_train, X_test, y_train, y_test):
    results = {}

    # ANN
    ann_model = build_ann_model(X_train.shape[1])
    ann_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
    ann_pred = np.argmax(ann_model.predict(X_test), axis=1)
    ann_acc = accuracy_score(np.argmax(y_test, axis=1), ann_pred)
    results['ANN'] = ann_acc

    # CNN
    X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1)
    cnn_model = build_cnn_model(X_train_cnn.shape[1])
    cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=10, verbose=0)
    cnn_pred = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
    cnn_acc = accuracy_score(np.argmax(y_test, axis=1), cnn_pred)
    results['CNN'] = cnn_acc

    # RNN
    rnn_model = build_rnn_model(X_train_cnn.shape[1])
    rnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=10, verbose=0)
    rnn_pred = np.argmax(rnn_model.predict(X_test_cnn), axis=1)
    rnn_acc = accuracy_score(np.argmax(y_test, axis=1), rnn_pred)
    results['RNN'] = rnn_acc

    return results

if __name__ == "__main__":
    df = load_and_preprocess_data()
    auto_visualize(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    stats_results = run_statistical_tests(df)
    model_results = run_models(X_train, X_test, y_train, y_test)
    print(stats_results)
    print(model_results)
