import numpy as np
import pandas as pd
import cv2
import os
import glob
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, wilcoxon, mannwhitneyu, kruskal, friedmanchisquare, zscore

# Define image directory and classes
IMAGE_DIR = 'images'
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

# Load and preprocess numerical data
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

# Load and preprocess image data
def load_and_preprocess_image_data(image_dir):
    images = []
    labels = []
    
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(image_dir, class_name)
        image_paths = glob.glob(os.path.join(class_dir, '*.png'))
        
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                continue
            image_resized = cv2.resize(image, (224, 224))
            image_preprocessed = preprocess_input(image_resized)
            images.append(image_preprocessed)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def preprocess_image_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Define models
def build_ann():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(4,)))
    model.add(Dense(32, activation='relu'))
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

def build_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate models
def train_and_evaluate_models(model_type, X_train, y_train, X_test, y_test):
    if model_type in ['Logistic Regression', 'Naive Bayes', 'Support Vector Machine', 'K-Nearest Neighbors']:
        models = {
            'Logistic Regression': LogisticRegression(),
            'Naive Bayes': GaussianNB(),
            'Support Vector Machine': SVC(),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
        model = models[model_type]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)
        return model, accuracy, report, confusion
    elif model_type == 'Artificial Neural Network':
        model = build_ann()
        model.fit(X_train, y_train, epochs=10, verbose=0)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, y_pred_classes, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred_classes)
        return model, accuracy, report, confusion
    elif model_type == 'Recurrent Neural Network':
        X_train_rnn = X_train.reshape(-1, 4, 1)
        X_test_rnn = X_test.reshape(-1, 4, 1)
        model = build_rnn()
        model.fit(X_train_rnn, y_train, epochs=10, verbose=0)
        loss, accuracy = model.evaluate(X_test_rnn, y_test, verbose=0)
        y_pred = model.predict(X_test_rnn)
        y_pred_classes = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, y_pred_classes, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred_classes)
        return model, accuracy, report, confusion
    elif model_type == 'VGG16':
        model = build_vgg16()
        model.fit(X_train, y_train, epochs=10, verbose=0)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, y_pred_classes, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred_classes)
        return model, accuracy, report, confusion

def perform_statistical_tests(data):
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
    results['z-score test'] = {'z_scores_summary': np.mean(z_scores, axis=0)}

    return results

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load")
    image_resized = cv2.resize(image, (224, 224))
    image_preprocessed = preprocess_input(image_resized)
    return np.expand_dims(image_preprocessed, axis=0)

# Example usage
if __name__ == '__main__':
    # Numerical data example
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train and evaluate numerical models
    for model_type in ['Logistic Regression', 'Naive Bayes', 'Support Vector Machine', 'K-Nearest Neighbors', 'Artificial Neural Network', 'Recurrent Neural Network']:
        model, accuracy, report, confusion = train_and_evaluate_models(model_type, X_train, y_train, X_test, y_test)
        print(f"{model_type}:\n")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Classification Report:\n{report}")
        print(f"Confusion Matrix:\n{confusion}\n")
    
    # Image data example
    images, labels = load_and_preprocess_image_data(IMAGE_DIR)
    X_train_img, X_test_img, y_train_img, y_test_img = preprocess_image_data(images, labels)
    
    # Train and evaluate VGG16 on image data
    model, accuracy, report, confusion = train_and_evaluate_models('VGG16', X_train_img, y_train_img, X_test_img, y_test_img)
    print("VGG16:\n")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{confusion}\n")
