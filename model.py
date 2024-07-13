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
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import cv2
import os
import gdown

# Define Google Drive file link
DRIVE_LINK = 'https://drive.google.com/drive/folders/1SOhtRf0EltGkHJPaip12cAElyV2Z0bsZ?usp=sharing'
LOCAL_PATH = 'iris_images'  # Directory to store downloaded images

# Function to download files from Google Drive
def download_images():
    if not os.path.exists(LOCAL_PATH):
        os.makedirs(LOCAL_PATH)
    
    print("Downloading images...")
    url = DRIVE_LINK
    output = os.path.join(LOCAL_PATH, "images.zip")
    gdown.download(url, output, quiet=False, fuzzy=True)

    # Extract the ZIP file
    import zipfile
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(LOCAL_PATH)

# Download images
download_images()

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

# Load images and labels
def load_images_from_directory(directory):
    images = []
    labels = []
    label_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}  # Update as needed based on your dataset

    for label_name, label in label_map.items():
        label_dir = os.path.join(directory, label_name)
        for filename in os.listdir(label_dir):
            file_path = os.path.join(label_dir, filename)
            image = cv2.imread(file_path)
            if image is not None:
                image_resized = cv2.resize(image, (224, 224))
                image_preprocessed = preprocess_input(image_resized)
                images.append(image_preprocessed)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Train and evaluate models
def train_and_evaluate_models(model_type, X_train, y_train, X_test, y_test):
    if model_type in models:
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
        X_train_images, y_train_images = load_images_from_directory(LOCAL_PATH)  # Update path accordingly
        X_test_images, y_test_images = load_images_from_directory(LOCAL_PATH)  # Ideally, use a separate test set
        
        model = build_vgg16()
        model.fit(X_train_images, y_train_images, epochs=10, verbose=0)
        loss, accuracy = model.evaluate(X_test_images, y_test_images, verbose=0)
        y_pred = model.predict(X_test_images)
        y_pred_classes = np.argmax(y_pred, axis=1)
        report = classification_report(y_test_images, y_pred_classes, output_dict=True)
        confusion = confusion_matrix(y_test_images, y_pred_classes)
        return model, accuracy, report, confusion

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
    results['z-score test'] = {'z_scores_summary': np.mean(z_scores, axis=0)}

    return results

def process_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))
    image_preprocessed = preprocess_input(image_resized)
    image_preprocessed = np.expand_dims(image_preprocessed, axis=0)
    return image_preprocessed

# Main execution
X, y = load_and_preprocess_data()
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Train and evaluate models
results = {}
for model_name in models.keys():
    model, accuracy, report, confusion = train_and_evaluate_models(model_name, X_train, y_train, X_test, y_test)
    results[model_name] = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': confusion
    }

ann_model, ann_accuracy, ann_report, ann_confusion = train_and_evaluate_models('Artificial Neural Network', X_train, y_train, X_test, y_test)
rnn_model, rnn_accuracy, rnn_report, rnn_confusion = train_and_evaluate_models('Recurrent Neural Network', X_train, y_train, X_test, y_test)
vgg_model, vgg_accuracy, vgg_report, vgg_confusion = train_and_evaluate_models('VGG16', X_train, y_train, X_test, y_test)

results['Artificial Neural Network'] = {
    'accuracy': ann_accuracy,
    'classification_report': ann_report,
    'confusion_matrix': ann_confusion
}
results['Recurrent Neural Network'] = {
    'accuracy': rnn_accuracy,
    'classification_report': rnn_report,
    'confusion_matrix': rnn_confusion
}
results['VGG16'] = {
    'accuracy': vgg_accuracy,
    'classification_report': vgg_report,
    'confusion_matrix': vgg_confusion
}

# Perform statistical tests
statistical_results = perform_statistical_tests((X, y))
