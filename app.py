# app.py

import streamlit as st
import numpy as np
import pandas as pd
from model import load_and_preprocess_data, preprocess_data, train_and_evaluate_models, perform_statistical_tests

# Load and preprocess the data
X, y = load_and_preprocess_data()
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Sidebar options
st.sidebar.title('Model Options')
model_type = st.sidebar.selectbox('Choose a model', ['Logistic Regression', 'Naive Bayes', 'Support Vector Machine', 'K-Nearest Neighbors', 'Artificial Neural Network', 'Convolutional Neural Network', 'Recurrent Neural Network'])

# Main content
st.title('AI Model Comparison')
st.write('This app compares various machine learning and neural network models on the Iris dataset.')

# Train and evaluate the selected model
accuracy, report, confusion = train_and_evaluate_models(model_type, X_train, y_train, X_test, y_test)

# Display results
st.write(f'## {model_type} Results')
st.write(f'### Accuracy: {accuracy:.4f}')
st.write('### Classification Report')
st.text(report)
st.write('### Confusion Matrix')
st.write(confusion)

# Perform statistical tests
st.write('## Statistical Tests')
stat_results = perform_statistical_tests((X, y))
for test_name, result in stat_results.items():
    st.write(f'### {test_name}')
    st.write(result)

# User input for prediction
st.sidebar.header('Predict a new sample')
sepal_length = st.sidebar.slider('Sepal Length', min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.sidebar.slider('Sepal Width', min_value=2.0, max_value=4.5, step=0.1)
petal_length = st.sidebar.slider('Petal Length', min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.sidebar.slider('Petal Width', min_value=0.1, max_value=2.5, step=0.1)

# Make prediction
new_sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
new_sample_scaled = StandardScaler().fit_transform(new_sample)
model = train_and_evaluate_models(model_type, X_train, y_train, X_test, y_test)
if model_type == 'Artificial Neural Network':
    prediction = model.predict(new_sample_scaled)
elif model_type == 'Convolutional Neural Network':
    new_sample_scaled = new_sample_scaled.reshape(-1, 2, 2, 1)
    prediction = model.predict(new_sample_scaled)
elif model_type == 'Recurrent Neural Network':
    new_sample_scaled = new_sample_scaled.reshape(-1, 4, 1)
    prediction = model.predict(new_sample_scaled)
else:
    prediction = model.predict(new_sample_scaled)

predicted_class = np.argmax(prediction) if model_type in ['Artificial Neural Network', 'Convolutional Neural Network', 'Recurrent Neural Network'] else prediction[0]
class_names = iris.target_names

st.sidebar.write(f'## Prediction: {class_names[predicted_class]}')

# Visualize model comparison
results = train_and_evaluate_models()
st.write('## Model Comparison')
st.bar_chart(results)
