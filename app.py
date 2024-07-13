import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import load_and_preprocess_data, preprocess_data, train_and_evaluate_models, perform_statistical_tests, get_llm_summary

# Function to load user-uploaded data
def load_user_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# Sidebar options
st.sidebar.title('Model Options')
uploaded_file = st.sidebar.file_uploader('Upload your CSV file', type=['csv'])
model_type = st.sidebar.selectbox('Choose a model', [
    'Logistic Regression', 'Naive Bayes', 'Support Vector Machine', 
    'K-Nearest Neighbors', 'Artificial Neural Network', 
    'Recurrent Neural Network'
])

# Load and preprocess data
if uploaded_file:
    X, y = load_user_data(uploaded_file)
else:
    X, y = load_and_preprocess_data()
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Main content
st.title('AI Model Comparison')
st.write('This app compares various machine learning and neural network models.')

# Train and evaluate the selected model
accuracy, report, confusion, model = train_and_evaluate_models(model_type, X_train, y_train, X_test, y_test)

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

# Get summary and interpretation using LLM
summary = get_llm_summary(stat_results)
st.write('## Summary and Interpretation')
st.write(summary)

# User input for prediction
st.sidebar.header('Predict a new sample')
num_features = X.shape[1]
input_data = []
for i in range(num_features):
    feature_value = st.sidebar.number_input(f'Feature {i+1}', value=float(X[0, i]))
    input_data.append(feature_value)
new_sample = np.array([input_data])

# Make prediction
scaler = StandardScaler()
new_sample_scaled = scaler.fit_transform(new_sample)
if model_type == 'Artificial Neural Network':
    prediction = model.predict(new_sample_scaled)
    predicted_class = np.argmax(prediction)
elif model_type == 'Recurrent Neural Network':
    new_sample_scaled = new_sample_scaled.reshape(-1, 4, 1)
    prediction = model.predict(new_sample_scaled)
    predicted_class = np.argmax(prediction)
else:
    predicted_class = model.predict(new_sample_scaled)[0]

class_names = ['Class 0', 'Class 1', 'Class 2']
predicted_class_name = class_names[predicted_class]

st.sidebar.write('### Prediction')
st.sidebar.write(f'The predicted class is: {predicted_class_name}')
