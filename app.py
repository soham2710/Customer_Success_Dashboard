import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import (
    load_and_preprocess_data,
    preprocess_data,
    load_and_preprocess_image_data,
    preprocess_image_data,
    train_and_evaluate_models,
    train_and_evaluate_image_model,
    perform_statistical_tests,
    process_image
)

# Create a temporary directory for uploaded images
if not os.path.exists('temp'):
    os.makedirs('temp')

# Sidebar options
st.sidebar.title('Model Options')
uploaded_image = st.sidebar.file_uploader('Upload an Image of Iris Flower', type=['jpg', 'jpeg', 'png'])
model_type = st.sidebar.selectbox(
    'Choose a model',
    [
        'Logistic Regression', 'Naive Bayes', 'Support Vector Machine',
        'K-Nearest Neighbors', 'Artificial Neural Network', 'Recurrent Neural Network', 'VGG16'
    ]
)

# Load and preprocess data
X, y = load_and_preprocess_data()
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Load and preprocess image data
image_dir = 'images'
images, labels = load_and_preprocess_image_data(image_dir)
X_train_img, X_test_img, y_train_img, y_test_img = preprocess_image_data(images, labels)

# Train and evaluate selected model
if model_type in ['Logistic Regression', 'Naive Bayes', 'Support Vector Machine', 'K-Nearest Neighbors']:
    accuracy, report, confusion = train_and_evaluate_models(model_type, X_train, y_train, X_test, y_test)

    # Display model performance
    st.header('Model Performance')
    st.write(f'**Model Type:** {model_type}')
    st.write(f'**Accuracy:** {accuracy:.2f}')
    st.write('**Classification Report:**')
    st.dataframe(pd.DataFrame(report).transpose())
    st.write('**Confusion Matrix:**')
    st.dataframe(pd.DataFrame(confusion))

elif model_type in ['Artificial Neural Network', 'Recurrent Neural Network', 'VGG16']:
    if model_type == 'VGG16':
        model, accuracy, report, confusion = train_and_evaluate_image_model(X_train_img, y_train_img, X_test_img, y_test_img)
    else:
        accuracy, report, confusion = train_and_evaluate_models(model_type, X_train, y_train, X_test, y_test)

    # Display model performance
    st.header('Model Performance')
    st.write(f'**Model Type:** {model_type}')
    st.write(f'**Accuracy:** {accuracy:.2f}')
    st.write('**Classification Report:**')
    st.dataframe(pd.DataFrame(report).transpose())
    st.write('**Confusion Matrix:**')
    st.dataframe(pd.DataFrame(confusion))

    # Perform statistical tests
    stat_results = perform_statistical_tests((X, y))

    # Display statistical test results
    st.header('Statistical Test Results')
    st.write(pd.DataFrame(stat_results).transpose())

    # Process and predict image if uploaded
    if uploaded_image:
        image_path = f'temp/{uploaded_image.name}'
        with open(image_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())
        image = process_image(image_path)
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)
        st.header('Image Prediction')
        st.write(f'The uploaded image is predicted as class: {predicted_class[0]}')
