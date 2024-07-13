import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import (
    load_and_preprocess_data,
    preprocess_data,
    train_and_evaluate_models,
    perform_statistical_tests,
    process_image,
    build_vgg16,
)

# Function to load user-uploaded data
def load_user_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# Sidebar options
st.sidebar.title('Model Options')
uploaded_file = st.sidebar.file_uploader('Upload your CSV file', type=['csv'])
uploaded_image = st.sidebar.file_uploader('Upload an Image of Iris Flower', type=['jpg', 'jpeg', 'png'])
model_type = st.sidebar.selectbox(
    'Choose a model',
    [
        'Logistic Regression',
        'Naive Bayes',
        'Support Vector Machine',
        'K-Nearest Neighbors',
        'Artificial Neural Network',
        'Recurrent Neural Network',
        'VGG16',
    ],
)

# Load and preprocess data
if uploaded_file:
    X, y = load_user_data(uploaded_file)
else:
    X, y = load_and_preprocess_data()
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Train and evaluate selected model
model, accuracy, report, confusion = train_and_evaluate_models(model_type, X_train, y_train, X_test, y_test)

# Display model performance
st.header('Model Performance')
st.write(f'**Model Type:** {model_type}')
st.write(f'**Accuracy:** {accuracy:.2f}')
st.write('**Classification Report:**')
st.dataframe(pd.DataFrame(report).transpose())
st.write('**Confusion Matrix:**')
st.dataframe(confusion)

# Perform statistical tests
stat_results = perform_statistical_tests((X, y))

# Display statistical test results
st.header('Statistical Test Results')
for test_name, result in stat_results.items():
    st.subheader(test_name)
    for key, value in result.items():
        st.write(f'{key}: {value}')

# Process and predict image if uploaded
if uploaded_image and model_type == 'VGG16':
    image_path = f'./temp/{uploaded_image.name}'
    with open(image_path, 'wb') as f:
        f.write(uploaded_image.getbuffer())
    image = process_image(image_path)
    vgg_model = build_vgg16()
    predictions = vgg_model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    st.header('Image Prediction')
    st.write(f'The uploaded image is predicted as class: {predicted_class[0]}')

# Display model comparison
if uploaded_file:
    st.header('Model Comparison')
    all_models = [
        'Logistic Regression',
        'Naive Bayes',
        'Support Vector Machine',
        'K-Nearest Neighbors',
        'Artificial Neural Network',
        'Recurrent Neural Network',
        'VGG16',
    ]
    comparison_results = []
    for mdl in all_models:
        mdl_model, mdl_accuracy, mdl_report, _ = train_and_evaluate_models(mdl, X_train, y_train, X_test, y_test)
        comparison_results.append({
            'Model': mdl,
            'Accuracy': mdl_accuracy,
        })
    comparison_df = pd.DataFrame(comparison_results)
    st.dataframe(comparison_df)
    best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    st.write(f'**Best Model:** {best_model["Model"]} with accuracy {best_model["Accuracy"]:.2f}')
