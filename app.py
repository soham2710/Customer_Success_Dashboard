import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample Data for Customer Journey Mapping
journey_data = {
    'Stage': ['Onboarding', 'Product Usage', 'Support', 'Renewal'],
    'Customers': [100, 80, 50, 40],
    'Satisfaction': [4.5, 4.0, 3.5, 4.2]
}
journey_df = pd.DataFrame(journey_data)

# Sample Data for Customer Success Playbooks
playbooks_data = {
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Needs': np.random.randint(0, 2, 100)
}
playbooks_df = pd.DataFrame(playbooks_data)

# Navigation Sidebar
st.sidebar.title("Customer Success Dashboard")
selection = st.sidebar.selectbox("Select a Section", ["Customer Journey Mapping", "Customer Success Playbooks"])

# Customer Journey Mapping Section
if selection == "Customer Journey Mapping":
    st.title("Customer Journey Mapping and Optimization")

    st.header("Customer Journey Map")
    st.bar_chart(journey_df.set_index('Stage')['Customers'])

    st.header("Satisfaction Analysis")
    st.line_chart(journey_df.set_index('Stage')['Satisfaction'])

    st.header("Data Analytics")
    st.write("Identify bottlenecks and opportunities for improvement.")

    st.subheader("Machine Learning Model")
    # Simple TensorFlow Model for Prediction
    model_journey = Sequential([
        Dense(10, input_shape=(1,), activation='relu'),
        Dense(1)
    ])
    model_journey.compile(optimizer='adam', loss='mse')

    # Sample Data for Prediction
    X_journey = np.array([1, 2, 3, 4])  # Stages encoded as integers
    y_journey = np.array([4.5, 4.0, 3.5, 4.2])  # Satisfaction Scores

    model_journey.fit(X_journey, y_journey, epochs=100, verbose=0)

    predictions_journey = model_journey.predict(X_journey)
    journey_df['Predicted Satisfaction'] = predictions_journey

    st.line_chart(journey_df.set_index('Stage')[['Satisfaction', 'Predicted Satisfaction']])

# Customer Success Playbooks Section
if selection == "Customer Success Playbooks":
    st.title("Customer Success Playbooks Using Predictive Analytics")

    st.header("Customer Data")
    st.write(playbooks_df.head())

    st.header("Predictive Model")

    # Preprocess Data
    X_playbooks = playbooks_df[['Feature1', 'Feature2']]
    y_playbooks = playbooks_df['Needs']
    X_train, X_test, y_train, y_test = train_test_split(X_playbooks, y_playbooks, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build TensorFlow Model
    model_playbooks = Sequential([
        Dense(10, input_shape=(2,), activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_playbooks.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train Model
    model_playbooks.fit(X_train_scaled, y_train, epochs=50, verbose=0, validation_data=(X_test_scaled, y_test))

    st.subheader("Model Performance")
    loss, accuracy = model_playbooks.evaluate(X_test_scaled, y_test)
    st.write(f"Test Accuracy: {accuracy:.2f}")

    st.subheader("Predict Needs")
    feature1 = st.sidebar.slider("Feature1", 0.0, 1.0, 0.5)
    feature2 = st.sidebar.slider("Feature2", 0.0, 1.0, 0.5)
    prediction = model_playbooks.predict(np.array([[feature1, feature2]]))
    st.write(f"Predicted Need: {'Yes' if prediction > 0.5 else 'No'}")

    st.subheader("Dynamic Playbook")
    st.write("Suggest specific actions based on predicted needs.")
    if prediction > 0.5:
        st.write("Suggested Action: Offer a personalized demo and follow-up call.")
    else:
        st.write("Suggested Action: Send a thank you email and a feedback survey.")
