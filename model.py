import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_model(data):
    # Placeholder for the model
    model_playbooks = Sequential([
        Dense(10, input_shape=(5,), activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_playbooks.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Prepare data for training
    X = np.array([data['Support Tickets'], data['Feedback Score'], data['Purchase Amount'], data['Tenure'], data['Needs Engagement']]).T
    y = np.array(data['Usage Frequency'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model_playbooks.fit(X_train_scaled, y_train, epochs=50, verbose=0, validation_data=(X_test_scaled, y_test))

    return model_playbooks, scaler

def predict_needs(model_playbooks, scaler, support_tickets, feedback_score, purchase_amount, tenure, needs_engagement):
    try:
        input_data = np.array([[support_tickets, feedback_score, purchase_amount, tenure, needs_engagement]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model_playbooks.predict(input_data_scaled)[0]
        return prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
