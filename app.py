import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sample Data
data = {
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Needs': np.random.randint(0, 2, 100)
}
df = pd.DataFrame(data)

# Streamlit App
st.title("Customer Success Playbooks Using Predictive Analytics")

st.header("Customer Data")
st.write(df.head())

st.header("Predictive Model")

# Preprocess Data
X = df[['Feature1', 'Feature2']]
y = df['Needs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build TensorFlow Model
model = Sequential([
    Dense(10, input_shape=(2,), activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train_scaled, y_train, epochs=50, verbose=0, validation_data=(X_test_scaled, y_test))

st.subheader("Model Performance")
loss, accuracy = model.evaluate(X_test_scaled, y_test)
st.write(f"Test Accuracy: {accuracy:.2f}")

st.subheader("Predict Needs")
features = st.sidebar.slider("Features", 0.0, 1.0, (0.5, 0.5))
prediction = model.predict(np.array([features]))
st.write(f"Predicted Need: {'Yes' if prediction > 0.5 else 'No'}")

st.subheader("Dynamic Playbook")
st.write("Suggest specific actions based on predicted needs.")
if prediction > 0.5:
    st.write("Suggested Action: Offer a personalized demo and follow-up call.")
else:
    st.write("Suggested Action: Send a thank you email and a feedback survey.")
