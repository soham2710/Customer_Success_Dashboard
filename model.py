from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def train_model(data):
    X = data[['Support Tickets', 'Feedback Score', 'Purchase Amount', 'Tenure', 'Needs Engagement']]
    y = data['Usage Frequency']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_playbooks = Sequential([
        Dense(10, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model_playbooks.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model_playbooks.fit(X_train_scaled, y_train, epochs=50, verbose=0, validation_data=(X_test_scaled, y_test))

    return model_playbooks, scaler

def simulate_customer_data(num_customers):
    data = {
        'CustomerID': np.arange(1, num_customers + 1),
        'Usage Frequency': np.random.choice(['Daily', 'Weekly', 'Monthly'], num_customers),
        'Support Tickets': np.random.randint(0, 10, num_customers),
        'Feedback Score': np.round(np.random.uniform(2.0, 5.0, num_customers), 1),
        'Purchase Amount': np.random.randint(100, 1000, num_customers),
        'Tenure (Months)': np.random.randint(1, 36, num_customers)
    }
    return pd.DataFrame(data)

def generate_dummy_journey_data():
    stages = ['Awareness', 'Consideration', 'Purchase', 'Retention', 'Advocacy']
    customers = np.random.randint(20, 100, size=len(stages))
    satisfaction = np.round(np.random.uniform(3.0, 5.0, size=len(stages)), 1)
    data = {
        'Stage': stages,
        'Customers': customers,
        'Satisfaction': satisfaction
    }
    return pd.DataFrame(data)
