# model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to train the predictive model
def train_model(data):
    X = data[['Support Tickets', 'Feedback Score', 'Purchase Amount', 'Tenure (Months)', 'Needs Engagement']]
    y = data['Usage Frequency']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    return model

# Function to simulate customer data
def simulate_customer_data(num_customers):
    data = {
        'CustomerID': np.arange(1, num_customers + 1),
        'Usage Frequency': np.random.choice(['Daily', 'Weekly', 'Monthly'], num_customers),
        'Support Tickets': np.random.randint(0, 10, num_customers),
        'Feedback Score': np.round(np.random.uniform(2.0, 5.0, num_customers), 1),
        'Purchase Amount': np.random.randint(100, 1000, num_customers),
        'Tenure (Months)': np.random.randint(1, 36, num_customers),
        'Needs Engagement': np.random.randint(0, 2, num_customers)
    }
    return pd.DataFrame(data)

# Function to predict customer needs
def predict_needs(support_tickets, feedback_score, purchase_amount, tenure, needs_engagement):
    # Dummy implementation, replace with actual predictive model
    data = {
        'Support Tickets': [support_tickets],
        'Feedback Score': [feedback_score],
        'Purchase Amount': [purchase_amount],
        'Tenure (Months)': [tenure],
        'Needs Engagement': [needs_engagement]
    }
    df = pd.DataFrame(data)

    # Example: Use a trained model to predict usage frequency
    model = train_model(simulate_customer_data(100))  # Replace with actual data
    prediction = model.predict(df)

    # Convert numerical prediction to category
    if prediction == 0:
        return 'Daily'
    elif prediction == 1:
        return 'Weekly'
    elif prediction == 2:
        return 'Monthly'
    else:
        return 'Unknown'  # Handle unexpected cases
