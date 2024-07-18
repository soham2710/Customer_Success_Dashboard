import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to simulate predictive analytics data
def simulate_predictive_analytics_data(num_customers=100):
    data = {
        'CustomerID': np.arange(1, num_customers + 1),
        'Age': np.random.randint(18, 70, num_customers),
        'Annual Income (USD)': np.random.randint(30000, 150000, num_customers),
        'Credit Score': np.random.randint(300, 850, num_customers),
        'Previous Purchases': np.random.randint(0, 20, num_customers),
        'Churn Risk': np.random.choice([0, 1], num_customers),  # 0: Low Risk, 1: High Risk
        'NPS Score': np.random.randint(-100, 101, num_customers),  # Net Promoter Score
        'Retention Rate (%)': np.random.uniform(50, 100, num_customers)  # Retention Rate
    }
    return pd.DataFrame(data)

# Function to train the predictive model
def train_model(data):
    X = data[['Age', 'Annual Income (USD)', 'Credit Score', 'Previous Purchases', 'NPS Score', 'Retention Rate (%)']]
    y = data['Churn Risk']

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
        'Age': np.random.randint(18, 70, num_customers),
        'Annual Income (USD)': np.random.randint(30000, 150000, num_customers),
        'Credit Score': np.random.randint(300, 850, num_customers),
        'Previous Purchases': np.random.randint(0, 20, num_customers),
        'Churn Risk': np.random.choice([0, 1], num_customers),  # 0: Low Risk, 1: High Risk
        'NPS Score': np.random.randint(-100, 101, num_customers),  # Net Promoter Score
        'Retention Rate (%)': np.random.uniform(50, 100, num_customers)  # Retention Rate
    }
    return pd.DataFrame(data)

# Function to predict customer needs
def predict_needs(age, annual_income, credit_score, previous_purchases, nps_score, retention_rate):
    # Dummy implementation, replace with actual predictive model
    data = {
        'Age': [age],
        'Annual Income (USD)': [annual_income],
        'Credit Score': [credit_score],
        'Previous Purchases': [previous_purchases],
        'NPS Score': [nps_score],
        'Retention Rate (%)': [retention_rate]
    }
    df = pd.DataFrame(data)

    # Example: Use a trained model to predict churn risk
    model = train_model(simulate_predictive_analytics_data(100))  # Replace with actual data
    prediction = model.predict(df)

    # Convert numerical prediction to category
    return 'High Risk' if prediction[0] == 1 else 'Low Risk'

# Function to generate email templates based on prediction
def generate_email_templates(age, annual_income, credit_score, churn_risk, nps_score, retention_rate):
    templates = {
        'High Risk': {
            'Template 1': f"Subject: Urgent: Addressing Your Concerns\n\nDear Customer,\n\nWe noticed that you may be at risk of churning. Your NPS Score is {nps_score} and Retention Rate is {retention_rate}%. We would like to offer personalized support...",
            'Template 2': f"Subject: We're Here to Help!\n\nHello,\n\nAs a valued customer, we want to ensure that you are satisfied with our service. Your feedback is important to us. Let's discuss how we can improve your experience..."
        },
        'Low Risk': {
            'Template 1': f"Subject: Thank You for Your Continued Engagement\n\nDear Customer,\n\nWe appreciate your loyalty and want to thank you for your continued engagement. Your NPS Score is {nps_score} and Retention Rate is {retention_rate}%. Here’s how we can enhance your experience further...",
            'Template 2': f"Subject: Your Satisfaction Matters\n\nHello,\n\nThank you for being a valued customer. We are committed to providing you with the best experience. Your feedback is important to us, and we look forward to serving you better..."
        }
    }

    return templates.get(churn_risk, {}).get('Template 1', "No template available for the selected options.")
