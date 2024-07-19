import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Define your email templates
email_templates = {
    "Welcome Email": {
        "Subject": "Welcome to [Company Name]!",
        "Body": """
        Hi [Customer Name],

        Welcome to [Company Name]! We are excited to have you on board. If you have any questions or need assistance, feel free to reach out to us. 

        Best regards,
        The [Company Name] Team
        """
    },
    "New Feature Announcement": {
        "Subject": "Check Out Our New Feature!",
        "Body": """
        Hi [Customer Name],

        We are thrilled to announce a new feature in [Product/Service Name]! This feature will help you [briefly describe the benefit]. 

        Explore it today and let us know your feedback!

        Best,
        The [Company Name] Team
        """
    },
    "Engagement Follow-Up": {
        "Subject": "We Miss You at [Company Name]!",
        "Body": """
        Hi [Customer Name],

        We noticed that it's been a while since you last engaged with us. We would love to hear your thoughts and help you with anything you need.

        Looking forward to your return!

        Cheers,
        The [Company Name] Team
        """
    },
    "Customer Feedback Request": {
        "Subject": "Your Feedback Matters to Us",
        "Body": """
        Hi [Customer Name],

        We value your feedback and would appreciate if you could take a few minutes to share your thoughts on our recent [product/service]. Your input will help us improve and serve you better.

        Thank you,
        The [Company Name] Team
        """
    },
    "Special Offer": {
        "Subject": "Exclusive Offer Just for You!",
        "Body": """
        Hi [Customer Name],

        As a valued customer, we're excited to offer you an exclusive [discount/offer]. Don't miss out on this special deal!

        Use code [OFFER CODE] at checkout.

        Best regards,
        The [Company Name] Team
        """
    },
    "Reminder Email": {
        "Subject": "Reminder: [Action Required]",
        "Body": """
        Hi [Customer Name],

        Just a friendly reminder to [action required]. We want to make sure you don’t miss out on [benefit or important date].

        Thank you,
        The [Company Name] Team
        """
    },
    "Thank You Email": {
        "Subject": "Thank You for Your Purchase!",
        "Body": """
        Hi [Customer Name],

        Thank you for your recent purchase of [Product/Service]. We hope you are satisfied with your purchase. 

        If you need any assistance, please do not hesitate to contact us.

        Warm regards,
        The [Company Name] Team
        """
    },
    "Churn Prevention": {
        "Subject": "We’re Here to Help",
        "Body": """
        Hi [Customer Name],

        We noticed that you haven't been using [Product/Service] recently. Is there anything we can assist you with? We value your business and want to ensure you're getting the most out of our services.

        Please let us know how we can help.

        Best,
        The [Company Name] Team
        """
    },
    "Renewal Reminder": {
        "Subject": "Your Subscription is About to Expire",
        "Body": """
        Hi [Customer Name],

        This is a reminder that your subscription to [Product/Service] is about to expire on [Expiration Date]. 

        Renew now to continue enjoying uninterrupted service.

        Thank you,
        The [Company Name] Team
        """
    },
    "Customer Appreciation": {
        "Subject": "We Appreciate You!",
        "Body": """
        Hi [Customer Name],

        We just wanted to take a moment to say thank you for being a loyal customer. We appreciate your support and look forward to continuing to serve you.

        Best wishes,
        The [Company Name] Team
        """
    }
}

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

    return model

# Function to train email template prediction model
def train_email_template_model(data):
    # Encode the target variable (Email Template)
    label_encoder = LabelEncoder()
    data['Email Template Encoded'] = label_encoder.fit_transform(data['Email Template'])

    X = data[['Churn Risk', 'NPS Score', 'Retention Rate (%)']]
    y = data['Email Template Encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model and label encoder
    with open('email_template_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    return model

# Function to predict the email template
def suggest_email_template(churn_risk, nps_score, retention_rate):
    # Load the trained model and label encoder
    with open('email_template_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Predict the email template
    features = np.array([[churn_risk, nps_score, retention_rate]])
    predicted_index = model.predict(features)[0]
    
    # Decode the predicted template
    predicted_template = label_encoder.inverse_transform([predicted_index])[0]
    
    return predicted_template

# Function to suggest the email template using the model
def suggest_email_template(churn_risk, nps_score, retention_rate):
    features = np.array([[churn_risk, nps_score, retention_rate]])
    predicted_index = model.predict(features)[0]
    predicted_template = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_template

# Function to simulate customer data
def simulate_customer_data(num_customers):
    np.random.seed(42)
    ages = np.random.randint(18, 70, size=num_customers)
    incomes = np.random.randint(20000, 120000, size=num_customers)
    credit_scores = np.random.randint(300, 850, size=num_customers)
    purchases = np.random.randint(1, 20, size=num_customers)
    churn_risks = np.random.randint(0, 2, size=num_customers)
    nps_scores = np.random.randint(0, 100, size=num_customers)
    retention_rates = np.random.uniform(50, 100, size=num_customers)

    data = pd.DataFrame({
        'CustomerID': range(1, num_customers + 1),
        'Age': ages,
        'Annual Income (USD)': incomes,
        'Credit Score': credit_scores,
        'Previous Purchases': purchases,
        'Churn Risk': churn_risks,
        'NPS Score': nps_scores,
        'Retention Rate (%)': retention_rates
    })

    return data
