import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

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
        'Churn Risk': np.random.choice([0, 1], num_customers),
        'NPS Score': np.random.randint(-100, 101, num_customers),
        'Retention Rate (%)': np.random.uniform(50, 100, num_customers)
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
    label_encoder = LabelEncoder()
    data['Email Template Encoded'] = label_encoder.fit_transform(data['Email Template'])

    X = data[['Churn Risk', 'NPS Score', 'Retention Rate (%)']]
    y = data['Email Template Encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open('email_template_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    return model

# Function to suggest the email template
def suggest_email_template(churn_risk, nps_score, retention_rate):
    with open('email_template_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    features = np.array([[churn_risk, nps_score, retention_rate]])
    predicted_index = model.predict(features)[0]
    predicted_template = label_encoder.inverse_transform([predicted_index])[0]
    
    return predicted_template

# Function to simulate customer data
def simulate_customer_data(num_customers):
    data = {
        'CustomerID': np.arange(1, num_customers + 1),
        'Age': np.random.randint(18, 70, num_customers),
        'Annual Income (USD)': np.random.randint(30000, 150000, num_customers),
        'Credit Score': np.random.randint(300, 850, num_customers),
        'Previous Purchases': np.random.randint(0, 20, num_customers),
        'Churn Risk': np.random.choice([0, 1], num_customers),
        'NPS Score': np.random.randint(-100, 101, num_customers),
        'Retention Rate (%)': np.random.uniform(50, 100, num_customers)
    }
    return pd.DataFrame(data)

# Function to predict customer needs (dummy implementation)
def predict_needs(age, annual_income, credit_score, previous_purchases, nps_score, retention_rate):
    data = {
        'Age': [age],
        'Annual Income (USD)': [annual_income],
        'Credit Score': [credit_score],
        'Previous Purchases': [previous_purchases],
        'NPS Score': [nps_score],
        'Retention Rate (%)': [retention_rate]
    }
    df = pd.DataFrame(data)

    model = train_model(simulate_predictive_analytics_data(100))
    prediction = model.predict(df)

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

# Function to select an email template based on churn risk and scores
def select_email_template(churn_risk, nps_score, retention_rate):
    if churn_risk == 0:
        if nps_score > 50:
            return email_templates["Welcome Email"], email_templates["New Feature Announcement"], email_templates["Special Offer"], email_templates["Thank You Email"], email_templates["Customer Appreciation"]
        else:
            return email_templates["Engagement Follow-Up"], email_templates["Customer Feedback Request"], email_templates["Reminder Email"]
    else:
        if retention_rate > 75:
            return email_templates["Engagement Follow-Up"], email_templates["Special Offer"], email_templates["Reminder Email"]
        else:
            return email_templates["Churn Prevention"], email_templates["Renewal Reminder"]
