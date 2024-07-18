import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys

# Add the directory containing model.py to Python path if necessary
sys.path.insert(0, './')  # Adjust the path as needed

from model import train_model, simulate_customer_data, predict_needs, generate_email_templates

# Page configuration
st.set_page_config(page_title="Customer Success App", layout="wide")

# Function to simulate customer data
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

# Predictive Analytics Page
def predictive_analytics_page():
    st.title("Predictive Analytics")

    num_customers = st.number_input("Number of Customers", min_value=1, value=100)
    customer_data = simulate_customer_data(num_customers)

    st.subheader("Simulated Customer Data")
    st.dataframe(customer_data)

    st.subheader("Prediction")
    selected_customer = st.selectbox("Select Customer", customer_data['CustomerID'])
    selected_data = customer_data[customer_data['CustomerID'] == selected_customer].iloc[0]

    prediction = predict_needs(
        selected_data['Support Tickets'],
        selected_data['Feedback Score'],
        selected_data['Purchase Amount'],
        selected_data['Tenure (Months)'],
        np.random.randint(0, 2)  # Example engagement data
    )

    st.write(f"Predicted Usage Frequency: {prediction}")

    st.subheader("Generate Email Template")
    selected_template = st.selectbox("Select Template", ['Template 1', 'Template 2'])
    email_template = generate_email_templates(prediction, selected_template)
    st.code(email_template, language='markdown')

# Introduction Page
def introduction_page():
    st.title("Introduction")
    st.write("""
        Welcome to the Customer Success App! This app is designed to help you manage and optimize customer engagement.
        - **Predictive Analytics**: Simulate customer data and predict engagement needs.
        - **Articles**: Explore articles and resources on customer success.
        - **Showcase Cards**: View case studies and projects.
        - **Customer Journey Mapping and Optimization**: Learn about mapping customer journeys and optimizing them.

        Use the navigation menu to explore different sections of the app.
    """)

# List of Articles Page
def articles_page():
    st.title("Articles")
    st.write("Explore our collection of articles on customer success:")
    
    articles = [
        {"title": "How to Improve Customer Engagement", "summary": "Learn strategies for increasing customer engagement.", "link": "https://example.com/article1"},
        {"title": "Optimizing Customer Journeys", "summary": "A guide to mapping and optimizing customer journeys.", "link": "https://example.com/article2"},
        {"title": "Using AI for Customer Success", "summary": "Discover how AI can enhance customer success initiatives.", "link": "https://example.com/article3"},
    ]
    
    for article in articles:
        st.write(f"### {article['title']}")
        st.write(f"{article['summary']}")
        st.write(f"[Read more]({article['link']})")

# Showcase Cards Page
def showcase_cards_page():
    st.title("Showcase Cards")
    st.write("Here are some key projects and case studies:")
    
    projects = [
        {"title": "Project A", "description": "A comprehensive case study on improving customer retention.", "link": "https://example.com/projectA", "image": "https://via.placeholder.com/150"},
        {"title": "Project B", "description": "An innovative approach to customer journey mapping.", "link": "https://example.com/projectB", "image": "https://via.placeholder.com/150"},
        {"title": "Project C", "description": "Using AI to predict customer needs and enhance satisfaction.", "link": "https://example.com/projectC", "image": "https://via.placeholder.com/150"},
    ]
    
    for project in projects:
        st.image(project["image"], width=150)
        st.write(f"### {project['title']}")
        st.write(f"{project['description']}")
        st.write(f"[Learn more]({project['link']})")

# Customer Journey Mapping and Optimization Page
def customer_journey_page():
    st.title("Customer Journey Mapping and Optimization")
    st.write("""
        Customer journey mapping is a powerful tool for understanding and optimizing the experiences customers have with your brand.
        
        **Steps to create a customer journey map:**
        1. Define your objectives.
        2. Gather customer data.
        3. Identify customer touchpoints.
        4. Outline the customer journey.
        5. Analyze and optimize.

        **Examples:**
        - Mapping the onboarding process for new customers.
        - Optimizing the support ticket resolution journey.
        - Enhancing the renewal and upsell journey.

        By understanding and optimizing these journeys, you can improve customer satisfaction and drive better business outcomes.
    """)

# Main app logic
def main():
    st.sidebar.title("Navigation")
    pages = {
        "Introduction": introduction_page,
        "Predictive Analytics": predictive_analytics_page,
        "Articles": articles_page,
        "Showcase Cards": showcase_cards_page,
        "Customer Journey Mapping and Optimization": customer_journey_page
    }
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()

if __name__ == "__main__":
    main()
