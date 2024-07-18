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
    st.write("Welcome to the Customer Success App!")

# List of Articles Page
def articles_page():
    st.title("Articles")
    st.write("List of Articles will be shown here.")

# Showcase Cards Page
def showcase_cards_page():
    st.title("Showcase Cards")
    st.write("Showcase Cards will be displayed here.")

# Customer Journey Mapping and Optimization Page
def customer_journey_page():
    st.title("Customer Journey Mapping and Optimization")
    st.write("Customer Journey Mapping and Optimization content will be shown here.")

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
