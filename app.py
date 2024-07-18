import pandas as pd
import streamlit as st
import numpy as np
from model import train_model, predict_needs

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

# Function to generate dummy data for customer journey mapping
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

# Function to generate email templates based on prediction label and selected action
def generate_email_templates(prediction_label, selected_action):
    templates = {
        'Yes': {
            'Offer a product demo': """Dear [Customer], As a valued customer, we would like to offer you a product demo to help you get the most out of our product. Please let us know a convenient time for you.""",
            'Schedule a follow-up call': """Dear [Customer], We would like to schedule a follow-up call to discuss your experience with our product and address any concerns you may have. Please let us know a suitable time for you.""",
            'Invite to customer success webinar': """Dear [Customer], We are excited to invite you to our upcoming customer success webinar, where we will share tips and best practices for using our product effectively. Please join us on [date].""",
            'Send promotional offers': """Dear [Customer], We have some exciting promotional offers just for you! Check your account for exclusive discounts and offers."""
        },
        'No': {
            'Send a feedback survey': """Dear [Customer], Thank you for your recent interaction with us. We would appreciate your feedback to help us improve our services. Please take a moment to complete our feedback survey.""",
            'Thank you for your feedback': """Dear [Customer], Thank you for your valuable feedback. We appreciate your input and will use it to enhance your experience with us."""
        }
    }

    # Check if prediction_label and selected_action are valid keys in templates
    if prediction_label in templates and selected_action in templates[prediction_label]:
        return templates[prediction_label][selected_action]
    else:
        return f"No email template found for '{selected_action}' under '{prediction_label}' scenario."

# Customer Journey Mapping and Optimization Page
def customer_journey_page():
    st.title("Customer Journey Mapping and Optimization")
    st.markdown("""
                This page visualizes customer journey maps and optimizes touchpoints for better customer experiences.
                """)
    
    # Generate dummy customer journey data
    df_journey = generate_dummy_journey_data()
    
    # Display customer journey data table
    st.subheader("Customer Journey Data")
    st.dataframe(df_journey)
    
    # Customer journey map
    st.subheader("Customer Journey Map")
    # Include your visualization code here using Plotly or other libraries

    # Optimization strategies
    st.subheader("Optimization Strategies")
    st.markdown("""
                - **Journey Mapping:** Create visual maps highlighting key touchpoints.
                - **Data Analytics:** Analyze data at each touchpoint to identify bottlenecks and pain points.
                - **Continuous Improvement:** Implement changes based on data insights and customer feedback.
                """)

# Predictive Analytics Playbooks Using Predictive Analytics Page
def predictive_analytics_page(model_playbooks, scaler):
    st.title("Customer Success Playbooks Using Predictive Analytics")
    st.markdown("""
                This page develops dynamic playbooks using predictive analytics to enhance customer success efforts.
                """)
    
    # Placeholder for predictive analytics content
    st.subheader("Predictive Analytics Content")
    st.markdown("""
                - **Predictive Modeling:** Use historical data to predict customer needs and issues.
                - **Dynamic Playbooks:** Create playbooks that adapt based on predictive insights.
                - **Personalized Engagement:** Tailor support and engagement based on predicted customer behavior.
                """)
    
    # Predictive model simulation
    st.subheader("Predict Customer Needs")
    support_tickets = st.slider("Support Tickets", 0, 10, 5)
    feedback_score = st.slider("Feedback Score", 2.0, 5.0, 3.5, 0.1)
    purchase_amount = st.slider("Purchase Amount", 100, 1000, 500)
    tenure = st.slider("Tenure (Months)", 1, 60, 30)
    needs_engagement = st.radio("Needs Engagement", options=['Yes', 'No'])
    
    needs_engagement_binary = 1 if needs_engagement == 'Yes' else 0
    prediction = predict_needs(model_playbooks, scaler, support_tickets, feedback_score, purchase_amount, tenure, needs_engagement_binary)
    
    st.write(f"Predicted Usage Frequency: {prediction}")
    
    # Select email template based on prediction
    st.subheader("Select Email Template")
    if prediction is not None:
        if prediction > 0.5:
            selected_template = st.selectbox("Choose Email Template", 
                                             ["Offer a product demo", "Schedule a follow-up call", 
                                              "Invite to customer success webinar", "Send promotional offers"])
        else:
            selected_template = st.selectbox("Choose Email Template", 
                                             ["Send a feedback survey", "Thank you for your feedback"])
        
        # Display selected email template
        st.subheader("Email Template")
        st.code(generate_email_templates('Yes' if prediction > 0.5 else 'No', selected_template), language='markdown')
    else:
        st.warning("Please adjust the input sliders to predict customer needs.")

# Introduction to Customer Success Dashboard Page
def introduction_page():
    st.title("Introduction to Customer Success Dashboard")
    st.markdown("""
                Welcome to the Customer Success Dashboard application! This dashboard leverages predictive analytics to enhance customer success efforts. Here, you can visualize customer journey maps, optimize touchpoints, and develop dynamic playbooks tailored to customer needs.
                """)
    st.markdown("""
                This application is divided into two main sections:
                - **Customer Journey Mapping and Optimization:** Visualizes customer journey maps, identifies optimization strategies, and suggests improvements based on data insights.
                - **Predictive Analytics Playbooks:** Utilizes predictive modeling to predict customer needs, recommends personalized engagement strategies, and generates email templates based on predictions.
                """)
    st.markdown("""
                To get started, use the navigation sidebar to explore different sections of the dashboard.
                """)

# Main app logic
def main():
    st.sidebar.title("Navigation")
    pages = {
        "Introduction": introduction_page,
        "Customer Journey Mapping": customer_journey_page,
        "Predictive Analytics Playbooks": lambda: predictive_analytics_page(model_playbooks, scaler)
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection]()

# Initialize model and scaler
if __name__ == "__main__":
    data = simulate_customer_data(100)  # Simulate 100 customers
    model_playbooks, scaler = train_model(data)
    main()
