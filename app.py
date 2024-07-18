import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from model import train_model, simulate_customer_data, predict_needs, generate_email_templates, generate_dummy_journey_data

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

# Predictive Analytics Playbooks Page
def predictive_analytics_page():
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
    
    # Call predict_needs and print prediction
    prediction = predict_needs(support_tickets, feedback_score, purchase_amount, tenure, needs_engagement_binary)
    print(f"Prediction: {prediction}")  # Add this print statement
    
    # Handle prediction type if necessary (example: convert to float)
    if isinstance(prediction, np.ndarray):
        prediction = prediction[0]  # Assuming the first element is the prediction
    
    st.write(f"Predicted Usage Frequency: {prediction}")
    
    # Select email template based on prediction
    st.subheader("Select Email Template")
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



# Introduction Page
def introduction_page():
    st.title("Introduction to Customer Success Dashboard App")
    st.image("your_profile_picture.jpg", use_column_width=True)
    st.markdown("""
                Welcome to the Customer Success Dashboard App! This application leverages predictive analytics
                to enhance customer success efforts. It includes tools for simulating customer data, developing
                dynamic playbooks, and visualizing customer journey maps.

                ### Features:
                - **Customer Data Simulation:** Simulate customer data based on usage frequency, support tickets, feedback score, purchase amount, and tenure.
                - **Predictive Analytics Playbooks:** Use predictive models to forecast customer needs and suggest appropriate engagement strategies.
                - **Customer Journey Mapping:** Visualize customer journey maps and optimize touchpoints for better customer experiences.
                """)

    # Social sharing options (example buttons)
    st.subheader("Share This App")
    st.markdown("""
                - LinkedIn
                - Twitter
                - Facebook
                """)

# List of Articles Page
def articles_page():
    st.title("List of Articles")
    st.markdown("""
                Here is a list of articles I have written:
                - Article 1: [Link to Article 1]
                - Article 2: [Link to Article 2]
                - Article 3: [Link to Article 3]
                """)

# Showcase Cards Page
def showcase_cards_page():
    st.title("Showcase Cards")
    st.markdown("""
                Showcase of websites I have worked on:
                """)
    
    # Example cards (replace with actual content)
    st.markdown("""
                ### Website 1
                ![Website 1](website1_image.jpg)
                - Description: A brief description of Website 1.
                - [Link to Website 1]

                ### Website 2
                ![Website 2](website2_image.jpg)
                - Description: A brief description of Website 2.
                - [Link to Website 2]
                """)

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
    fig = px.scatter(df_journey, x='Stage', y='Customers', size='Satisfaction', 
                     hover_data=['Satisfaction'], color='Stage',
                     title='Customer Journey Map')
    fig.update_layout(xaxis_title='Stage', yaxis_title='Number of Customers')
    st.plotly_chart(fig)

    # Optimization strategies
    st.subheader("Optimization Strategies")
    st.markdown("""
                - **Journey Mapping:** Create visual maps highlighting key touchpoints.
                - **Data Analytics:** Analyze data at each touchpoint to identify bottlenecks and pain points.
                - **Continuous Improvement:** Implement changes based on data insights and customer feedback.
                """)

# Main app logic
def main():
    st.sidebar.title("Navigation")
    pages = {
        "Introduction": introduction_page,
        "Customer Journey Mapping": customer_journey_page,
        "Predictive Analytics Playbooks": predictive_analytics_page,
        "List of Articles": articles_page,
        "Showcase Cards": showcase_cards_page
    }
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection]()

if __name__ == "__main__":
    main()
