import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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

# Customer Journey Mapping and Optimization Page
def customer_journey_page():
    st.title("Customer Journey Mapping and Optimization")
    st.markdown("""
                This page visualizes customer journey maps and optimizes touchpoints for better customer experiences.
                """)
    
    # Simulated customer data
    num_customers = 50
    df_customers = simulate_customer_data(num_customers)
    
    # Display customer data table
    st.subheader("Customer Data")
    st.dataframe(df_customers)
    
    # Customer journey map
    st.subheader("Customer Journey Map")
    fig = px.scatter(df_customers, x='Tenure (Months)', y='Purchase Amount', color='Usage Frequency', size='Support Tickets', hover_data=['Feedback Score'])
    st.plotly_chart(fig)

    # Optimization strategies
    st.subheader("Optimization Strategies")
    st.markdown("""
                - **Journey Mapping:** Create visual maps highlighting key touchpoints.
                - **Data Analytics:** Analyze data at each touchpoint to identify bottlenecks and pain points.
                - **Continuous Improvement:** Implement changes based on data insights and customer feedback.
                """)

# Customer Success Playbooks Using Predictive Analytics Page
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

# Main app logic to switch between pages
def main():
    menu = ["Customer Journey Mapping", "Predictive Analytics Playbooks"]
    choice = st.sidebar.selectbox("Select Page", menu)

    if choice == "Customer Journey Mapping":
        customer_journey_page()
    elif choice == "Predictive Analytics Playbooks":
        predictive_analytics_page()

if __name__ == "__main__":
    main()
