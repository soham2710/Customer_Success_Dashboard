import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import train_model, simulate_customer_data, predict_needs, generate_dummy_journey_data

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

# Introduction Page
def introduction_page():
    st.title("Introduction to Customer Success Dashboard App")
    st.image("https://via.placeholder.com/600x400", use_column_width=True)
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
    articles_data = {
        "Article 1": "Link to Article 1",
        "Article 2": "Link to Article 2",
        "Article 3": "Link to Article 3"
    }
    df_articles = pd.DataFrame(list(articles_data.items()), columns=["Article Title", "Link"])
    st.table(df_articles)

# Websites Page
def websites_page():
    st.title("Websites")
    st.markdown("""
                Showcase of websites I have worked on:
                """)
    
    # Example card elements (replace with actual content)
    websites_data = [
        {
            "image": "https://via.placeholder.com/300x200",
            "description": "A brief description of Website 1.",
            "link": "[Link to Website 1]"
        },
        {
            "image": "https://via.placeholder.com/300x200",
            "description": "A brief description of Website 2.",
            "link": "[Link to Website 2]"
        }
    ]
    
    for website in websites_data:
        st.markdown(f"""
                    ### Website
                    [![Website]({website['image']})]({website['link']})
                    - Description: {website['description']}
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
        "Websites": websites_page
    }
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection]()

if __name__ == "__main__":
    main()
