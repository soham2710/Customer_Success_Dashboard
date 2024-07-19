import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import pickle
from io import BytesIO

# Page configuration should be the very first Streamlit command
st.set_page_config(page_title="Customer Success App", layout="wide")

# Define URLs for the pickle files
email_template_model_url = "https://github.com/soham2710/Customer_Success_Dashboard/raw/main/email_template_model.pkl"
label_encoder_url = "https://github.com/soham2710/Customer_Success_Dashboard/raw/main/label_encoder.pkl"

def load_pickle_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return pickle.load(BytesIO(response.content))

# Load the models
try:
    email_template_model = load_pickle_from_url(email_template_model_url)
    label_encoder = load_pickle_from_url(label_encoder_url)
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")

# Import functions from model.py
from model import (
    simulate_predictive_analytics_data,
    train_model,
    train_email_template_model,
    suggest_email_template
)

def predictive_analytics_page():
    st.title("Predictive Analytics")
    st.write("""
        Predictive analytics uses historical data and statistical algorithms to predict future outcomes. It helps businesses make data-driven decisions and anticipate customer needs.

        **Key Features:**
        - **Age**: Customer's age.
        - **Annual Income (USD)**: Customer's yearly income.
        - **Credit Score**: Customer's credit score.
        - **Previous Purchases**: Number of past purchases.
        - **Churn Risk**: Likelihood of customer churn (0: Low Risk, 1: High Risk).
        - **NPS Score**: Net Promoter Score, indicating customer satisfaction.
        - **Retention Rate (%)**: Percentage of customers retained over a period.

        Predictive analytics can help in identifying high-risk customers, targeting potential upsell opportunities, and optimizing marketing strategies.
    """)

    num_customers = st.number_input("Number of Customers", min_value=1, value=100)
    analytics_data = simulate_predictive_analytics_data(num_customers)

    st.subheader("Simulated Predictive Analytics Data")
    st.dataframe(analytics_data)

    st.subheader("Predictive Analysis")

    # Visualization of Churn Risk by Age
    fig_age_churn = px.scatter(
        analytics_data,
        x='Age',
        y='Churn Risk',
        color='Churn Risk',
        title='Churn Risk by Age',
        labels={'Churn Risk': 'Churn Risk (0: Low Risk, 1: High Risk)'}
    )
    st.plotly_chart(fig_age_churn)

    # Visualization of NPS Score by Retention Rate
    fig_nps_retention = px.scatter(
        analytics_data,
        x='NPS Score',
        y='Retention Rate (%)',
        color='Churn Risk',
        title='NPS Score vs. Retention Rate',
        labels={'Retention Rate (%)': 'Retention Rate (%)'}
    )
    st.plotly_chart(fig_nps_retention)

    # Model Training Section
    if st.button("Train Model"):
        try:
            model_data = simulate_predictive_analytics_data(num_customers)
            trained_model = train_model(model_data)
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"Error training model: {e}")

    if st.button("Train Email Template Model"):
        try:
            email_template_data = pd.DataFrame({
                'Churn Risk': np.random.choice([0, 1], num_customers),
                'NPS Score': np.random.randint(-100, 101, num_customers),
                'Retention Rate (%)': np.random.uniform(50, 100, num_customers),
                'Email Template': np.random.choice(email_templates.keys(), num_customers)
            })
            trained_email_model = train_email_template_model(email_template_data)
            st.success("Email Template Model trained successfully!")
        except Exception as e:
            st.error(f"Error training email template model: {e}")

def email_template_suggestion_page():
    st.title("Email Template Suggestion")
    churn_risk = st.selectbox("Churn Risk", [0, 1])
    nps_score = st.slider("NPS Score", -100, 100)
    retention_rate = st.slider("Retention Rate (%)", 0, 100)

    if st.button("Suggest Relevant Email Template"):
        try:
            suggested_template = suggest_email_template(churn_risk, nps_score, retention_rate)
            st.write(f"**Suggested Email Template:** {suggested_template}")
            st.write(email_templates[suggested_template])
        except Exception as e:
            st.error(f"Error suggesting email template: {e}")

def introduction_page():
    st.title("Introduction")

    st.header("Purpose of This Project")
    st.write("""
        This project aims to enhance customer success through the use of advanced analytics and optimization techniques. 
        The primary goals are to better understand customer behavior, improve engagement strategies, and optimize the customer journey.
        Each section of this application is designed to provide insights into different aspects of customer success and 
        predictive analytics to help in making informed decisions.
    """)

    st.header("Section Descriptions")

    st.subheader("Customer Journey Mapping and Optimization")
    st.write("""
        This section focuses on mapping out the customer journey and identifying key touchpoints where optimization can be implemented. 
        The purpose of this page is to visualize and analyze the different stages a customer goes through from initial contact to 
        post-purchase interactions. 

        Key Sections:
        - **Customer Journey Map**: Provides a visual representation of the customer journey, highlighting different phases and touchpoints.
        - **Optimization Insights**: Offers insights and recommendations on how to improve customer experience at various stages.
        - **Engagement Metrics**: Displays metrics related to customer engagement and provides suggestions for enhancing interactions.

        This page is designed to help businesses understand and optimize their customer interactions, ultimately leading to improved satisfaction and retention.
    """)

    st.subheader("Predictive Analytics")
    st.write("""
        The Predictive Analytics page is designed to forecast key metrics and trends using historical data. The purpose is to 
        provide actionable insights that can guide decision-making and strategic planning.

        **Purpose and Plan:**
        - **Forecasting Metrics**: Uses historical data to predict future trends, such as sales performance, customer behavior, or engagement levels.
        - **App Features**: The app includes various predictive models that allow users to select different parameters and view predictions based on different scenarios.
        - **Usage Guide**: Users can input historical data, select relevant features, and view predictive results. The app also provides visualizations to help interpret the forecasts.

        This page aims to equip users with the tools needed to anticipate future trends and make data-driven decisions to enhance their business strategies.
    """)

def articles_page():
    st.title("Articles")
    st.write("Explore our collection of articles on customer success:")

    # List of articles
    articles = [
        {"title": "Agile Approach to Data Strategy", "link": "https://datascience.salon/an-agile-approach-to-data-strategy"},
        {"title": "Anomaly detection in Machine learning", "link": "https://roundtable.datascience.salon/using-deep-learning-for-anomaly-detection-in-cybersecurity"},
        {"title": "Machine learning podcasts", "link": "https://datascience.salon/machine-learning-podcasts-top-shows-for-deepening-your-understanding"},
        {"title": "CNN", "link": "https://datascience.salon/convolutional-neural-networks-overview"},
        {"title": "Fundamentals of Product Development Roadmap", "link": "https://collato.com/the-fundamentals-of-a-product-development-roadmap"},
        {"title": "How to Reduce Bias in Machine learning", "link": "https://datascience.salon/reducing-bias-in-machine-learning"},
        {"title": "Incorporating Ethical Considerations into Product Development", "link": "https://productledalliance.com/incorporating-ethical-considerations-into-product-development"},
        {"title": "Leveraging ChatGPT for Product Managers", "link": "https://productledalliance.com/leveraging-chatgpt-for-product-managers-enhancing-productivity-and-collaboration"},
        {"title": "Correlation and Regression Analysis: Exploring Relationships in Data", "link": "https://roundtable.datascience.salon/correlation-and-regression-analysis-exploring-relationships-in-data"},
        {"title": "Beyond the Matrix: Advanced Prioritization Techniques for Product Managers", "link": "https://productledalliance.com/beyond-the-matrix-advanced-prioritization-techniques-for-product-managers"},
        {"title": "Building Minimum Viable Products (MVPs) that Matter", "link": "https://productledalliance.com/building-minimum-viable-products-that-matter"},
        {"title": "Time Series with RNN", "link": "https://encord.com/blog/time-series-predictions"}
    ]

    for article in articles:
        st.write(f"- [{article['title']}]({article['link']})")

def contact_page():
    st.title("Contact")
    st.write("""
        If you have any questions or feedback, please reach out to us through the following channels:
        
        - **Email:** support@customersuccessapp.com
        - **Twitter:** [@CustomerSuccessApp](https://twitter.com/CustomerSuccessApp)
        - **LinkedIn:** [Customer Success App](https://www.linkedin.com/company/customersuccessapp)
    """)

def main():
    st.sidebar.title("Navigation")
    options = ["Introduction", "Predictive Analytics", "Email Template Suggestion", "Articles", "Contact"]
    choice = st.sidebar.radio("Go to", options)

    # Display the selected page
    if choice == "Introduction":
        introduction_page()
    elif choice == "Predictive Analytics":
        predictive_analytics_page()
    elif choice == "Email Template Suggestion":
        email_template_suggestion_page()
    elif choice == "Articles":
        articles_page()
    elif choice == "Contact":
        contact_page()

    # Display profile picture
    profile_picture_url = "https://github.com/soham2710/Customer_Success_Dashboard/raw/main/profile_picture.jpg"
    st.sidebar.image(profile_picture_url, use_column_width=True, caption="Profile Picture")

    # Download resume button
    resume_url = "https://github.com/soham2710/Customer_Success_Dashboard/raw/main/resume.pdf"
    st.sidebar.markdown(f"[Download Resume]({resume_url})", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
