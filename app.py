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
        {"title": "Time Series with RNN", "link": "https://encord.com/blog/time-series-predictions-with-recurrent-neural-networks"},
        {"title": "Building and Leading High-Performing Product Teams", "link": "https://productledalliance.com/building-and-leading-high-performing-product-teams"},
        {"title": "Meta-Learning and Few-Shot Learning: Adapting Deep Learning Models to Learn from Limited Data", "link": "https://datascience.salon/meta-learning-and-few-shot-learning-adapting-deep-learning-models-to-learn-from-limited-data"},
        {"title": "Glossary terms", "link": "https://encord.com/glossary"},
        {"title": "User-Centric Product Design: Keeping Customer Needs at the Core of Development", "link": "https://collato.com/user-centric-product-design-how-to-understand-user-needs"},
        {"title": "Sudoku Solver", "link": "https://encord.com/blog/sudoku-solver-cv-project"},
        {"title": "Image thresholding", "link": "https://encord.com/blog/image-thresholding-image-processing"},
        {"title": "Text detection via CV", "link": "https://encord.com/blog/realtime-text-recognition-with-tesseract-using-opencv"},
        {"title": "Guide to Supervised learning", "link": "https://encord.com/blog/mastering-supervised-learning-a-comprehensive-guide"},
        {"title": "Logistic regression", "link": "https://encord.com/blog/what-is-logistic-regression"},
        {"title": "Federated learning", "link": "https://datascience.salon/federated-learning-for-privacy-preserving-ai"},
        {"title": "Creating a Live Virtual Pen Project with OpenCV", "link": "https://medium.com/p/ed477487b75f/edit"},
        {"title": "LabelBox Alternatives", "link": "https://encord.com/blog/labelbox-alternatives"},
        {"title": "Ensemble Learning", "link": "https://encord.com/blog/what-is-ensemble-learning"},
        {"title": "Challenges for AI in e-commerce", "link": "https://datascience.salon/challenges-for-ai-in-e-commerce"},
        {"title": "HRIS Data Security", "link": "https://www.vaulthrsolutions.com/blogs/essentials-of-hris-data-security"},
        {"title": "Exploring effective HRIS", "link": "https://www.vaulthrsolutions.com/blogs/exploring-effective-hris"}
    ]
    
    for article in articles:
        st.write(f"### {article['title']}")
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

# Profile Summary
def profile_summary():
    st.image("profile_picture.jpg", width=150)  # Add your profile picture
    st.write("### Soham Sharma")
    st.write("AI Researcher | Product Manager | Customer Success Enthusiast")
    st.write("Social Media Links: [LinkedIn](https://www.linkedin.com/in/sohamsharma) | [GitHub](https://github.com/sohamsharma)")
    st.markdown("---")

# Main function to run the Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Introduction", "Predictive Analytics", "Articles", "Showcase Cards", "Customer Journey Mapping and Optimization"])

    profile_summary()  # Add profile summary at the top

    if page == "Introduction":
        introduction_page()
    elif page == "Predictive Analytics":
        predictive_analytics_page()
    elif page == "Articles":
        articles_page()
    elif page == "Showcase Cards":
        showcase_cards_page()
    elif page == "Customer Journey Mapping and Optimization":
        customer_journey_page()

if __name__ == "__main__":
    main()
