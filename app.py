import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import requests
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

# Ensure to call introduction_page() in your main function or wherever you need this content

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
        {"title": "Project A", "description": "Description of Project A.", "image": "https://via.placeholder.com/150", "link": "https://example.com/project-a"},
        {"title": "Project B", "description": "Description of Project B.", "image": "https://via.placeholder.com/150", "link": "https://example.com/project-b"},
        {"title": "Project C", "description": "Description of Project C.", "image": "https://via.placeholder.com/150", "link": "https://example.com/project-c"}
    ]

    for project in projects:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(project["image"], width=150)
        with col2:
            st.write(f"**{project['title']}**")
            st.write(project["description"])
            st.write(f"[View Project]({project['link']})")

import plotly.graph_objects as go
# Function to simulate customer journey data
def simulate_customer_journey_data(num_customers=100):
    data = {
        'CustomerID': np.arange(1, num_customers + 1),
        'Stage': np.random.choice(['Awareness', 'Consideration', 'Purchase', 'Retention', 'Advocacy'], num_customers),
        'Interaction': np.random.choice(['Email', 'Ad', 'Social Media', 'In-Store', 'Support Call'], num_customers),
        'Feedback Score': np.round(np.random.uniform(1.0, 5.0, num_customers), 1),
        'Pain Point': np.random.choice(['Price', 'Quality', 'Customer Service', 'Delivery', 'None'], num_customers),
        'Resolution Time (Days)': np.random.randint(1, 15, num_customers)  # Example field
    }
    return pd.DataFrame(data)

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

    num_customers = st.number_input("Number of Customers", min_value=1, value=100)
    journey_data = simulate_customer_journey_data(num_customers)

    st.subheader("Simulated Customer Journey Data")
    st.dataframe(journey_data)

    st.subheader("Customer Journey Analysis")
    selected_stage = st.selectbox("Select Stage", journey_data['Stage'].unique())
    stage_data = journey_data[journey_data['Stage'] == selected_stage]
    st.write(f"### Insights for the {selected_stage} Stage")
    st.write(f"Number of Customers: {stage_data.shape[0]}")
    st.write(f"Average Feedback Score: {stage_data['Feedback Score'].mean():.2f}")
    st.write(f"Most Common Pain Point: {stage_data['Pain Point'].mode()[0]}")
    if 'Resolution Time (Days)' in stage_data.columns:
        st.write(f"Average Resolution Time: {stage_data['Resolution Time (Days)'].mean():.2f} days")

    st.subheader("Visualizations")
    
    # Visualization of Customer Feedback Scores by Stage
    fig_feedback = px.box(
        journey_data, 
        x='Stage', 
        y='Feedback Score', 
        title='Customer Feedback Scores by Stage'
    )
    st.plotly_chart(fig_feedback)

    # Visualization of Resolution Time by Stage
    fig_resolution = px.histogram(
        journey_data,
        x='Resolution Time (Days)',
        color='Stage',
        title='Distribution of Resolution Time by Stage'
    )
    st.plotly_chart(fig_resolution)

    # Optimization Suggestions
    st.subheader("Optimization Suggestions")
    st.write("""
        Based on the simulated data, here are some suggestions for optimizing the customer journey:
        
        - **Awareness Stage**: Focus on improving initial engagement through personalized outreach and targeted marketing.
        - **Consideration Stage**: Enhance support and provide detailed product information to address common pain points.
        - **Purchase Stage**: Streamline the checkout process and offer timely assistance to reduce cart abandonment.
        - **Retention Stage**: Implement proactive communication strategies to keep customers engaged and satisfied.
        - **Advocacy Stage**: Encourage positive feedback and referrals through loyalty programs and exceptional customer service.

        Regularly updating and analyzing customer journey maps can help in continuously improving customer experiences and achieving better outcomes.
    """)




# Define the profile_summary function
def profile_summary():
    st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)  # Replace with your profile image
    st.sidebar.write("**Name:** Soham Sharma")
    st.sidebar.write("**Title:** Customer Success and Product Management Expert")
    st.sidebar.write("**Summary:** A passionate AI and product management professional with extensive experience in data analytics, customer success strategies, and AI-driven solutions. Expertise in creating impactful AI projects and optimizing customer experiences.")

# File download
    url = "https://github.com/soham2710/Customer_Success_Dashboard/raw/main/Customer%20Success%20Resume.pdf"
    response = requests.get(url)
    file_content = response.content
    
    st.download_button(
        label="Download Resume",
        data=file_content,
        file_name="Customer_Success_Resume.pdf",
        mime="application/pdf"
    )
# Main function to run the Streamlit app
def main():
    st.sidebar.title("Navigation")
    profile_summary()  # Add profile summary at the top

    options = [
        "Introduction",
        "Predictive Analytics",
        "Articles",
        "Showcase Cards",
        "Customer Journey Mapping and Optimization"
    ]
    
    selected_option = st.sidebar.selectbox("Choose a Page", options)

    if selected_option == "Introduction":
        introduction_page()
    elif selected_option == "Predictive Analytics":
        predictive_analytics_page()
    elif selected_option == "Articles":
        articles_page()
    elif selected_option == "Showcase Cards":
        showcase_cards_page()
    elif selected_option == "Customer Journey Mapping and Optimization":
        customer_journey_page()

if __name__ == "__main__":
    main()








