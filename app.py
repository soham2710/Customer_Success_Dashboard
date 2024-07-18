import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import requests
from model import train_model, simulate_customer_data, predict_needs, generate_email_templates, simulate_predictive_analytics_data, select_email_template

# Add the directory containing model.py to Python path if necessary
sys.path.insert(0, './')  # Adjust the path as needed

# Page configuration
st.set_page_config(page_title="Customer Success App", layout="wide")

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
        model_data = simulate_predictive_analytics_data(num_customers)
        trained_model = train_model(model_data)
        st.success("Model trained successfully!")

    if st.button("Train Email Template Model"):
        email_template_data = pd.DataFrame({
            'Churn Risk': np.random.choice([0, 1], num_customers),
            'NPS Score': np.random.randint(-100, 101, num_customers),
            'Retention Rate (%)': np.random.uniform(50, 100, num_customers),
            'Email Template': np.random.choice(email_templates.keys(), num_customers)
        })
        trained_email_model = train_email_template_model(email_template_data)
        st.success("Email Template Model trained successfully!")

def email_template_suggestion_page():
    st.title("Email Template Suggestion")
    churn_risk = st.selectbox("Churn Risk", [0, 1])
    nps_score = st.slider("NPS Score", -100, 100)
    retention_rate = st.slider("Retention Rate (%)", 0, 100)

    if st.button("Suggest Relevant Email Template"):
        suggested_template = suggest_email_template(churn_risk, nps_score, retention_rate)
        st.write(f"**Suggested Email Template:** {suggested_template}")
        st.write(email_templates[suggested_template])

# Streamlit app
if __name__ == "__main__":
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Predictive Analytics", "Email Template Suggestion"])

    if page == "Predictive Analytics":
        predictive_analytics_page()
    elif page == "Email Template Suggestion":
        email_template_suggestion_page()


#--------------------------------

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

#Articles Page
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

    # Create DataFrame
    df = pd.DataFrame(articles)

    # Convert DataFrame to HTML
    html = df.to_html(escape=False, index=False, classes="table table-striped")
    
    # Add custom CSS for center alignment
    st.markdown("""
        <style>
        .table {
            width: 100%;
            border-collapse: collapse;
        }
        .table th, .table td {
            padding: 10px;
            text-align: center;
        }
        .table th {
            background-color: #000000;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display DataFrame with clickable links
    st.markdown(html, unsafe_allow_html=True)

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

#----------------------------------------------------------------

#Function to simulate customer journey data
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
        title='Customer Feedback Scores by Stage',
        color='Stage',
        category_orders={'Stage': ['Awareness', 'Consideration', 'Purchase', 'Retention', 'Advocacy']}
    )
    fig_feedback.update_layout(boxmode='group')
    st.plotly_chart(fig_feedback, use_container_width=True)

    # Visualization of Resolution Time by Stage
    fig_resolution = px.histogram(
        journey_data,
        x='Resolution Time (Days)',
        color='Stage',
        title='Distribution of Resolution Time by Stage',
        barmode='overlay',
        marginal='rug'
    )
    fig_resolution.update_layout(
        xaxis_title='Resolution Time (Days)',
        yaxis_title='Count'
    )
    st.plotly_chart(fig_resolution, use_container_width=True)

    # Trend Analysis: Average Feedback Score Over Time
    feedback_trend = journey_data.groupby('Stage').agg({'Feedback Score': ['mean', 'std']}).reset_index()
    feedback_trend.columns = ['Stage', 'Average Feedback Score', 'Feedback Score Std Dev']
    fig_feedback_trend = px.line(
        feedback_trend,
        x='Stage',
        y='Average Feedback Score',
        error_y='Feedback Score Std Dev',
        title='Trend of Average Feedback Score by Stage',
        markers=True
    )
    fig_feedback_trend.update_layout(
        xaxis_title='Customer Journey Stage',
        yaxis_title='Average Feedback Score'
    )
    st.plotly_chart(fig_feedback_trend, use_container_width=True)

    # Pain Points Distribution
    fig_pain_points = px.pie(
        journey_data,
        names='Pain Point',
        title='Distribution of Pain Points',
        color='Pain Point'
    )
    st.plotly_chart(fig_pain_points, use_container_width=True)

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



#----------------------------------------------------------------

def add_custom_css():
    st.markdown(
        """
        <style>
        /* Sidebar styling */
        .css-1d391kg {  /* Sidebar container */
            background-color: #f5f5f5;  /* Background color */
            padding: 20px;  /* Padding */
        }
        .css-1d391kg .css-1v0mbdj {  /* Sidebar title */
            color: #333;  /* Text color */
            font-size: 20px;  /* Font size */
            font-weight: bold;  /* Font weight */
        }
        .css-1d391kg .css-1m8z4rf {  /* Sidebar links */
            color: #007bff;  /* Link color */
            text-decoration: none;  /* Remove underline */
        }
        .css-1d391kg .css-1m8z4rf:hover {  /* Hover effect */
            text-decoration: underline;
        }
        /* Custom button styling */
        .css-1kp0b9p {
            background-color: #007bff;  /* Button background color */
            color: white;  /* Button text color */
        }
        .css-1kp0b9p:hover {
            background-color: #0056b3;  /* Button hover color */
        }
        </style>
        """, unsafe_allow_html=True
    )

def profile_summary():
    st.sidebar.title("Profile Summary")

    # Placeholder for profile image
    st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)  # Placeholder image

    st.sidebar.write(
        """
        **Profile Summary**

        I am an experienced AI and data science professional with over 8 years in the IT industry. My expertise spans product management, technical training, and customer success. I hold a Bachelor's degree in Aircraft Maintenance Engineering and am pursuing a PG Diploma in Applied Statistics. Skilled in Python, AWS, and AI tools, I focus on delivering innovative, scalable solutions to drive business growth.
        """
    )
    
    # Add a download button for the resume
    resume_url = "https://github.com/soham2710/Customer_Success_Dashboard/raw/main/Customer%20Success%20Resume.pdf"
    st.sidebar.download_button(
        label="Download Resume",
        data=requests.get(resume_url).content,
        file_name="Customer_Success_Resume.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    # Add social media links with icons
    st.sidebar.title("Connect with Me")
    social_links = {
        "LinkedIn": "https://www.linkedin.com/in/your-profile/",
        "GitHub": "https://github.com/your-profile/"
    }

    icons = {
        "LinkedIn": "🔗",
        "GitHub": "🐙"
    }

    for platform, url in social_links.items():
        st.sidebar.markdown(f"{icons[platform]} [ {platform} ]({url})", unsafe_allow_html=True)

def main():
    add_custom_css()  # Apply custom CSS
    st.sidebar.title("Navigation")
    
    profile_summary()  # Add profile summary, image, resume, and social media links

    # Define the page selection
    page = st.sidebar.selectbox("Choose a Page", [
        "Introduction",
        "Predictive Analytics",
        "Articles",
        "Showcase Cards",
        "Customer Journey Mapping and Optimization"
    ])

    # Page selection logic
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


