import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import requests
from model import train_model, predict_needs, generate_email_templates, simulate_predictive_analytics_data

# Add the directory containing model.py to Python path if necessary
sys.path.insert(0, './')  # Adjust the path as needed

# Page configuration
st.set_page_config(page_title="Customer Success App", layout="wide")


def simulate_customer_data(num_customers):
    np.random.seed(42)
    ages = np.random.randint(18, 70, size=num_customers)
    incomes = np.random.randint(20000, 120000, size=num_customers)
    credit_scores = np.random.randint(300, 850, size=num_customers)
    purchases = np.random.randint(1, 20, size=num_customers)
    churn_risks = np.random.randint(0, 2, size=num_customers)
    nps_scores = np.random.randint(0, 100, size=num_customers)
    retention_rates = np.random.uniform(50, 100, size=num_customers)

    data = pd.DataFrame({
        'CustomerID': range(1, num_customers + 1),
        'Age': ages,
        'Annual Income (USD)': incomes,
        'Credit Score': credit_scores,
        'Previous Purchases': purchases,
        'Churn Risk': churn_risks,
        'NPS Score': nps_scores,
        'Retention Rate (%)': retention_rates
    })

    return data

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from model import select_email_template  # Import the email template selection function

# Load the trained model and label encoder
@st.cache_resource
def load_model():
    with open('email_template_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model()

# Function to suggest the email template using the model
def suggest_email_template(churn_risk, nps_score, retention_rate):
    features = np.array([[churn_risk, nps_score, retention_rate]])
    predicted_index = model.predict(features)[0]
    predicted_template = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_template

# Function to simulate customer data
def simulate_customer_data(num_customers):
    np.random.seed(42)
    ages = np.random.randint(18, 70, size=num_customers)
    incomes = np.random.randint(20000, 120000, size=num_customers)
    credit_scores = np.random.randint(300, 850, size=num_customers)
    purchases = np.random.randint(1, 20, size=num_customers)
    churn_risks = np.random.randint(0, 2, size=num_customers)
    nps_scores = np.random.randint(0, 100, size=num_customers)
    retention_rates = np.random.uniform(50, 100, size=num_customers)

    data = pd.DataFrame({
        'CustomerID': range(1, num_customers + 1),
        'Age': ages,
        'Annual Income (USD)': incomes,
        'Credit Score': credit_scores,
        'Previous Purchases': purchases,
        'Churn Risk': churn_risks,
        'NPS Score': nps_scores,
        'Retention Rate (%)': retention_rates
    })

    return data

# Streamlit UI
def predictive_analytics_page():
    st.title("Customer Success Management - Predictive Analytics")

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
    analytics_data = simulate_customer_data(num_customers)

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
        labels={'Churn Risk': 'Churn Risk (0: Low, 1: High)'},
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_age_churn, use_container_width=True)

    # Visualization of Annual Income vs. Credit Score
    fig_income_credit = px.scatter(
        analytics_data,
        x='Annual Income (USD)',
        y='Credit Score',
        color='Churn Risk',
        title='Annual Income vs. Credit Score',
        labels={'Churn Risk': 'Churn Risk (0: Low, 1: High)'},
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_income_credit, use_container_width=True)

    # Visualization of Churn Risk Distribution
    fig_churn_distribution = px.histogram(
        analytics_data,
        x='Churn Risk',
        title='Distribution of Churn Risk',
        labels={'Churn Risk': 'Churn Risk (0: Low, 1: High)'}
    )
    st.plotly_chart(fig_churn_distribution, use_container_width=True)

    # User Selection for Detailed Analysis
    st.subheader("Select Customer for Detailed Analysis")
    selected_customer = st.selectbox("Select Customer", analytics_data['CustomerID'])
    selected_data = analytics_data[analytics_data['CustomerID'] == selected_customer].iloc[0]

    st.write(f"**Selected Customer Data:**")
    st.write(f"Age: {selected_data['Age']}")
    st.write(f"Annual Income: ${selected_data['Annual Income (USD)']}")
    st.write(f"Credit Score: {selected_data['Credit Score']}")
    st.write(f"Previous Purchases: {selected_data['Previous Purchases']}")
    st.write(f"Churn Risk: {'High Risk' if selected_data['Churn Risk'] == 1 else 'Low Risk'}")
    st.write(f"NPS Score: {selected_data['NPS Score']}")
    st.write(f"Retention Rate: {selected_data['Retention Rate (%)']}%")

    # Generate Email Templates
    st.subheader("Generate Email Template")
    email_templates = select_email_template(
        selected_data['Churn Risk'],
        selected_data['NPS Score'],
        selected_data['Retention Rate (%)']
    )

    st.write("**Suggested Email Templates:**")
    for template in email_templates:
        st.write(f"**Subject:** {template['Subject']}")
        st.write(f"**Body:** {template['Body']}")
        st.write("---")

# Main function to control the Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predictive Analytics", "Suggest Email Template"])

    if page == "Predictive Analytics":
        predictive_analytics_page()
    elif page == "Suggest Email Template":
        st.title("Email Template Suggestion")
        
        churn_risk = st.slider("Churn Risk (0 or 1)", 0, 1, 0)
        nps_score = st.slider("NPS Score", 0, 100, 50)
        retention_rate = st.slider("Retention Rate (%)", 0, 100, 75)
        
        if st.button("Suggest Relevant Email Template"):
            suggested_template = suggest_email_template(churn_risk, nps_score, retention_rate)
            st.subheader("Suggested Email Template")
            st.write(f"**Template:** {suggested_template}")

            # Display the full email template content
            email_templates = {
                "Welcome Email": {
                    "Subject": "Welcome to [Company Name]!",
                    "Body": """
                    Hi [Customer Name],

                    Welcome to [Company Name]! We are excited to have you on board. If you have any questions or need assistance, feel free to reach out to us. 

                    Best regards,
                    The [Company Name] Team
                    """
                },
                "New Feature Announcement": {
                    "Subject": "Check Out Our New Feature!",
                    "Body": """
                    Hi [Customer Name],

                    We are thrilled to announce a new feature in [Product/Service Name]! This feature will help you [briefly describe the benefit]. 

                    Explore it today and let us know your feedback!

                    Best,
                    The [Company Name] Team
                    """
                },
                "Engagement Follow-Up": {
                    "Subject": "We Miss You at [Company Name]!",
                    "Body": """
                    Hi [Customer Name],

                    We noticed that it's been a while since you last engaged with us. We would love to hear your thoughts and help you with anything you need.

                    Looking forward to your return!

                    Cheers,
                    The [Company Name] Team
                    """
                },
                "Customer Feedback Request": {
                    "Subject": "Your Feedback Matters to Us",
                    "Body": """
                    Hi [Customer Name],

                    We value your feedback and would appreciate if you could take a few minutes to share your thoughts on our recent [product/service]. Your input will help us improve and serve you better.

                    Thank you,
                    The [Company Name] Team
                    """
                },
                "Special Offer": {
                    "Subject": "Exclusive Offer Just for You!",
                    "Body": """
                    Hi [Customer Name],

                    As a valued customer, we're excited to offer you an exclusive [discount/offer]. Don't miss out on this special deal!

                    Use code [OFFER CODE] at checkout.

                    Best regards,
                    The [Company Name] Team
                    """
                },
                "Reminder Email": {
                    "Subject": "Reminder: [Action Required]",
                    "Body": """
                    Hi [Customer Name],

                    Just a friendly reminder to [action required]. We want to make sure you don’t miss out on [benefit or important date].

                    Thank you,
                    The [Company Name] Team
                    """
                },
                "Thank You Email": {
                    "Subject": "Thank You for Your Purchase!",
                    "Body": """
                    Hi [Customer Name],

                    Thank you for your recent purchase of [Product/Service]. We hope you are satisfied with your purchase. 

                    If you need any assistance, please do not hesitate to contact us.

                    Warm regards,
                    The [Company Name] Team
                    """
                },
                "Churn Prevention": {
                    "Subject": "We’re Here to Help",
                    "Body": """
                    Hi [Customer Name],

                    We noticed that you haven't been using [Product/Service] recently. Is there anything we can assist you with? We value your business and want to ensure you're getting the most out of our services.

                    Please let us know how we can help.

                    Best,
                    The [Company Name] Team
                    """
                },
                "Renewal Reminder": {
                    "Subject": "Your Subscription is About to Expire",
                    "Body": """
                    Hi [Customer Name],

                    This is a reminder that your subscription to [Product/Service] is about to expire on [Expiration Date]. 

                    Renew now to continue enjoying uninterrupted service.

                    Thank you,
                    The [Company Name] Team
                    """
                },
                "Customer Appreciation": {
                    "Subject": "We Appreciate You!",
                    "Body": """
                    Hi [Customer Name],

                    We just wanted to take a moment to say thank you for being a loyal customer. We appreciate your support and look forward to continuing to serve you.

                    Best wishes,
                    The [Company Name] Team
                    """
                }
            }

            # Display the full email content
            if suggested_template:
                template_details = email_templates.get(suggested_template)
                if template_details:
                    st.write(f"**Subject:** {template_details['Subject']}")
                    st.write(f"**Body:** {template_details['Body']}")
                    st.write("---")

if __name__ == "__main__":
    main()


'''# Predictive Analytics Page
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
    analytics_data = simulate_customer_data(num_customers)

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
        labels={'Churn Risk': 'Churn Risk (0: Low, 1: High)'},
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_age_churn, use_container_width=True)

    # Visualization of Annual Income vs. Credit Score
    fig_income_credit = px.scatter(
        analytics_data,
        x='Annual Income (USD)',
        y='Credit Score',
        color='Churn Risk',
        title='Annual Income vs. Credit Score',
        labels={'Churn Risk': 'Churn Risk (0: Low, 1: High)'},
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_income_credit, use_container_width=True)

    # Visualization of Churn Risk Distribution
    fig_churn_distribution = px.histogram(
        analytics_data,
        x='Churn Risk',
        title='Distribution of Churn Risk',
        labels={'Churn Risk': 'Churn Risk (0: Low, 1: High)'}
    )
    st.plotly_chart(fig_churn_distribution, use_container_width=True)

    # User Selection for Analysis
    st.subheader("Select Customer for Detailed Analysis")
    selected_customer = st.selectbox("Select Customer", analytics_data['CustomerID'])
    selected_data = analytics_data[analytics_data['CustomerID'] == selected_customer].iloc[0]

    st.write(f"**Selected Customer Data:**")
    st.write(f"Age: {selected_data['Age']}")
    st.write(f"Annual Income: ${selected_data['Annual Income (USD)']}")
    st.write(f"Credit Score: {selected_data['Credit Score']}")
    st.write(f"Previous Purchases: {selected_data['Previous Purchases']}")
    st.write(f"Churn Risk: {'High Risk' if selected_data['Churn Risk'] == 1 else 'Low Risk'}")
    st.write(f"NPS Score: {selected_data['NPS Score']}")
    st.write(f"Retention Rate: {selected_data['Retention Rate (%)']}%")

    # Generate Email Templates
    st.subheader("Generate Email Template")
    email_templates = generate_email_templates(
        selected_data['Age'],
        selected_data['Annual Income (USD)'],
        selected_data['Credit Score'],
        selected_data['Churn Risk'],
        selected_data['NPS Score'],
        selected_data['Retention Rate (%)']
    )
    
    for i, template in enumerate(email_templates, 1):
        st.write(f"### Email Template {i}")
        st.write(template)

    # Optimization Suggestions
    st.subheader("Optimization Suggestions")
    st.write("""
        Based on the predictive analytics data, here are some suggestions for optimizing customer engagement:

        - **High Churn Risk Customers**: Implement retention strategies such as personalized offers or proactive support.
        - **Low Credit Score**: Consider offering financial advice or improved payment options to enhance customer satisfaction.
        - **Frequent Purchasers**: Develop loyalty programs to reward frequent shoppers and encourage repeat business.
        - **Income-Based Targeting**: Customize marketing strategies based on income levels to better meet customer needs.
        - **NPS Score**: Address negative feedback and work to improve customer satisfaction. Use positive feedback to enhance marketing efforts.

        By leveraging predictive analytics, you can tailor your approach to effectively address customer needs and improve business outcomes.
    """)'''


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


