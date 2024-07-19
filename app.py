import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from io import BytesIO

def introduction_page():
    st.title("Introduction")

    # Introduction and profile summary
    st.header("Welcome to CSM Dashboard")
    st.write("Welcome to CSM web app! Here you will find information about this project articles.")
    
    st.header("About Me")
    st.write("Hello! I am Soham Sharma, and I am excited to share this project with you. Below is a brief summary of my professional background and the purpose of this project.")
    st.write("""
    - **Professional Summary:** Customer Success Manager with 8+ years of experience in the IT industry, specializing in customer success, relationship management, and customer engagement. Proven track record in managing customer lifecycles, driving customer retention, and delivering user-centric solutions to enhance customer satisfaction. Successfully executed multiple projects at Ziplyne, improving customer experience and operational efficiency. Strong proficiency in CRM tools, customer success metrics, and compliance with SOC 2, ISO, and GDPR standards. Passionate about leveraging innovative technologies to drive business growth and maximize customer satisfaction.
    - **Project Purpose:** This project aims to enhance customer success management by leveraging predictive analytics and providing actionable insights through various features in the dashboard.
    """)

    # Future improvement options
    st.header("Future Improvement Options")
    st.write("""
    We have several plans for future enhancements of this dashboard, including:
    - **Enhanced Data Visualization:** Adding more interactive graphs and charts to visualize customer data effectively.
    - **Advanced Predictive Analytics:** Implementing more sophisticated models for better accuracy in customer engagement predictions.
    - **Integration with Other Tools:** Incorporating features to integrate with various CRM tools for a seamless experience.
    """)

    # Contact section
    st.header("Connect with Me")
    st.write("""
    You can get in touch with me through the following channels:
    - **Email:** sohamnsharma@gmail.com
    - **Phone:** +91-9730084626
    - **Address:** Indore, MP, India
    """)
https://github.com/soham2710/Customer_Success_Dashboard/edit/main/app.py
    # Resume download button
    resume_url = "https://github.com/soham2710/Customer_Success_Dashboard/main/Customer Success Resume.pdf"  # Replace with your actual resume URL
    st.download_button(
        label="Download Resume",
        data=resume_url,
        file_name="Customer Success Resume.pdf",
        help="Click to download my resume"
    )

# Articles PAGE
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

    # Render HTML in Streamlit
    st.markdown(html, unsafe_allow_html=True)

# Customer Journey Mapping Page
def customer_journey_mapping_page():
    st.title("Customer Journey Mapping")
    st.write("Visualize and analyze the customer journey with the following graphs:")

    # User input for the number of data points
    num_data_points = st.sidebar.slider("Select number of data points", min_value=10, max_value=500, value=100)

    # Generate enhanced dummy data
    np.random.seed(42)  # For reproducibility
    stages = ['Awareness', 'Consideration', 'Purchase', 'Retention', 'Advocacy']
    data = {
        'Stage': np.random.choice(stages, num_data_points),
        'Count': np.random.randint(50, 1000, num_data_points),
        'Conversion Rate': np.random.rand(num_data_points)
    }
    df = pd.DataFrame(data)

    # Group by stage to summarize the data
    summary_df = df.groupby('Stage').agg({'Count': 'sum', 'Conversion Rate': 'mean'}).reset_index()

    # Detailed Summary of Customer Journey Data
    st.subheader("Summary of Customer Journey Data")
    st.write("""
        The customer journey is a critical framework used to understand and optimize the path that customers take from their first interaction with a brand to their final purchase and beyond. This journey is typically segmented into the following stages:

        - **Awareness**: This is the initial stage where potential customers first become aware of your product or service. It's crucial to attract their attention and create a positive first impression.
        - **Consideration**: At this stage, customers are actively considering your product or service as a potential solution to their needs. They compare it with other options available in the market.
        - **Purchase**: This is the stage where the customer makes the decision to buy your product or service. Effective strategies are needed to convert prospects into paying customers.
        - **Retention**: After the purchase, the focus shifts to retaining customers by ensuring they have a positive experience with the product or service. Retention efforts help build customer loyalty.
        - **Advocacy**: In this final stage, satisfied customers become advocates for your brand. They recommend your product or service to others, leading to new customer acquisitions.

        In our simulated data, we have captured the count of customers at each stage and their conversion rates. Let's explore the data in detail:
        - **Stage**: Represents the different stages in the customer journey.
        - **Count**: The number of customers at each stage.
        - **Conversion Rate**: The rate at which customers move from one stage to the next.

        The visualizations below will help us understand the dynamics at each stage of the customer journey.
    """)

    # Display the raw data
    st.write("### Raw Data")
    st.write(df.head())

    # 1. Funnel Chart using Plotly
    st.subheader("Customer Journey Funnel")
    funnel_fig = go.Figure(go.Funnel(
        y=summary_df['Stage'],
        x=summary_df['Count'],
        textinfo="value+percent initial"
    ))
    funnel_fig.update_layout(title="Customer Journey Funnel")
    st.plotly_chart(funnel_fig)
    st.write("""
        The funnel chart illustrates the number of customers at each stage of the customer journey. 
        It helps identify at which stage the most significant drop-off occurs. From the chart, you can see:
        - The highest number of customers are at the Awareness stage, indicating strong initial outreach efforts.
        - As customers move through the stages, the count decreases, highlighting the conversion challenges at each step.
    """)

    # 2. Conversion Rates Line Plot using Plotly
    st.subheader("Conversion Rates Over Stages")
    line_fig = px.line(summary_df, x='Stage', y='Conversion Rate', markers=True, title="Conversion Rates Over Customer Journey Stages")
    st.plotly_chart(line_fig)
    st.write("""
        The line plot shows the conversion rates across different stages of the customer journey. 
        It highlights which stages have higher or lower conversion rates, aiding in identifying areas for improvement. 
        For example:
        - A steep drop in conversion rate between the Consideration and Purchase stages may indicate the need for better incentives or information to help customers make purchasing decisions.
    """)

    # 3. Heatmap of Stage vs. Count using Plotly
    st.subheader("Heatmap of Customer Journey Stages")
    heatmap_data = df.pivot_table(index='Stage', columns='Count', values='Conversion Rate', aggfunc='mean')
    heatmap_fig = px.imshow(heatmap_data, labels=dict(x="Count", y="Stage", color="Conversion Rate"),
                            title="Heatmap of Customer Journey Stages")
    st.plotly_chart(heatmap_fig)
    st.write("""
        The heatmap provides a visual representation of the conversion rates at various stages and customer counts. 
        It helps identify patterns and correlations in the data. 
        For instance:
        - Higher conversion rates may be observed at certain customer counts, indicating a sweet spot for targeted marketing efforts.
    """)

    # 4. Bar Chart of Counts per Stage
    st.subheader("Counts per Stage")
    bar_fig = px.bar(summary_df, x='Stage', y='Count', title="Counts per Stage", text_auto=True)
    st.plotly_chart(bar_fig)
    st.write("""
        The bar chart displays the total number of customers at each stage. 
        It helps understand the distribution of customers across the different stages of the journey. 
        Key insights might include:
        - A large number of customers at the Awareness stage, suggesting effective initial marketing strategies.
        - Fewer customers at the Advocacy stage, indicating potential areas for enhancing customer loyalty programs.
    """)

    # 5. Pie Chart of Stage Distribution
    st.subheader("Stage Distribution")
    pie_fig = px.pie(summary_df, names='Stage', values='Count', title="Stage Distribution")
    st.plotly_chart(pie_fig)
    st.write("""
        The pie chart shows the distribution of customers across different stages as a percentage of the total. 
        This visualization helps to quickly grasp the proportion of customers at each stage. 
        For example:
        - A balanced distribution indicates a well-maintained customer journey pipeline.
        - An imbalanced distribution may highlight stages that require more attention or resources.
    """)

    # 6. Scatter Plot of Conversion Rates vs. Counts
    st.subheader("Conversion Rates vs. Counts")
    scatter_fig = px.scatter(df, x='Count', y='Conversion Rate', color='Stage', title="Conversion Rates vs. Counts", 
                             labels={"Count": "Customer Count", "Conversion Rate": "Conversion Rate"})
    st.plotly_chart(scatter_fig)
    st.write("""
        The scatter plot illustrates the relationship between customer count and conversion rates across different stages. 
        It helps identify trends and outliers. 
        Insights from this plot could include:
        - Clusters of high conversion rates at specific customer counts, suggesting optimal points for focused engagement strategies.
    """)

    # Customer Success Journey Roadmap
    st.subheader("Customer Success Journey Roadmap")
    
    st.write("""
        **Introduction:**
        The Customer Success Journey Roadmap is a strategic framework designed to help organizations visualize and optimize the entire lifecycle of their customers. By understanding the key stages of this journey, businesses can develop targeted strategies to enhance customer satisfaction, increase retention rates, and foster long-term loyalty. Each stage represents a critical touchpoint where specific actions and interventions can significantly impact the overall customer experience.
        
        **Stages of the Customer Journey:**
        
        1. **Awareness**
           - **Description**: This is the initial stage where potential customers first become aware of your product or service. It is crucial for capturing their attention and creating a strong first impression.
           - **Objectives**: Increase brand visibility, attract potential customers, and generate interest.
           - **Key Actions**:
             - Utilize digital marketing campaigns (e.g., SEO, PPC).
             - Engage in social media outreach and influencer collaborations.
             - Attend industry events and webinars.

        2. **Consideration**
           - **Description**: At this stage, customers are actively considering your product or service as a potential solution to their needs. They compare it with other options available in the market.
           - **Objectives**: Educate potential customers, highlight unique value propositions, and build trust.
           - **Key Actions**:
             - Provide detailed product information and case studies.
             - Offer free trials or demos.
             - Engage with potential customers through personalized emails and follow-ups.

        3. **Purchase**
           - **Description**: This is the stage where the customer makes the decision to buy your product or service. Effective strategies are needed to convert prospects into paying customers.
           - **Objectives**: Facilitate a smooth purchasing process and ensure a positive buying experience.
           - **Key Actions**:
             - Simplify the checkout process and offer multiple payment options.
             - Provide excellent customer support during the purchase.
             - Implement promotions or discounts to encourage conversion.

        4. **Retention**
           - **Description**: After the purchase, the focus shifts to retaining customers by ensuring they have a positive experience with the product or service. Retention efforts help build customer loyalty.
           - **Objectives**: Enhance customer satisfaction, encourage repeat purchases, and reduce churn.
           - **Key Actions**:
             - Offer post-purchase support and resources.
             - Implement loyalty programs or rewards.
             - Regularly solicit feedback and address any issues promptly.

        5. **Advocacy**
           - **Description**: In this final stage, satisfied customers become advocates for your brand. They recommend your product or service to others, leading to new customer acquisitions.
           - **Objectives**: Leverage satisfied customers to generate referrals and positive word-of-mouth.
           - **Key Actions**:
             - Encourage customers to leave reviews and testimonials.
             - Create referral programs and incentives.
             - Engage with customers on social media and highlight their success stories.
    """)

    # Roadmap Visualization
    roadmap_fig = go.Figure(data=go.Scatter(
        x=["Awareness", "Consideration", "Purchase", "Retention", "Advocacy"],
        y=[1, 2, 3, 4, 5],
        mode="lines+markers+text",
        text=["Awareness", "Consideration", "Purchase", "Retention", "Advocacy"],
        textposition="top center"
    ))
    roadmap_fig.update_layout(title="Customer Success Journey Roadmap",
                              xaxis_title="Stage",
                              yaxis_title="Sequence",
                              yaxis=dict(showticklabels=False))
    st.plotly_chart(roadmap_fig)

######Predictive Analytics Page
from model import train_model

# Load the model directly
model = train_model()

def generate_predictions(features):
    if model:
        try:
            predictions = model.predict(features)
            return predictions
        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            return np.zeros((1, 12))
    else:
        st.error("Model not loaded.")
        return np.zeros((1, 12))

def predictive_analytics_page():
    st.title("Customer Predictive Analytics")

    st.sidebar.header("Input Data")
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=30)
    annual_income = st.sidebar.slider("Annual Income", min_value=0, max_value=200000, value=50000)
    credit_score = st.sidebar.slider("Credit Score", min_value=300, max_value=850, value=650)
    churn_risk = st.sidebar.slider("Churn Risk", min_value=0.0, max_value=100.0, value=50.0)

    features = np.array([[age, annual_income, credit_score, churn_risk]])
    predictions = generate_predictions(features)

    st.subheader("Predicted Metrics")

    metrics = {
        "Net Promoter Score": predictions[0][0],
        "Customer Lifetime Value": predictions[0][1],
        "Customer Acquisition Cost": predictions[0][2],
        "Churn Rate": predictions[0][3],
        "Customer Satisfaction Score": predictions[0][4],
        "Customer Retention Rate": predictions[0][5],
        "Monthly Recurring Revenue": predictions[0][6],
        "Average Time on Platform": predictions[0][7],
        "First Contact Resolution Rate": predictions[0][8],
        "Free Trial Conversion Rate": predictions[0][9],
        "Repeat Purchase Rate": predictions[0][10],
        "Customer Effort Score": predictions[0][11],
    }

    # Add thresholds for each metric
    thresholds = {
        "Net Promoter Score": 50,
        "Customer Lifetime Value": 5000,
        "Customer Acquisition Cost": 200,
        "Churn Rate": 10,
        "Customer Satisfaction Score": 75,
        "Customer Retention Rate": 70,
        "Monthly Recurring Revenue": 500,
        "Average Time on Platform": 6,
        "First Contact Resolution Rate": 80,
        "Free Trial Conversion Rate": 25,
        "Repeat Purchase Rate": 30,
        "Customer Effort Score": 20,
    }

    # Create a DataFrame for metrics and thresholds
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])
    metrics_df['Threshold'] = metrics_df['Metric'].map(thresholds)
    metrics_df['Needs Improvement'] = metrics_df.apply(lambda row: "Yes" if row['Score'] < row['Threshold'] else "No", axis=1)

    st.write(metrics_df.style.set_properties(**{'height': '900px'}))

    st.subheader("Improvement Suggestions")

    suggestions = {
        "Net Promoter Score": [
            "1. **Explanation**: Measures customer loyalty and the likelihood of recommending your product/service.",
            "2. **Formula**: (Percentage of Promoters - Percentage of Detractors) × 100",
            "3. **Improvement**: Increase customer satisfaction by addressing key pain points and delivering exceptional customer service."
        ],
        "Customer Lifetime Value": [
            "1. **Explanation**: The total revenue a business can expect from a customer over their lifetime.",
            "2. **Formula**: Average Purchase Value × Number of Purchases per Year × Customer Lifespan",
            "3. **Improvement**: Increase customer retention through loyalty programs and personalized offers."
        ],
        "Customer Acquisition Cost": [
            "1. **Explanation**: The cost associated with acquiring a new customer.",
            "2. **Formula**: Total Cost of Acquiring New Customers / Number of New Customers Acquired",
            "3. **Improvement**: Optimize marketing strategies to reduce costs and improve conversion rates."
        ],
        "Churn Rate": [
            "1. **Explanation**: The percentage of customers who stop using your product/service during a given period.",
            "2. **Formula**: (Number of Customers Lost During Period / Total Number of Customers at Start of Period) × 100",
            "3. **Improvement**: Implement retention strategies such as regular check-ins and addressing customer feedback promptly."
        ],
        "Customer Satisfaction Score": [
            "1. **Explanation**: Measures how satisfied customers are with your product/service.",
            "2. **Formula**: (Number of Satisfied Customers / Number of Survey Responses) × 100",
            "3. **Improvement**: Enhance product/service quality and provide excellent customer support."
        ],
        "Customer Retention Rate": [
            "1. **Explanation**: The percentage of customers who continue to use your product/service over a given period.",
            "2. **Formula**: ((Number of Customers at End of Period - Number of New Customers) / Number of Customers at Start of Period) × 100",
            "3. **Improvement**: Foster strong relationships with existing customers and offer value-added services."
        ],
        "Monthly Recurring Revenue": [
            "1. **Explanation**: The total revenue generated from subscriptions on a monthly basis.",
            "2. **Formula**: Total Revenue from Subscriptions / Number of Months",
            "3. **Improvement**: Increase subscription rates and offer upsells to boost recurring revenue."
        ],
        "Average Time on Platform": [
            "1. **Explanation**: Measures the average time users spend on your platform.",
            "2. **Formula**: Total Time Spent by All Users / Number of Users",
            "3. **Improvement**: Enhance user experience to encourage longer engagement with the platform."
        ],
        "First Contact Resolution Rate": [
            "1. **Explanation**: The percentage of customer issues resolved on the first contact.",
            "2. **Formula**: (Number of Issues Resolved on First Contact / Total Number of Issues) × 100",
            "3. **Improvement**: Improve training for customer service representatives to resolve issues more effectively on the first contact."
        ],
        "Free Trial Conversion Rate": [
            "1. **Explanation**: The percentage of free trial users who convert to paying customers.",
            "2. **Formula**: (Number of Free Trial Users Who Become Paying Customers / Total Number of Free Trial Users) × 100",
            "3. **Improvement**: Optimize the free trial experience to highlight key benefits and increase conversions."
        ],
        "Repeat Purchase Rate": [
            "1. **Explanation**: The percentage of purchases made by repeat customers.",
            "2. **Formula**: (Number of Repeat Purchases / Total Number of Purchases) × 100",
            "3. **Improvement**: Encourage repeat purchases through targeted marketing and loyalty programs."
        ],
        "Customer Effort Score": [
            "1. **Explanation**: Measures the ease of customer interaction and resolution.",
            "2. **Formula**: Average Score from Customer Effort Surveys",
            "3. **Improvement**: Simplify processes and interactions to reduce customer effort and enhance satisfaction."
        ],
    }

    for metric, details in suggestions.items():
        st.write(f"### {metric}")
        for detail in details:
            st.write(f"- {detail}")
######NAVBAR

import streamlit as st

def shownavbar():
    st.sidebar.title("CSM Dashboard")
    page = st.sidebar.radio("Select a Page", ("Introduction", "Contact", "Articles", "Customer Journey Mapping", "Predictive Analytics"))

    return page

# Example usage in your app
if __name__ == "__main__":
    page = shownavbar()

    if page == "Introduction":
        # Call the function to display the Introduction page
        introduction_page()
    elif page == "Contact":
        # Call the function to display the Contact page
        contact_page()
    elif page == "Articles":
        # Call the function to display the Articles page
        articles_page()
    elif page == "Customer Journey Mapping":
        # Call the function to display the Customer Journey Mapping page
        customer_journey_mapping_page()
    elif page == "Predictive Analytics":
        # Call the function to display the Predictive Analytics page
        predictive_analytics_page()
