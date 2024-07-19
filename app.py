import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Introduction Page content
def introduction_page():
    st.title("Introduction")
    st.write("Welcome to our web app! Here you will find information about our services, articles, and more.")
    st.write("Feel free to explore the pages using the navigation bar.")

# CONTACT PAGE
def contact_page():
    st.title("Contact Us")
    st.write("Get in touch with us through the following channels:")
    st.write("- Email: example@example.com")
    st.write("- Phone: +1234567890")
    st.write("- Address: 123 Main Street, Anytown, USA")

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
    st.write("""
        The roadmap chart outlines the sequence of stages in the customer success journey. 
        It serves as a high-level overview, helping to align strategies and actions at each stage. 
        Understanding this sequence is crucial for developing targeted interventions and optimizing the customer experience at every stage.
    """)
    
def show_navbar():
    st.sidebar.title("Navigation")

    profile_image_url = "https://github.com/soham2710/Customer_Success_Dashboard/raw/main/BH6A0835.jpg"
    st.sidebar.image(profile_image_url, use_column_width=True)
    st.sidebar.write("**Name:** Your Name")
    st.sidebar.write("**Position:** Your Position")
    st.sidebar.write("**Bio:** Brief bio or description.")

    resume_url = "https://github.com/soham2710/Customer_Success_Dashboard/raw/main/Customer%20Success%20Resume.pdf"
    response = requests.get(resume_url)
    st.sidebar.download_button(
        label="Download Resume",
        data=response.content,
        file_name="resume.pdf",
        mime="application/pdf"
    )

    pages = ["Introduction", "Contact", "Articles", "Customer Journey Mapping"]
    selected_page = st.sidebar.radio("Select a page", pages)
    return selected_page

def main():
    st.set_page_config(page_title="My Web App", page_icon=":guardsman:", layout="wide")
    selected_page = show_navbar()

    if selected_page == "Introduction":
        introduction_page()
    elif selected_page == "Contact":
        contact_page()
    elif selected_page == "Articles":
        articles_page()
    elif selected_page == "Customer Journey Mapping":
        customer_journey_mapping_page()

if __name__ == "__main__":
    main()
