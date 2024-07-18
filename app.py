import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Function to suggest email templates based on prediction
def suggest_email_template(prediction_label, selected_action):
    templates = {
        'Yes': {
            'Offer a product demo': """Dear [Customer], As a valued customer, we would like to offer you a product demo to help you get the most out of our product. Please let us know a convenient time for you.""",
            'Schedule a follow-up call': """Dear [Customer], We would like to schedule a follow-up call to discuss your experience with our product and address any concerns you may have. Please let us know a suitable time for you.""",
            'Invite to customer success webinar': """Dear [Customer], We are excited to invite you to our upcoming customer success webinar, where we will share tips and best practices for using our product effectively. Please join us on [date].""",
            'Send promotional offers': """Dear [Customer], We have some exciting promotional offers just for you! Check your account for exclusive discounts and offers."""
        },
        'No': {
            'Send a feedback survey': """Dear [Customer], Thank you for your recent interaction with us. We would appreciate your feedback to help us improve our services. Please take a moment to complete our feedback survey.""",
            'Thank you for your feedback': """Dear [Customer], Thank you for your valuable feedback. We appreciate your input and will use it to enhance your experience with us."""
        }
    }
    return templates[prediction_label][selected_action]

# Generate synthetic dataset for predictive analytics
np.random.seed(42)
data = {
    'Usage Frequency': np.random.choice(['Daily', 'Weekly', 'Monthly'], 1000),
    'Support Tickets': np.random.randint(0, 10, 1000),
    'Feedback Score': np.round(np.random.uniform(1, 5, 1000), 1),
    'Purchase Amount': np.random.uniform(100, 1000, 1000),
    'Tenure': np.random.randint(1, 60, 1000),
    'Needs Engagement': np.random.randint(0, 2, 1000)
}

# Placeholder for predictive model
model_playbooks = Sequential([
    Dense(10, input_shape=(5,), activation='relu'),
    Dense(1, activation='sigmoid')
])
model_playbooks.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train predictive model
X = np.array([data['Support Tickets'], data['Feedback Score'], data['Purchase Amount'], data['Tenure'], data['Needs Engagement']]).T
y = np.array(data['Usage Frequency'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_playbooks.fit(X_train_scaled, y_train, epochs=50, verbose=0, validation_data=(X_test_scaled, y_test))

# Function to predict needs based on user input
def predict_needs(support_tickets, feedback_score, purchase_amount, tenure, needs_engagement):
    try:
        input_data = np.array([[support_tickets, feedback_score, purchase_amount, tenure, needs_engagement]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model_playbooks.predict(input_data_scaled)[0]
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Function to generate email templates based on prediction label and selected action
def generate_email_templates(prediction_label, selected_action):
    templates = suggest_email_template(prediction_label, selected_action)
    return templates[selected_action]

# Function to generate dummy data for customer journey mapping
def generate_dummy_journey_data():
    stages = ['Awareness', 'Consideration', 'Purchase', 'Retention', 'Advocacy']
    customers = np.random.randint(20, 100, size=len(stages))
    satisfaction = np.round(np.random.uniform(3.0, 5.0, size=len(stages)), 1)
    data = {
        'Stage': stages,
        'Customers': customers,
        'Satisfaction': satisfaction
    }
    return pd.DataFrame(data)

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
    
    # Predictive model simulation
    st.subheader("Predict Customer Needs")
    support_tickets = st.slider("Support Tickets", 0, 10, 5)
    feedback_score = st.slider("Feedback Score", 2.0, 5.0, 3.5, 0.1)
    purchase_amount = st.slider("Purchase Amount", 100, 1000, 500)
    tenure = st.slider("Tenure (Months)", 1, 60, 30)
    needs_engagement = st.radio("Needs Engagement", options=['Yes', 'No'])
    
    needs_engagement_binary = 1 if needs_engagement == 'Yes' else 0
    prediction = predict_needs(support_tickets, feedback_score, purchase_amount, tenure, needs_engagement_binary)
    
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

# Main app logic
def main():
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
                        Choose a page to view:
                        - **Customer Journey Mapping**
                        - **Predictive Analytics Playbooks**
                        """)
    selection = st.sidebar.radio("Go to", ["Customer Journey Mapping", "Predictive Analytics Playbooks"])

    if selection == "Customer Journey Mapping":
        customer_journey_page()
    elif selection == "Predictive Analytics Playbooks":
        predictive_analytics_page()

if __name__ == "__main__":
    main()
