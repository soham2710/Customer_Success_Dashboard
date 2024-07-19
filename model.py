import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Generate dummy data
def generate_dummy_data(num_samples=1000):
    np.random.seed(0)
    data = {
        'Age': np.random.randint(18, 70, num_samples),
        'Annual Income': np.random.randint(20000, 150000, num_samples),
        'Credit Score': np.random.randint(300, 850, num_samples),
        'Churn Risk': np.random.uniform(0, 100, num_samples),
        'NPS': np.random.uniform(0, 100, num_samples),
        'Customer Satisfaction Score': np.random.uniform(0, 100, num_samples),
        'Churn Rate': np.random.uniform(0, 100, num_samples),
        'Customer Lifetime Value': np.random.uniform(1000, 10000, num_samples),
        'Customer Acquisition Cost': np.random.uniform(50, 500, num_samples),
        'Customer Retention Rate': np.random.uniform(0, 100, num_samples),
        'Monthly Recurring Revenue': np.random.uniform(100, 1000, num_samples),
        'Average Time on Platform': np.random.uniform(1, 12, num_samples),
        'First Contact Resolution Rate': np.random.uniform(0, 100, num_samples),
        'Free Trial Conversion Rate': np.random.uniform(0, 100, num_samples),
        'Repeat Purchase Rate': np.random.uniform(0, 100, num_samples),
        'Customer Effort Score': np.random.uniform(0, 100, num_samples)
    }
    df = pd.DataFrame(data)
    return df

# Train the predictive model
def train_model():
    df = generate_dummy_data()
    
    # Features and target variable
    features = df[['Age', 'Annual Income', 'Credit Score', 'Churn Risk']]
    targets = df[['NPS', 'Customer Satisfaction Score', 'Churn Rate',
                  'Customer Lifetime Value', 'Customer Acquisition Cost', 
                  'Customer Retention Rate', 'Monthly Recurring Revenue',
                  'Average Time on Platform', 'First Contact Resolution Rate',
                  'Free Trial Conversion Rate', 'Repeat Purchase Rate',
                  'Customer Effort Score']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    return model

# Call the function to train the model
if __name__ == "__main__":
    model = train_model()
