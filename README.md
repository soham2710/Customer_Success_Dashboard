# Iris Dataset Classification and Statistical Analysis

Welcome to the Iris Dataset Classification and Statistical Analysis Streamlit app! 

This application allows you to train various machine learning models on the Iris dataset, evaluate their performance, and perform a range of statistical tests.

#Table of Contents
Introduction
Features
Models
Statistical Tests
Usage
Installation
Contributing
License
Introduction
The Iris dataset is a classic dataset used in machine learning and statistics. It consists of 150 samples from three different species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor), with four features measured for each sample: sepal length, sepal width, petal length, and petal width.

This app allows you to:

Train and evaluate multiple machine learning models on the Iris dataset.
Perform various statistical tests on the dataset.
Visualize the results directly in the browser using Streamlit.
Features
Model Training and Evaluation: Train and evaluate multiple machine learning models, including logistic regression, Naive Bayes, SVM, KNN, and neural networks (ANN, CNN, RNN).
Statistical Analysis: Perform a variety of statistical tests on the dataset, including t-tests, chi-squared tests, ANOVA, and more.
Interactive Interface: Use an interactive Streamlit interface to select models, view results, and perform statistical tests.
Models
Logistic Regression
A linear model for binary classification that estimates the probability that a given input belongs to a certain class.

Naive Bayes
A probabilistic classifier based on Bayes' theorem, assuming independence between features.

Support Vector Machine (SVM)
A classifier that finds the hyperplane that best separates the data into different classes.

K-Nearest Neighbors (KNN)
A non-parametric classifier that predicts the class of a sample based on the majority class among its k-nearest neighbors.

Artificial Neural Network (ANN)
A neural network model consisting of multiple layers of neurons, used to capture complex patterns in the data.

Convolutional Neural Network (CNN)
A neural network model specialized for processing grid-like data, such as images, using convolutional layers.

Recurrent Neural Network (RNN)
A neural network model designed for sequential data, where the output from previous steps is used as input for the current step.

Statistical Tests
T-test
Compares the means of two groups to determine if they are significantly different from each other.

Chi-squared Test
Tests the independence of two categorical variables.

ANOVA (Analysis of Variance)
Compares the means of three or more groups to determine if at least one of them is significantly different.

Wilcoxon Test
A non-parametric test that compares two paired groups.

Mann-Whitney U Test
A non-parametric test that compares two independent groups.

Kruskal-Wallis H Test
A non-parametric test that compares three or more independent groups.

Friedman Test
A non-parametric test for detecting differences in treatments across multiple test attempts.

Z-score Test
Measures the number of standard deviations a data point is from the mean.

Usage
Install Dependencies: Ensure you have Python and the required libraries installed. You can install the required libraries using:

bash
Copy code
pip install streamlit tensorflow scikit-learn pandas numpy
Run the App: Save the app.py file and run the Streamlit app:

bash
Copy code
streamlit run app.py
Interact with the App: Open the provided URL in your browser, select models and statistical tests from the sidebar, and view the results.

Installation
To set up and run this project locally, follow these steps:

Clone the repository:

bash
Copy code
git clone <repository_url>
Navigate to the project directory:

bash
Copy code
cd iris-classification-app
Install the dependencies:

bash
Copy code
pip install streamlit tensorflow scikit-learn pandas numpy
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure your code adheres to the project's coding standards and includes appropriate tests.

License
This project is licensed under the MIT License. See the LICENSE file for details.
