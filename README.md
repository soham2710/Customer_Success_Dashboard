# Streamlit Model Comparison App

## Overview

This Streamlit app allows users to upload a CSV file containing their data and choose from various machine learning and neural network models. The app trains and evaluates the selected model, performs statistical tests, and provides a summary and interpretation of the results using a Large Language Model (LLM). Users can also input new data to make predictions with the trained model.

## Features

- **Data Upload:** Upload CSV files for analysis.
- **Model Selection:** Choose from a variety of machine learning and neural network models.
- **Model Evaluation:** View accuracy, classification reports, and confusion matrices.
- **Statistical Tests:** Perform and display results of various statistical tests.
- **LLM Integration:** Get summaries and interpretations of statistical tests.
- **Prediction:** Input new data to make predictions with the trained model.
- **Model Comparison:** Visualize performance of different models.

## Requirements

- Python 3.x
- Streamlit
- numpy
- pandas
- scikit-learn
- OpenAI (for LLM integration)

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/streamlit-model-comparison.git
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key:
    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Upload Data:** Use the sidebar to upload your CSV file.
2. **Select Model:** Choose a model from the dropdown menu.
3. **View Results:** See model evaluation metrics, statistical tests, and LLM summaries.
4. **Predict New Data:** Enter feature values in the sidebar to make predictions.
5. **Compare Models:** View a bar chart comparing the performance of different models.

## Code Explanation

- **app.py:** The main script that runs the Streamlit app.
- **model.py:** Contains functions for data preprocessing, model training, evaluation, statistical tests, and LLM summaries.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact [your_email@example.com](mailto:your_email@example.com).

