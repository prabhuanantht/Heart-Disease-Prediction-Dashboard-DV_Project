# Heart Disease Dashboard with Streamlit

This project is a **Streamlit-based dashboard** for visualizing and analyzing heart disease data. The dashboard provides insights into the dataset through various visualizations and evaluates a machine learning model (XGBoost) for predicting heart disease.

## Features

- **Data Cleaning**: Handles missing and incorrect data values.
- **Data Visualization**:
  - Target distribution
  - Correlation heatmap
  - Numerical and categorical feature distributions
  - Pairplots for feature relationships
  - Chest pain type analysis
  - Age vs cholesterol scatter plot
- **Machine Learning**:
  - Preprocessing with feature scaling.
  - Train-test split.
  - XGBoost model training and evaluation.
  - Classification report and confusion matrix.
  
## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/prabhuanantht/Heart-Disease-Prediction-Dashboard-DV_Project.git
   cd heart-disease-dashboard
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The project uses the `heart.csv` dataset. Ensure the dataset is present in the project directory.

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the link provided by Streamlit in your browser to access the dashboard.

## Dashboard Navigation

- **Target Distribution**: Visualizes the distribution of heart disease (1 = Yes, 0 = No).
- **Correlation Heatmap**: Shows the correlation between features in the dataset.
- **Numerical Feature Distributions**: Histograms for numerical features.
- **Categorical Feature Distributions**: Count plots for categorical features, categorized by the target variable.
- **Pairplot**: Relationship plots for selected numerical features.
- **Chest Pain Type Analysis**: Breakdown of chest pain types by heart disease status.
- **Age vs Cholesterol Scatter Plot**: Examines the relationship between age, cholesterol levels, and heart disease.
- **Model Evaluation**: Trains an XGBoost model and displays the classification report and confusion matrix.

## File Structure

- `dvlab.py`: Main script for running the Streamlit dashboard.
- `heart.csv`: Dataset file.
- `requirements.txt`: Python dependencies.

## Dependencies

The required libraries are listed in the `requirements.txt`. Major dependencies include:
- Streamlit
- Pandas
- Numpy
- Seaborn
- Matplotlib
- Scikit-learn
- XGBoost

## Model Evaluation

The XGBoost model provides a detailed classification report and confusion matrix for evaluating its performance on the dataset.

## Future Enhancements

- Add more machine learning models for comparison.
- Improve feature engineering.
- Integrate interactive data filtering.

## License

This project is licensed under the MIT License.
