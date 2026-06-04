# Weather Temperature Prediction Project

## Overview
This project aims to predict daily weather temperature based on historical weather data. The workflow involves data loading, preprocessing, exploratory data analysis (EDA), model training (Linear Regression and Random Forest), hyperparameter tuning, and a Streamlit web application for interactive predictions.

## Data Source
The historical weather data for 2020 was downloaded from KaggleHub: `ahmedgaitani/historical-weather-data-for-2020`.

## Data Science Workflow

1.  **Data Loading & Initial Exploration**:
    *   Common Python libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `kagglehub`) were imported.
    *   The dataset was loaded into a pandas DataFrame (`df`).
    *   Basic checks (`df.head()`, `df.columns`, `df.info()`, `df.shape`, `df.isnull().sum()`, `df.duplicated().sum()`) confirmed data integrity and provided an overview.

2.  **Feature Engineering**:
    *   The 'Date' column was converted to datetime objects and decomposed into 'Year', 'Month', and 'Day'. The original 'Date' column was then dropped.
    *   Categorical features ('Station', 'WeatherCondition') were one-hot encoded using `pd.get_dummies` (resulting in `df_encoded`). The preprocessed DataFrame (before one-hot encoding for pipeline use) was saved as `preprocessed_weather_data_no_encoding.pkl`.

3.  **Exploratory Data Analysis (EDA)**:
    *   Visualizations were generated to explore relationships, including scatter plots for humidity vs. temperature, and line plots for monthly average temperature and precipitation.
    *   A correlation matrix was calculated and visualized as a heatmap.
    *   Distribution plots for 'WindSpeed' and 'Precipitation', and their scatter plot, were created.
    *   It was noted that the dataset lacked a 'Pressure' column.

4.  **Model Preparation**:
    *   Input features `X` were selected: 'Station', 'Precipitation', 'Humidity', 'WindSpeed', 'WeatherCondition', 'Month', 'Day'.
    *   The target variable `y` was 'Temperature'.
    *   The data was split into training and testing sets (80/20 ratio).

5.  **Linear Regression Model**:
    *   A Linear Regression model was trained within a `Pipeline` that included `OneHotEncoder` for categorical features.
    *   **Evaluation**: The model performed very poorly (R-squared: -0.03), indicating it was worse than simply predicting the mean.
    *   Feature importance (coefficients) analysis showed 'WeatherCondition' and 'Station' as influential features.

6.  **Random Forest Regressor Model**:
    *   Given the poor performance of Linear Regression, a Random Forest Regressor was trained using the same preprocessing pipeline.
    *   **Evaluation**: This model also performed poorly (R-squared: -0.13), suggesting more complex models alone are not sufficient.

7.  **Hyperparameter Tuning for Random Forest**:
    *   `RandomizedSearchCV` was employed to tune the Random Forest model's hyperparameters.
    *   **Note**: A `FitFailedWarning` occurred due to `'auto'` being a deprecated value for `max_features` in newer scikit-learn versions, but the search completed with valid parameters.
    *   **Evaluation**: The tuned Random Forest model showed marginal improvement (R-squared: -0.07) but still performed poorly.

8.  **Workflow Export**:
    *   The data processing and modeling workflow was modularized into `preprocess.py`, `train.py`, and `predict.py` scripts.

9.  **Streamlit Web Application**:
    *   A Streamlit application (`app.py`) was generated to provide an interactive interface for temperature prediction using the *tuned* Random Forest model.

## Model Performance Summary
Both Linear Regression and Random Forest models consistently showed very poor predictive power, with negative R-squared values indicating they perform worse than a baseline model that always predicts the mean. This suggests that the current features or model configurations are insufficient to capture the underlying patterns in the data.

| Metric      | Linear Regression | Random Forest | Tuned Random Forest |
| :---------- | :---------------- | :------------ | :------------------ |
| MAE         | 12.06             | 12.49         | 12.23               |
| MSE         | 187.70            | 206.92        | 194.93              |
| RMSE        | 13.70             | 14.38         | 13.96               |
| R-squared   | -0.03             | -0.13         | -0.07               |

## Next Steps for Improvement
To improve model performance, the following avenues could be explored:

*   **More Advanced Feature Engineering**: Investigate creating lag features (e.g., previous day's temperature), rolling averages, interaction terms between existing features, or time-series specific features.
*   **Exploring More Sophisticated Models**: Consider Gradient Boosting Machines (like XGBoost or LightGBM) or even neural networks, which are capable of capturing more complex non-linear relationships.
*   **Incorporating External Data Sources**: The absence of a 'Pressure' column was noted. Adding more relevant meteorological data (e.g., atmospheric pressure, wind direction, cloud cover, geographical data like altitude) could significantly enhance predictive power.
*   **Time Series Analysis**: Since the data is time-series based, specific time-series models (e.g., ARIMA, Prophet) might be more appropriate.

## Requirements
The project dependencies are listed in `requirements.txt`:
```
pandas
numpy==2.0.2
matplotlib
seaborn
kagglehub
scikit-learn==1.6.1
joblib==1.5.3
streamlit
```

## How to Run the Streamlit Application

1.  **Ensure Dependencies are Installed**: If you haven't already, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Model and Scripts**:
    *   Make sure you have `tuned_random_forest_model.pkl` (the trained and tuned model), `preprocess.py`, `train.py`, `predict.py`, and `app.py` in your working directory.
    *   If you're running this from a fresh environment, execute the relevant cells in the notebook to generate `preprocessed_weather_data_no_encoding.pkl`, `linear_regression_model.pkl`, `random_forest_model.pkl`, and `tuned_random_forest_model.pkl`.

3.  **Run the Streamlit App**:
    *   Open your terminal or command prompt.
    *   Navigate to the directory where `app.py` is located.
    *   Execute the command:
        ```bash
        streamlit run app.py
        ```
    *   This will open the Streamlit application in your web browser, allowing you to interactively predict temperatures.
