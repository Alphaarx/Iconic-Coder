# Iconic-Coder
# Survey Trend Analysis Tool

This Python project provides a complete framework for simulating, analyzing, visualizing, and forecasting trends in survey data related to programming languages, IDEs, and job roles in the tech industry.

## 📊 Features

- Generate realistic sample survey data with growth/decline trends
- Perform comprehensive exploratory data analysis (EDA)
- Analyze historical patterns and growth rates
- Visualize correlations between survey attributes
- Forecast trends using ARIMA and simplified Prophet-style models
- Compare technologies over time
- Generate a detailed text-based analysis report

## 🛠️ Requirements

- Python 3.7+
- Required Python packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `statsmodels`
  - `scikit-learn`

Install dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
🚀 Getting Started
Clone the Repository or download the survey_trend_analysis.py file.

Run the Script:

bash
Copy
Edit
python survey_trend_analysis.py
By default, it will:

Generate 5 years of synthetic survey data

Display insights and visualizations

Perform trend forecasting

Generate a full textual report

📁 Project Structure
SurveyTrendAnalyzer class: main engine for data generation, exploration, analysis, and forecasting.

generate_sample_data(): simulates synthetic survey data.

load_data(filepath): load real survey data from CSV file.

explore_data(): visualizes distributions and trends.

analyze_historical_patterns(): identifies growth patterns and correlations.

forecast_arima() / forecast_prophet(): provides trend forecasts.

compare_technologies(): plots comparative trends.

generate_report(): prints key statistics, growth leaders, and predictions.

📈 Forecasting Support
ARIMA Forecasting:

Automatically selects best (p, d, q) parameters.

Includes 95% confidence intervals.

Prophet-style Forecasting:

Simplified linear + seasonal approach.

No Facebook Prophet dependency required.

📝 Using Your Own Data
Replace the sample generation with:

python
Copy
Edit
analyzer.load_data('your_file.csv')
Ensure your CSV has the following columns:

date

language

ide

job_role

experience_years

satisfaction_score

📌 Notes
All visualizations are built using matplotlib and seaborn.

Label encoding is used for correlation matrices.

Missing time-series periods are automatically filled with zero values.

📄 License
This project is provided for educational and demonstration purposes. You may modify or reuse it freely in your projects.
