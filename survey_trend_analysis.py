import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For time series forecasting
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SurveyTrendAnalyzer:
    def __init__(self):
        self.data = None
        self.time_series_data = None
        self.forecasts = {}
        
    def generate_sample_data(self, years=5, responses_per_year=1000):
        """Generate realistic survey data for demonstration"""
        np.random.seed(42)
        
        # Define technology trends
        languages = ['Python', 'JavaScript', 'Java', 'C#', 'Go', 'Rust', 'TypeScript']
        ides = ['VS Code', 'IntelliJ', 'PyCharm', 'Sublime', 'Atom', 'Vim', 'Eclipse']
        job_roles = ['Data Scientist', 'Full Stack Developer', 'Backend Developer', 
                    'Frontend Developer', 'DevOps Engineer', 'ML Engineer', 'Software Architect']
        
        # Simulate growth/decline trends
        language_trends = {
            'Python': 0.15,      # Strong growth
            'JavaScript': 0.08,   # Steady growth
            'TypeScript': 0.25,   # Rapid growth
            'Java': -0.05,        # Slight decline
            'C#': 0.02,          # Stable
            'Go': 0.18,          # Growing
            'Rust': 0.30         # Emerging/growing
        }
        
        data_list = []
        start_date = datetime.now() - timedelta(days=365*years)
        
        for year in range(years):
            for month in range(12):
                survey_date = start_date + timedelta(days=30*month + 365*year)
                monthly_responses = responses_per_year // 12
                
                for _ in range(monthly_responses):
                    # Apply trends to language selection
                    lang_weights = []
                    for lang in languages:
                        base_weight = 1.0
                        if lang in language_trends:
                            # Apply compound growth
                            growth_factor = (1 + language_trends[lang]) ** (year + month/12)
                            base_weight *= growth_factor
                        lang_weights.append(base_weight)
                    
                    # Normalize weights
                    lang_weights = np.array(lang_weights) / sum(lang_weights)
                    
                    selected_lang = np.random.choice(languages, p=lang_weights)
                    selected_ide = np.random.choice(ides)
                    selected_role = np.random.choice(job_roles)
                    
                    # Add some correlation between role and language
                    if selected_role == 'Data Scientist' and np.random.random() < 0.7:
                        selected_lang = 'Python'
                    elif selected_role == 'Frontend Developer' and np.random.random() < 0.8:
                        selected_lang = np.random.choice(['JavaScript', 'TypeScript'])
                    
                    data_list.append({
                        'date': survey_date,
                        'language': selected_lang,
                        'ide': selected_ide,
                        'job_role': selected_role,
                        'experience_years': np.random.exponential(3) + 1,
                        'satisfaction_score': np.random.normal(7.5, 1.5)
                    })
        
        self.data = pd.DataFrame(data_list)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['year_month'] = self.data['date'].dt.to_period('M')
        
        print(f"Generated {len(self.data)} survey responses over {years} years")
        return self.data
    
    def load_data(self, filepath):
        """Load actual survey data from CSV"""
        try:
            self.data = pd.read_csv(filepath)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data['year_month'] = self.data['date'].dt.to_period('M')
            print(f"Loaded {len(self.data)} survey responses")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        """Comprehensive exploratory data analysis"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("=== SURVEY DATA OVERVIEW ===")
        print(f"Total responses: {len(self.data)}")
        print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"Unique languages: {self.data['language'].nunique()}")
        print(f"Unique IDEs: {self.data['ide'].nunique()}")
        print(f"Unique job roles: {self.data['job_role'].nunique()}")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Survey Response Analysis', fontsize=16, fontweight='bold')
        
        # 1. Language popularity over time
        lang_time = self.data.groupby(['year', 'language']).size().unstack(fill_value=0)
        lang_time_pct = lang_time.div(lang_time.sum(axis=1), axis=0) * 100
        
        top_languages = self.data['language'].value_counts().head(5).index
        for lang in top_languages:
            if lang in lang_time_pct.columns:
                axes[0,0].plot(lang_time_pct.index, lang_time_pct[lang], 
                              marker='o', linewidth=2, label=lang)
        
        axes[0,0].set_title('Programming Language Trends (%)')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Percentage of Responses')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. IDE popularity
        ide_counts = self.data['ide'].value_counts().head(8)
        axes[0,1].bar(range(len(ide_counts)), ide_counts.values)
        axes[0,1].set_title('IDE Popularity')
        axes[0,1].set_xlabel('IDEs')
        axes[0,1].set_ylabel('Number of Users')
        axes[0,1].set_xticks(range(len(ide_counts)))
        axes[0,1].set_xticklabels(ide_counts.index, rotation=45)
        
        # 3. Job role distribution
        role_counts = self.data['job_role'].value_counts()
        axes[0,2].pie(role_counts.values, labels=role_counts.index, autopct='%1.1f%%')
        axes[0,2].set_title('Job Role Distribution')
        
        # 4. Experience distribution
        axes[1,0].hist(self.data['experience_years'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('Experience Years Distribution')
        axes[1,0].set_xlabel('Years of Experience')
        axes[1,0].set_ylabel('Frequency')
        
        # 5. Satisfaction scores over time
        satisfaction_time = self.data.groupby('year')['satisfaction_score'].mean()
        axes[1,1].plot(satisfaction_time.index, satisfaction_time.values, 
                      marker='o', linewidth=3, color='green')
        axes[1,1].set_title('Average Satisfaction Score Over Time')
        axes[1,1].set_xlabel('Year')
        axes[1,1].set_ylabel('Average Satisfaction')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Monthly response volume
        monthly_responses = self.data.groupby('year_month').size()
        axes[1,2].plot(range(len(monthly_responses)), monthly_responses.values, 
                      marker='o', linewidth=2, color='orange')
        axes[1,2].set_title('Monthly Response Volume')
        axes[1,2].set_xlabel('Time Period')
        axes[1,2].set_ylabel('Number of Responses')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print key insights
        print("\n=== KEY INSIGHTS ===")
        print(f"Most popular language: {self.data['language'].mode()[0]}")
        print(f"Most popular IDE: {self.data['ide'].mode()[0]}")
        print(f"Most common job role: {self.data['job_role'].mode()[0]}")
        print(f"Average experience: {self.data['experience_years'].mean():.1f} years")
        print(f"Average satisfaction: {self.data['satisfaction_score'].mean():.1f}/10")
    
    def analyze_historical_patterns(self):
        """Analyze historical patterns and correlations"""
        if self.data is None:
            print("No data loaded.")
            return
        
        print("\n=== HISTORICAL PATTERN ANALYSIS ===")
        
        # Language trend analysis
        lang_yearly = self.data.groupby(['year', 'language']).size().unstack(fill_value=0)
        lang_growth = lang_yearly.pct_change().mean() * 100
        
        print("\nLanguage Growth Rates (Average % change per year):")
        for lang, growth in lang_growth.sort_values(ascending=False).items():
            if not pd.isna(growth):
                print(f"  {lang}: {growth:+.1f}%")
        
        # Job role evolution
        role_yearly = self.data.groupby(['year', 'job_role']).size().unstack(fill_value=0)
        role_growth = role_yearly.pct_change().mean() * 100
        
        print("\nJob Role Growth Rates (Average % change per year):")
        for role, growth in role_growth.sort_values(ascending=False).items():
            if not pd.isna(growth):
                print(f"  {role}: {growth:+.1f}%")
        
        # Correlation analysis
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Create correlation matrix for numerical variables
        corr_data = self.data[['experience_years', 'satisfaction_score']].copy()
        
        # Add encoded categorical variables for correlation
        le_lang = LabelEncoder()
        le_ide = LabelEncoder()
        le_role = LabelEncoder()
        
        corr_data['language_encoded'] = le_lang.fit_transform(self.data['language'])
        corr_data['ide_encoded'] = le_ide.fit_transform(self.data['ide'])
        corr_data['role_encoded'] = le_role.fit_transform(self.data['job_role'])
        
        correlation_matrix = corr_data.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix of Survey Variables')
        plt.tight_layout()
        plt.show()
    
    def prepare_time_series_data(self, category='language', item=None):
        """Prepare time series data for forecasting"""
        if self.data is None:
            print("No data loaded.")
            return None
        
        if item is None:
            # Use the most popular item in the category
            item = self.data[category].mode()[0]
        
        # Create monthly time series
        monthly_data = (self.data[self.data[category] == item]
                       .groupby('year_month')
                       .size()
                       .reset_index(name='count'))
        
        # Fill missing months with 0
        all_months = pd.period_range(start=self.data['year_month'].min(),
                                   end=self.data['year_month'].max(),
                                   freq='M')
        
        monthly_data = monthly_data.set_index('year_month').reindex(all_months, fill_value=0)
        monthly_data.index = monthly_data.index.to_timestamp()
        
        self.time_series_data = monthly_data
        
        print(f"Prepared time series data for {category}: {item}")
        print(f"Data points: {len(monthly_data)}")
        
        return monthly_data
    
    def forecast_arima(self, category='language', item=None, forecast_periods=12):
        """Perform ARIMA forecasting"""
        ts_data = self.prepare_time_series_data(category, item)
        if ts_data is None:
            return None
        
        # Check stationarity
        def check_stationarity(timeseries):
            result = adfuller(timeseries)
            return result[1] <= 0.05  # p-value threshold
        
        # Make series stationary if needed
        ts_values = ts_data['count'].values
        if not check_stationarity(ts_values):
            ts_values = np.diff(ts_values)
        
        # Fit ARIMA model (auto-select parameters)
        best_aic = float('inf')
        best_params = None
        best_model = None
        
        # Grid search for best parameters
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(ts_data['count'], order=(p,d,q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p,d,q)
                            best_model = fitted_model
                    except:
                        continue
        
        if best_model is None:
            print("Could not fit ARIMA model")
            return None
        
        # Generate forecasts
        forecast = best_model.forecast(steps=forecast_periods)
        forecast_ci = best_model.get_forecast(steps=forecast_periods).conf_int()
        
        # Create forecast dates
        last_date = ts_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                     periods=forecast_periods, freq='M')
        
        # Store results
        forecast_key = f"{category}_{item or 'default'}"
        self.forecasts[forecast_key] = {
            'historical': ts_data,
            'forecast': pd.Series(forecast, index=forecast_dates),
            'confidence_interval': forecast_ci,
            'model_params': best_params,
            'aic': best_aic
        }
        
        # Plot results
        plt.figure(figsize=(15, 6))
        
        # Historical data
        plt.plot(ts_data.index, ts_data['count'], label='Historical', 
                linewidth=2, marker='o')
        
        # Forecast
        plt.plot(forecast_dates, forecast, label='Forecast', 
                linewidth=2, marker='s', color='red')
        
        # Confidence intervals
        plt.fill_between(forecast_dates, 
                        forecast_ci.iloc[:, 0], 
                        forecast_ci.iloc[:, 1], 
                        color='red', alpha=0.3, label='95% Confidence Interval')
        
        plt.title(f'ARIMA Forecast for {category.title()}: {item or "Default"}')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print(f"\nARIMA Model Results:")
        print(f"Best parameters (p,d,q): {best_params}")
        print(f"AIC: {best_aic:.2f}")
        print(f"Forecast for next {forecast_periods} months:")
        for date, value in zip(forecast_dates, forecast):
            print(f"  {date.strftime('%Y-%m')}: {value:.1f}")
        
        return self.forecasts[forecast_key]
    
    def forecast_prophet(self, category='language', item=None, forecast_periods=12):
        """Perform Prophet forecasting (simplified version)"""
        # Note: This is a simplified implementation since Prophet might not be available
        # In practice, you would use Facebook Prophet for more sophisticated forecasting
        
        ts_data = self.prepare_time_series_data(category, item)
        if ts_data is None:
            return None
        
        # Simple trend + seasonal forecast
        ts_values = ts_data['count'].values
        
        # Calculate trend
        x = np.arange(len(ts_values))
        trend_coef = np.polyfit(x, ts_values, 1)
        trend = np.poly1d(trend_coef)
        
        # Simple seasonal component (monthly pattern)
        seasonal_pattern = []
        for month in range(12):
            month_data = []
            for i in range(month, len(ts_values), 12):
                month_data.append(ts_values[i])
            if month_data:
                seasonal_pattern.append(np.mean(month_data))
            else:
                seasonal_pattern.append(0)
        
        # Generate forecasts
        last_date = ts_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                     periods=forecast_periods, freq='M')
        
        forecasts = []
        for i, date in enumerate(forecast_dates):
            trend_value = trend(len(ts_values) + i)
            seasonal_value = seasonal_pattern[date.month - 1]
            forecast_value = max(0, trend_value + seasonal_value - np.mean(ts_values))
            forecasts.append(forecast_value)
        
        forecast_series = pd.Series(forecasts, index=forecast_dates)
        
        # Plot results
        plt.figure(figsize=(15, 6))
        plt.plot(ts_data.index, ts_data['count'], label='Historical', 
                linewidth=2, marker='o')
        plt.plot(forecast_dates, forecasts, label='Prophet-style Forecast', 
                linewidth=2, marker='s', color='green')
        
        plt.title(f'Trend + Seasonal Forecast for {category.title()}: {item or "Default"}')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print(f"\nTrend + Seasonal Forecast for next {forecast_periods} months:")
        for date, value in zip(forecast_dates, forecasts):
            print(f"  {date.strftime('%Y-%m')}: {value:.1f}")
        
        return forecast_series
    
    def compare_technologies(self, technologies, category='language'):
        """Compare trends between multiple technologies"""
        plt.figure(figsize=(15, 8))
        
        for tech in technologies:
            if tech in self.data[category].values:
                tech_data = (self.data[self.data[category] == tech]
                           .groupby('year_month')
                           .size())
                
                # Convert to percentage of total responses per month
                monthly_totals = self.data.groupby('year_month').size()
                tech_pct = (tech_data / monthly_totals * 100).fillna(0)
                
                plt.plot(range(len(tech_pct)), tech_pct.values, 
                        marker='o', linewidth=2, label=tech)
        
        plt.title(f'{category.title()} Trend Comparison (% of Monthly Responses)')
        plt.xlabel('Time Period')
        plt.ylabel('Percentage of Responses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive trend analysis report"""
        if self.data is None:
            print("No data loaded.")
            return
        
        print("=" * 60)
        print("           SURVEY TREND ANALYSIS REPORT")
        print("=" * 60)
        
        # Key statistics
        print(f"\nSurvey Period: {self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}")
        print(f"Total Responses: {len(self.data):,}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Top technologies
        print(f"\n--- TOP TECHNOLOGIES ---")
        print("Programming Languages:")
        for i, (lang, count) in enumerate(self.data['language'].value_counts().head(5).items(), 1):
            pct = count / len(self.data) * 100
            print(f"  {i}. {lang}: {count:,} ({pct:.1f}%)")
        
        print("\nIDEs:")
        for i, (ide, count) in enumerate(self.data['ide'].value_counts().head(5).items(), 1):
            pct = count / len(self.data) * 100
            print(f"  {i}. {ide}: {count:,} ({pct:.1f}%)")
        
        # Growth leaders
        lang_yearly = self.data.groupby(['year', 'language']).size().unstack(fill_value=0)
        lang_growth = lang_yearly.pct_change().mean() * 100
        
        print(f"\n--- GROWTH LEADERS ---")
        top_growth = lang_growth.sort_values(ascending=False).head(3)
        for lang, growth in top_growth.items():
            if not pd.isna(growth):
                print(f"  {lang}: {growth:+.1f}% avg yearly growth")
        
        print(f"\n--- DECLINING TECHNOLOGIES ---")
        declining = lang_growth.sort_values().head(3)
        for lang, growth in declining.items():
            if not pd.isna(growth) and growth < 0:
                print(f"  {lang}: {growth:+.1f}% avg yearly change")
        
        # Future predictions
        print(f"\n--- PREDICTIONS ---")
        print("Based on current trends, technologies to watch:")
        
        emerging_threshold = 5  # Less than 5% current market share but growing
        for lang in self.data['language'].unique():
            current_share = (self.data['language'] == lang).mean() * 100
            if current_share < emerging_threshold:
                growth_rate = lang_growth.get(lang, 0)
                if growth_rate > 10:  # Growing >10% per year
                    print(f"  {lang}: {current_share:.1f}% share, {growth_rate:+.1f}% growth")
        
        print("\n" + "=" * 60)

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SurveyTrendAnalyzer()
    
    # Generate sample data (replace with analyzer.load_data('your_file.csv') for real data)
    print("Generating sample survey data...")
    data = analyzer.generate_sample_data(years=5, responses_per_year=2000)
    
    # Explore the data
    print("\n1. Exploring survey data...")
    analyzer.explore_data()
    
    # Analyze historical patterns
    print("\n2. Analyzing historical patterns...")
    analyzer.analyze_historical_patterns()
    
    # Perform forecasting
    print("\n3. Performing ARIMA forecasting for Python...")
    analyzer.forecast_arima(category='language', item='Python', forecast_periods=12)
    
    print("\n4. Performing trend-based forecasting for JavaScript...")
    analyzer.forecast_prophet(category='language', item='JavaScript', forecast_periods=12)
    
    # Compare technologies
    print("\n5. Comparing technology trends...")
    analyzer.compare_technologies(['Python', 'JavaScript', 'TypeScript', 'Go'], category='language')
    
    # Generate comprehensive report
    print("\n6. Generating comprehensive report...")
    analyzer.generate_report()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nTo use with your own data:")
    print("1. Replace generate_sample_data() with load_data('your_file.csv')")
    print("2. Ensure your CSV has columns: date, language, ide, job_role")
    print("3. Run the analysis methods as demonstrated above")
    print("4. Customize categories and items for specific forecasting needs")
