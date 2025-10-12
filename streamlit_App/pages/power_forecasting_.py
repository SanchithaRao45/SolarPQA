"""
Solar Power Forecasting System - ML Model & Streamlit App
5KW Rooftop System with 85% Efficiency
Uses NASA POWER API for weather data
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
SYSTEM_CAPACITY = 5.0  # kW
SYSTEM_EFFICIENCY = 0.85
PANEL_AREA = 33  # m² (approximate for 5kW system)
TEMPERATURE_COEFFICIENT = -0.004  # Power loss per °C above 25°C
BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946

# BESCOM Tariff 2024-25 (Residential)
BESCOM_TARIFF = {
    'slab1': {'limit': 50, 'rate': 4.15},
    'slab2': {'limit': 100, 'rate': 5.75},
    'slab3': {'limit': 200, 'rate': 7.60},
    'slab4': {'limit': float('inf'), 'rate': 8.75}
}

# ==================== NASA POWER API ====================
def fetch_nasa_power_data(lat, lon, start_date, end_date):
    """Fetch solar irradiance and weather data from NASA POWER API"""
    
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    parameters = [
        "ALLSKY_SFC_SW_DWN",  # Solar irradiance
        "T2M",  # Temperature at 2m
        "T2M_MAX",  # Max temperature
        "T2M_MIN",  # Min temperature
        "RH2M",  # Relative humidity
        "WS2M",  # Wind speed at 2m
        "PRECTOTCORR",  # Precipitation
        "CLOUD_AMT"  # Cloud amount
    ]
    
    params = {
        "parameters": ",".join(parameters),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['properties']['parameter'])
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df = df.replace(-999, np.nan)  # Replace missing values
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching NASA POWER data: {str(e)}")
        return None

# ==================== FEATURE ENGINEERING ====================
def engineer_features(df):
    """Create features for ML model"""
    
    df = df.copy()
    
    # Time-based features
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['season'] = df['month'].apply(lambda x: 
        'winter' if x in [12, 1, 2] else
        'summer' if x in [3, 4, 5] else
        'monsoon' if x in [6, 7, 8, 9] else 'post_monsoon'
    )
    
    # Cyclical encoding for time features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Weather interaction features
    df['temp_irradiance'] = df['T2M'] * df['ALLSKY_SFC_SW_DWN']
    df['humidity_temp'] = df['RH2M'] * df['T2M']
    df['cloud_irradiance'] = df['CLOUD_AMT'] * df['ALLSKY_SFC_SW_DWN']
    
    # Rolling averages
    df['irradiance_ma3'] = df['ALLSKY_SFC_SW_DWN'].rolling(window=3, min_periods=1).mean()
    df['irradiance_ma7'] = df['ALLSKY_SFC_SW_DWN'].rolling(window=7, min_periods=1).mean()
    df['temp_ma3'] = df['T2M'].rolling(window=3, min_periods=1).mean()
    
    # Temperature effect on efficiency
    df['temp_efficiency_factor'] = 1 + TEMPERATURE_COEFFICIENT * (df['T2M'] - 25)
    
    return df

# ==================== POWER CALCULATION ====================
def calculate_power_output(irradiance, temperature, cloud_cover, humidity):
    """Calculate theoretical power output based on weather conditions"""
    
    # Base power from irradiance (W/m²)
    # Standard Test Conditions: 1000 W/m² irradiance
    base_power = (irradiance / 1000) * SYSTEM_CAPACITY
    
    # Apply system efficiency
    power = base_power * SYSTEM_EFFICIENCY
    
    # Temperature derating (efficiency decreases with temperature)
    temp_factor = 1 + TEMPERATURE_COEFFICIENT * (temperature - 25)
    power *= temp_factor
    
    # Cloud cover effect (reduce power based on cloud percentage)
    cloud_factor = 1 - (cloud_cover / 100) * 0.7
    power *= cloud_factor
    
    # Humidity effect (slight reduction at high humidity)
    humidity_factor = 1 - (max(0, humidity - 60) / 100) * 0.1
    power *= humidity_factor
    
    # Ensure non-negative
    power = max(0, power)
    
    return power

# ==================== TRAINING DATA GENERATION ====================
def generate_training_data(df):
    """Generate power output labels for training"""
    
    df = df.copy()
    
    # Calculate theoretical power output
    df['power_output'] = df.apply(
        lambda row: calculate_power_output(
            row['ALLSKY_SFC_SW_DWN'],
            row['T2M'],
            row['CLOUD_AMT'],
            row['RH2M']
        ),
        axis=1
    )
    
    # Add random variation to simulate real conditions (±5%)
    np.random.seed(42)
    df['power_output'] *= np.random.uniform(0.95, 1.05, size=len(df))
    
    # Calculate daily energy (kWh) - assuming 6 hours of effective sunlight
    df['daily_energy'] = df['power_output'] * 6
    
    return df

# ==================== ML MODEL ====================
class SolarPowerForecaster:
    """ML Model for Solar Power Forecasting"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
        
    def prepare_features(self, df):
        """Prepare features for model"""
        
        feature_cols = [
            'ALLSKY_SFC_SW_DWN', 'T2M', 'T2M_MAX', 'T2M_MIN',
            'RH2M', 'WS2M', 'CLOUD_AMT',
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'temp_irradiance', 'humidity_temp', 'cloud_irradiance',
            'irradiance_ma3', 'irradiance_ma7', 'temp_ma3',
            'temp_efficiency_factor'
        ]
        
        return df[feature_cols].fillna(method='ffill').fillna(method='bfill')
    
    def train(self, df):
        """Train the model"""
        
        X = self.prepare_features(df)
        y = df['power_output']
        
        self.feature_cols = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        self.is_trained = True
        
        return metrics, X_test, y_test, y_pred
    
    def predict(self, df):
        """Make predictions"""
        
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance"""
        
        if not self.is_trained:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

# ==================== PERFORMANCE RATIO ====================
def calculate_pr_ratio(actual_energy, irradiance, system_capacity):
    """Calculate Performance Ratio"""
    
    # Theoretical energy under standard conditions
    theoretical_energy = (irradiance / 1000) * system_capacity * 6  # 6 hours
    
    if theoretical_energy == 0:
        return 0
    
    pr = (actual_energy / theoretical_energy) * 100
    return min(pr, 100)  # Cap at 100%

# ==================== CONDITION CLASSIFICATION ====================
def classify_condition(irradiance, cloud_cover, precipitation):
    """Classify weather condition"""
    
    if precipitation > 5:
        return "Rainy", "🌧️"
    elif cloud_cover > 75:
        return "Heavy Clouds", "☁️"
    elif cloud_cover > 50:
        return "Cloudy", "⛅"
    elif cloud_cover > 25:
        return "Partly Cloudy", "🌤️"
    elif irradiance > 600:
        return "Sunny", "☀️"
    else:
        return "Clear", "🌞"

# ==================== COST CALCULATION ====================
def calculate_savings(daily_energy_kwh):
    """Calculate cost savings based on BESCOM tariff"""
    
    monthly_energy = daily_energy_kwh * 30
    
    # Calculate cost for each slab
    total_cost = 0
    remaining_energy = monthly_energy
    
    for slab in ['slab1', 'slab2', 'slab3', 'slab4']:
        slab_limit = BESCOM_TARIFF[slab]['limit']
        slab_rate = BESCOM_TARIFF[slab]['rate']
        
        if remaining_energy <= 0:
            break
        
        energy_in_slab = min(remaining_energy, slab_limit if slab != 'slab4' else remaining_energy)
        total_cost += energy_in_slab * slab_rate
        remaining_energy -= energy_in_slab
    
    return {
        'daily_savings': (daily_energy_kwh * 6.5),  # Average rate
        'monthly_savings': total_cost,
        'yearly_savings': total_cost * 12,
        'monthly_energy': monthly_energy
    }

# ==================== STREAMLIT APP ====================
def main():
    st.set_page_config(
        page_title="Solar Power Forecasting System",
        page_icon="☀️",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("☀️ Solar Power Forecasting System")
    st.markdown("### 5KW Rooftop System - ML-Based Prediction with NASA POWER API")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
        st.metric("System Capacity", f"{SYSTEM_CAPACITY} kW")
        st.metric("System Efficiency", f"{SYSTEM_EFFICIENCY * 100}%")
        st.metric("Panel Area", f"{PANEL_AREA} m²")
        
        st.markdown("---")
        st.header("📍 Location")
        st.write(f"Latitude: {BENGALURU_LAT}")
        st.write(f"Longitude: {BENGALURU_LON}")
        
        st.markdown("---")
        date_range = st.slider(
            "Historical Data (months)",
            min_value=6,
            max_value=36,
            value=12,
            step=6
        )
        
        forecast_days = st.slider(
            "Forecast Days",
            min_value=1,
            max_value=7,
            value=3
        )
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🤖 Train Model", 
        "🔮 Forecasting", 
        "📈 Analytics",
        "💰 Cost Analysis"
    ])
    
    # ==================== TAB 1: DASHBOARD ====================
    with tab1:
        st.header("Real-Time Dashboard")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔄 Fetch Live Data", type="primary"):
                with st.spinner("Fetching data from NASA POWER API..."):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)
                    
                    live_data = fetch_nasa_power_data(
                        BENGALURU_LAT, 
                        BENGALURU_LON,
                        start_date,
                        end_date
                    )
                    
                    if live_data is not None:
                        st.success("✅ Data fetched successfully!")
                        
                        # Get latest data
                        latest = live_data.iloc[-1]
                        
                        # Current metrics
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        with col_m1:
                            st.metric(
                                "☀️ Irradiance",
                                f"{latest['ALLSKY_SFC_SW_DWN']:.1f} W/m²"
                            )
                        
                        with col_m2:
                            st.metric(
                                "🌡️ Temperature",
                                f"{latest['T2M']:.1f}°C"
                            )
                        
                        with col_m3:
                            st.metric(
                                "☁️ Cloud Cover",
                                f"{latest['CLOUD_AMT']:.0f}%"
                            )
                        
                        with col_m4:
                            st.metric(
                                "💧 Humidity",
                                f"{latest['RH2M']:.0f}%"
                            )
                        
                        # Calculate current power
                        current_power = calculate_power_output(
                            latest['ALLSKY_SFC_SW_DWN'],
                            latest['T2M'],
                            latest['CLOUD_AMT'],
                            latest['RH2M']
                        )
                        
                        daily_energy = current_power * 6
                        
                        # Condition
                        condition, emoji = classify_condition(
                            latest['ALLSKY_SFC_SW_DWN'],
                            latest['CLOUD_AMT'],
                            latest.get('PRECTOTCORR', 0)
                        )
                        
                        # Performance metrics
                        col_p1, col_p2, col_p3 = st.columns(3)
                        
                        with col_p1:
                            st.metric(
                                "⚡ Current Power Output",
                                f"{current_power:.2f} kW",
                                f"{(current_power/SYSTEM_CAPACITY)*100:.1f}% of capacity"
                            )
                        
                        with col_p2:
                            st.metric(
                                "🔋 Daily Energy (Est.)",
                                f"{daily_energy:.2f} kWh"
                            )
                        
                        with col_p3:
                            pr = calculate_pr_ratio(
                                daily_energy,
                                latest['ALLSKY_SFC_SW_DWN'],
                                SYSTEM_CAPACITY
                            )
                            st.metric(
                                "📊 Performance Ratio",
                                f"{pr:.1f}%"
                            )
                        
                        # Condition display
                        st.markdown(f"### {emoji} Current Condition: **{condition}**")
                        
                        # 7-day trend
                        st.subheader("📈 7-Day Irradiance Trend")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=live_data.index,
                            y=live_data['ALLSKY_SFC_SW_DWN'],
                            mode='lines+markers',
                            name='Solar Irradiance',
                            line=dict(color='orange', width=3)
                        ))
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Irradiance (W/m²)",
                            hovermode='x unified',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.info("""
            ### 📌 System Status
            
            **Live Monitoring**
            - Real-time weather data
            - Power output calculation
            - Performance tracking
            
            **Data Source**
            - NASA POWER API
            - Updated daily
            - Historical archive
            """)
    
    # ==================== TAB 2: TRAIN MODEL ====================
    with tab2:
        st.header("🤖 Train ML Model")
        
        st.info("""
        Train a machine learning model using historical NASA POWER data to predict solar power output.
        The model uses weather parameters and engineered features for accurate predictions.
        """)
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Fetching training data from NASA POWER API..."):
                end_date = datetime.now()
                start_date = end_date - timedelta(days=date_range * 30)
                
                # Fetch data
                training_data = fetch_nasa_power_data(
                    BENGALURU_LAT,
                    BENGALURU_LON,
                    start_date,
                    end_date
                )
                
                if training_data is not None:
                    st.success(f"✅ Fetched {len(training_data)} days of data")
                    
                    # Engineer features
                    with st.spinner("Engineering features..."):
                        training_data = engineer_features(training_data)
                        training_data = generate_training_data(training_data)
                        st.session_state.training_data = training_data
                    
                    # Train model
                    with st.spinner("Training ML model..."):
                        progress_bar = st.progress(0)
                        
                        model = SolarPowerForecaster()
                        metrics, X_test, y_test, y_pred = model.train(training_data)
                        
                        st.session_state.model = model
                        
                        progress_bar.progress(100)
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("RMSE", f"{metrics['rmse']:.3f} kW")
                    with col2:
                        st.metric("MAE", f"{metrics['mae']:.3f} kW")
                    with col3:
                        st.metric("R² Score", f"{metrics['r2']:.3f}")
                    with col4:
                        st.metric("MAPE", f"{metrics['mape']:.2f}%")
                    
                    # Prediction vs Actual plot
                    st.subheader("📊 Model Performance - Actual vs Predicted")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=X_test.index,
                        y=y_test,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=X_test.index,
                        y=y_pred,
                        mode='lines',
                        name='Predicted',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Power Output (kW)",
                        hovermode='x unified',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance
                    st.subheader("🎯 Feature Importance")
                    importance_df = model.get_feature_importance()
                    
                    fig = px.bar(
                        importance_df.head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Top 10 Important Features'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.model is not None:
            st.success("✅ Model is trained and ready for forecasting!")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Model"):
                    joblib.dump(st.session_state.model, 'solar_forecast_model.pkl')
                    st.success("Model saved as 'solar_forecast_model.pkl'")
            
            with col2:
                if st.button("📥 Load Model"):
                    try:
                        st.session_state.model = joblib.load('solar_forecast_model.pkl')
                        st.success("Model loaded successfully!")
                    except:
                        st.error("No saved model found!")
    
    # ==================== TAB 3: FORECASTING ====================
    with tab3:
        st.header("🔮 Solar Power Forecasting")
        
        if st.session_state.model is None:
            st.warning("⚠️ Please train the model first in the 'Train Model' tab!")
        else:
            if st.button("📡 Generate Forecast", type="primary"):
                with st.spinner("Generating forecast..."):
                    end_date = datetime.now() + timedelta(days=forecast_days)
                    start_date = datetime.now()
                    
                    # Fetch forecast data
                    forecast_data = fetch_nasa_power_data(
                        BENGALURU_LAT,
                        BENGALURU_LON,
                        start_date,
                        end_date
                    )
                    
                    if forecast_data is not None:
                        # Engineer features
                        forecast_data = engineer_features(forecast_data)
                        
                        # Make predictions
                        predictions = st.session_state.model.predict(forecast_data)
                        forecast_data['predicted_power'] = predictions
                        forecast_data['predicted_energy'] = predictions * 6
                        
                        st.success(f"✅ Generated {forecast_days}-day forecast")
                        
                        # Display forecast
                        st.subheader(f"📅 {forecast_days}-Day Power Output Forecast")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=forecast_data.index,
                            y=forecast_data['predicted_power'],
                            mode='lines+markers',
                            name='Predicted Power',
                            line=dict(color='green', width=3),
                            marker=dict(size=8)
                        ))
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Power Output (kW)",
                            hovermode='x unified',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast table
                        st.subheader("📋 Detailed Forecast")
                        
                        forecast_display = forecast_data[[
                            'ALLSKY_SFC_SW_DWN', 'T2M', 'CLOUD_AMT',
                            'predicted_power', 'predicted_energy'
                        ]].copy()
                        
                        forecast_display.columns = [
                            'Irradiance (W/m²)', 'Temperature (°C)', 'Cloud Cover (%)',
                            'Power (kW)', 'Energy (kWh)'
                        ]
                        
                        # Add conditions
                        forecast_display['Condition'] = forecast_data.apply(
                            lambda row: classify_condition(
                                row['ALLSKY_SFC_SW_DWN'],
                                row['CLOUD_AMT'],
                                row.get('PRECTOTCORR', 0)
                            )[0],
                            axis=1
                        )
                        
                        st.dataframe(
                            forecast_display.style.format({
                                'Irradiance (W/m²)': '{:.1f}',
                                'Temperature (°C)': '{:.1f}',
                                'Cloud Cover (%)': '{:.0f}',
                                'Power (kW)': '{:.2f}',
                                'Energy (kWh)': '{:.2f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_power = forecast_data['predicted_power'].mean()
                            st.metric("Average Power", f"{avg_power:.2f} kW")
                        
                        with col2:
                            total_energy = forecast_data['predicted_energy'].sum()
                            st.metric("Total Energy", f"{total_energy:.2f} kWh")
                        
                        with col3:
                            peak_power = forecast_data['predicted_power'].max()
                            st.metric("Peak Power", f"{peak_power:.2f} kW")
    
    # ==================== TAB 4: ANALYTICS ====================
    with tab4:
        st.header("📈 Performance Analytics")
        
        if st.session_state.training_data is not None:
            data = st.session_state.training_data
            
            # Monthly analysis
            st.subheader("📊 Monthly Energy Production")
            
            monthly_data = data.groupby(data.index.to_period('M')).agg({
                'daily_energy': 'sum',
                'power_output': 'mean',
                'ALLSKY_SFC_SW_DWN': 'mean',
                'T2M': 'mean'
            })
            
            monthly_data.index = monthly_data.index.to_timestamp()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly_data.index,
                y=monthly_data['daily_energy'],
                name='Monthly Energy',
                marker_color='lightblue'
            ))
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Energy (kWh)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal analysis
            st.subheader("🌦️ Seasonal Performance")
            
            seasonal_data = data.groupby('season').agg({
                'power_output': 'mean',
                'daily_energy': 'mean',
                'ALLSKY_SFC_SW_DWN': 'mean'
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    seasonal_data,
                    y='power_output',
                    title='Average Power Output by Season',
                    color=seasonal_data.index
                )
                st.plotly_chart(fig, use_container_width=True)
