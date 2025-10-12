"""
Solar Power Forecasting System - Single Screen Dashboard
5KW Rooftop System with 85% Efficiency
Real-time monitoring with NASA POWER API
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# ==================== CONFIGURATION ====================
SYSTEM_CAPACITY = 5.0  # kW
SYSTEM_EFFICIENCY = 0.85
PANEL_AREA = 33  # m² (approximate for 5kW system)
TEMPERATURE_COEFFICIENT = -0.004  # Power loss per °C above 25°C
BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946

# BESCOM Tariff 2024-25 (Residential)
BESCOM_TARIFF = [
    {'limit': 50, 'rate': 4.15},
    {'limit': 100, 'rate': 5.75},
    {'limit': 200, 'rate': 7.60},
    {'limit': float('inf'), 'rate': 8.75}
]

# ==================== NASA POWER API ====================
@st.cache_data(ttl=3600)
def fetch_nasa_power_data(lat, lon, start_date, end_date):
    """Fetch solar irradiance and weather data from NASA POWER API"""
    
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    parameters = [
        "ALLSKY_SFC_SW_DWN",  # Solar irradiance (W/m²)
        "T2M",  # Temperature at 2m (°C)
        "T2M_MAX",  # Max temperature
        "T2M_MIN",  # Min temperature
        "RH2M",  # Relative humidity (%)
        "WS2M",  # Wind speed at 2m (m/s)
        "PRECTOTCORR",  # Precipitation (mm)
        "CLOUD_AMT"  # Cloud amount (%)
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
        df = df.replace(-999, np.nan)
        df = df.ffill().bfill()  # Forward and backward fill
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error fetching NASA POWER data: {str(e)}")
        return None

# ==================== POWER CALCULATION ====================
def calculate_power_output(irradiance, temperature, cloud_cover, humidity, wind_speed):
    """Calculate realistic power output based on weather conditions"""
    
    # Base power from irradiance (Standard Test Conditions: 1000 W/m²)
    if irradiance < 50:  # Too low irradiance
        return 0
    
    base_power = (irradiance / 1000) * SYSTEM_CAPACITY
    
    # Apply system efficiency
    power = base_power * SYSTEM_EFFICIENCY
    
    # Temperature derating (efficiency decreases with temperature)
    temp_factor = 1 + TEMPERATURE_COEFFICIENT * (temperature - 25)
    power *= max(temp_factor, 0.7)  # Minimum 70% efficiency
    
    # Cloud cover effect (exponential reduction)
    cloud_factor = 1 - (cloud_cover / 100) * 0.75
    power *= max(cloud_factor, 0.1)
    
    # Humidity effect (high humidity reduces efficiency slightly)
    if humidity > 60:
        humidity_factor = 1 - ((humidity - 60) / 200)
        power *= max(humidity_factor, 0.85)
    
    # Wind cooling effect (positive impact at high temps)
    if temperature > 35 and wind_speed > 3:
        wind_cooling = 1 + (wind_speed * 0.01)
        power *= min(wind_cooling, 1.05)
    
    return max(0, min(power, SYSTEM_CAPACITY))

# ==================== PERFORMANCE RATIO ====================
def calculate_pr_ratio(actual_power, irradiance):
    """Calculate Performance Ratio"""
    
    theoretical_power = (irradiance / 1000) * SYSTEM_CAPACITY * SYSTEM_EFFICIENCY
    
    if theoretical_power == 0:
        return 0
    
    pr = (actual_power / theoretical_power) * 100
    return min(pr, 100)

# ==================== CONDITION CLASSIFICATION ====================
def classify_condition(irradiance, cloud_cover, precipitation):
    """Classify weather condition with emoji"""
    
    if precipitation > 5:
        return "Rainy", "🌧️", "#4A90E2"
    elif cloud_cover > 75:
        return "Heavy Clouds", "☁️", "#95A5A6"
    elif cloud_cover > 50:
        return "Cloudy", "⛅", "#BDC3C7"
    elif cloud_cover > 25:
        return "Partly Cloudy", "🌤️", "#F39C12"
    elif irradiance > 600:
        return "Sunny", "☀️", "#F1C40F"
    else:
        return "Clear", "🌞", "#E67E22"

# ==================== COST CALCULATION ====================
def calculate_savings(daily_energy_kwh):
    """Calculate cost savings based on BESCOM tariff"""
    
    monthly_energy = daily_energy_kwh * 30
    yearly_energy = daily_energy_kwh * 365
    
    # Calculate monthly cost
    total_cost = 0
    remaining_energy = monthly_energy
    
    for i, slab in enumerate(BESCOM_TARIFF):
        if remaining_energy <= 0:
            break
        
        if i == 0:
            prev_limit = 0
        else:
            prev_limit = BESCOM_TARIFF[i-1]['limit']
        
        slab_energy = slab['limit'] - prev_limit if slab['limit'] != float('inf') else remaining_energy
        energy_in_slab = min(remaining_energy, slab_energy)
        
        total_cost += energy_in_slab * slab['rate']
        remaining_energy -= energy_in_slab
    
    return {
        'daily_savings': daily_energy_kwh * 6.5,  # Average rate
        'monthly_savings': total_cost,
        'yearly_savings': total_cost * 12,
        'monthly_energy': monthly_energy,
        'yearly_energy': yearly_energy
    }

# ==================== GENERATE HOURLY DATA ====================
def generate_hourly_profile(daily_data):
    """Generate realistic hourly power profile for today"""
    
    hours = list(range(24))
    hourly_power = []
    
    # Sun profile (bell curve from 6 AM to 6 PM)
    for hour in hours:
        if 6 <= hour <= 18:
            # Peak at noon (12)
            normalized_hour = (hour - 12) / 6
            sun_intensity = np.exp(-0.5 * (normalized_hour ** 2))
            
            # Apply weather conditions
            power = daily_data['power_output'] * sun_intensity
            hourly_power.append(power)
        else:
            hourly_power.append(0)
    
    return pd.DataFrame({
        'hour': hours,
        'power': hourly_power
    })

# ==================== STREAMLIT APP ====================
def main():
   
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .sub-header {
            text-align: center;
            color: #7f8c8d;
            margin-top: 0;
        }
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        .status-badge {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">☀️ Solar Power Forecasting System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">5KW Rooftop System | 85% Efficiency | Real-time NASA POWER API Integration</p>', unsafe_allow_html=True)
    
    # Auto-refresh toggle
    col_refresh1, col_refresh2, col_refresh3 = st.columns([2, 1, 2])
    with col_refresh2:
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
    
    st.markdown("---")
    
    # Fetch data
    with st.spinner("📡 Fetching real-time data from NASA POWER API..."):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = fetch_nasa_power_data(BENGALURU_LAT, BENGALURU_LON, start_date, end_date)
        
        if data is None:
            st.error("Failed to fetch data. Please try again later.")
            st.stop()
    
    # Calculate power output for all days
    data['power_output'] = data.apply(
        lambda row: calculate_power_output(
            row['ALLSKY_SFC_SW_DWN'],
            row['T2M'],
            row['CLOUD_AMT'],
            row['RH2M'],
            row['WS2M']
        ),
        axis=1
    )
    
    data['daily_energy'] = data['power_output'] * 6  # 6 hours effective sunlight
    data['pr_ratio'] = data.apply(
        lambda row: calculate_pr_ratio(row['power_output'], row['ALLSKY_SFC_SW_DWN']),
        axis=1
    )
    
    # Get latest data
    latest = data.iloc[-1]
    yesterday = data.iloc[-2] if len(data) > 1 else latest
    
    # Classify condition
    condition, emoji, color = classify_condition(
        latest['ALLSKY_SFC_SW_DWN'],
        latest['CLOUD_AMT'],
        latest.get('PRECTOTCORR', 0)
    )
    
    # Calculate savings
    avg_daily_energy = data['daily_energy'].tail(7).mean()
    savings = calculate_savings(avg_daily_energy)
    
    # ==================== MAIN METRICS ====================
    st.markdown("## 📊 Real-Time System Status")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        power_change = ((latest['power_output'] - yesterday['power_output']) / yesterday['power_output'] * 100) if yesterday['power_output'] > 0 else 0
        st.metric(
            "⚡ Current Power",
            f"{latest['power_output']:.2f} kW",
            f"{power_change:+.1f}%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "🔋 Daily Energy",
            f"{latest['daily_energy']:.2f} kWh",
            f"{(latest['power_output']/SYSTEM_CAPACITY)*100:.0f}% capacity"
        )
    
    with col3:
        st.metric(
            "📊 Performance Ratio",
            f"{latest['pr_ratio']:.1f}%",
            "Excellent" if latest['pr_ratio'] > 80 else "Good" if latest['pr_ratio'] > 60 else "Fair"
        )
    
    with col4:
        st.metric(
            "☀️ Solar Irradiance",
            f"{latest['ALLSKY_SFC_SW_DWN']:.0f} W/m²",
            f"{latest['CLOUD_AMT']:.0f}% clouds"
        )
    
    with col5:
        st.metric(
            "🌡️ Temperature",
            f"{latest['T2M']:.1f}°C",
            f"{latest['RH2M']:.0f}% humidity"
        )
    
    # Condition Badge
    st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <span class="status-badge" style="background-color: {color}; color: white;">
                {emoji} {condition}
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== ROW 1: GRAPHS ====================
    st.markdown("## 📈 Power Generation & Weather Analysis")
    
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        st.markdown("### ⚡ Today's Hourly Power Profile")
        hourly_data = generate_hourly_profile(latest)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=hourly_data['hour'],
            y=hourly_data['power'],
            mode='lines+markers',
            name='Power Output',
            fill='tozeroy',
            line=dict(color='#F39C12', width=3),
            marker=dict(size=8)
        ))
        fig1.add_hline(y=SYSTEM_CAPACITY, line_dash="dash", line_color="red", 
                      annotation_text="Max Capacity")
        fig1.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Power Output (kW)",
            height=350,
            showlegend=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_graph2:
        st.markdown("### ☀️ 30-Day Solar Irradiance Trend")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=data.index,
            y=data['ALLSKY_SFC_SW_DWN'],
            mode='lines',
            name='Irradiance',
            fill='tozeroy',
            line=dict(color='#E67E22', width=2)
        ))
        fig2.update_layout(
            xaxis_title="Date",
            yaxis_title="Irradiance (W/m²)",
            height=350,
            showlegend=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # ==================== ROW 2: MORE GRAPHS ====================
    col_graph3, col_graph4 = st.columns(2)
    
    with col_graph3:
        st.markdown("### 🔋 30-Day Energy Production")
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=data.index,
            y=data['daily_energy'],
            name='Daily Energy',
            marker_color='#3498DB'
        ))
        fig3.add_hline(y=data['daily_energy'].mean(), line_dash="dash", 
                      line_color="green", annotation_text="Average")
        fig3.update_layout(
            xaxis_title="Date",
            yaxis_title="Energy (kWh)",
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_graph4:
        st.markdown("### 📊 Performance Ratio Trend")
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=data.index,
            y=data['pr_ratio'],
            mode='lines+markers',
            name='PR Ratio',
            line=dict(color='#9B59B6', width=2),
            marker=dict(size=6)
        ))
        fig4.add_hline(y=75, line_dash="dash", line_color="orange", 
                      annotation_text="Target: 75%")
        fig4.update_layout(
            xaxis_title="Date",
            yaxis_title="PR Ratio (%)",
            height=350,
            showlegend=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== ROW 3: WEATHER & COST ====================
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("## 🌦️ Current Weather Conditions")
        
        weather_cols = st.columns(3)
        
        with weather_cols[0]:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, #E67E22 0%, #E74C3C 100%);">
                <div class="metric-value">{latest['T2M']:.1f}°C</div>
                <div class="metric-label">🌡️ Temperature</div>
                <div style="font-size: 0.8rem; margin-top: 5px;">
                    Max: {latest['T2M_MAX']:.1f}°C | Min: {latest['T2M_MIN']:.1f}°C
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with weather_cols[1]:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);">
                <div class="metric-value">{latest['RH2M']:.0f}%</div>
                <div class="metric-label">💧 Humidity</div>
                <div style="font-size: 0.8rem; margin-top: 5px;">
                    Wind: {latest['WS2M']:.1f} m/s
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with weather_cols[2]:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, #95A5A6 0%, #7F8C8D 100%);">
                <div class="metric-value">{latest['CLOUD_AMT']:.0f}%</div>
                <div class="metric-label">☁️ Cloud Cover</div>
                <div style="font-size: 0.8rem; margin-top: 5px;">
                    Precip: {latest.get('PRECTOTCORR', 0):.1f} mm
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Weather impact chart
        st.markdown("### 🌤️ Weather Impact on Performance")
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=data['CLOUD_AMT'],
            y=data['power_output'],
            mode='markers',
            marker=dict(
                size=10,
                color=data['T2M'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Temp (°C)")
            ),
            text=data.index.strftime('%Y-%m-%d'),
            hovertemplate='<b>%{text}</b><br>Cloud: %{x:.0f}%<br>Power: %{y:.2f} kW<extra></extra>'
        ))
        fig5.update_layout(
            xaxis_title="Cloud Cover (%)",
            yaxis_title="Power Output (kW)",
            height=300
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with col_right:
        st.markdown("## 💰 Cost Savings Analysis (BESCOM Tariff)")
        
        savings_cols = st.columns(3)
        
        with savings_cols[0]:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, #27AE60 0%, #229954 100%);">
                <div class="metric-value">₹{savings['daily_savings']:.2f}</div>
                <div class="metric-label">💵 Daily Savings</div>
            </div>
            """, unsafe_allow_html=True)
        
        with savings_cols[1]:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, #16A085 0%, #138D75 100%);">
                <div class="metric-value">₹{savings['monthly_savings']:.2f}</div>
                <div class="metric-label">📅 Monthly Savings</div>
            </div>
            """, unsafe_allow_html=True)
        
        with savings_cols[2]:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, #D4AC0D 0%, #B7950B 100%);">
                <div class="metric-value">₹{savings['yearly_savings']:.2f}</div>
                <div class="metric-label">📆 Yearly Savings</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # BESCOM Tariff Table
        st.markdown("### 📋 BESCOM Residential Tariff (2024-25)")
        
        tariff_df = pd.DataFrame([
            {"Slab": "0 - 50 units", "Rate (₹/kWh)": "4.15", "Monthly Cost": f"₹{50 * 4.15:.2f}"},
            {"Slab": "51 - 100 units", "Rate (₹/kWh)": "5.75", "Monthly Cost": f"₹{50 * 5.75:.2f}"},
            {"Slab": "101 - 200 units", "Rate (₹/kWh)": "7.60", "Monthly Cost": f"₹{100 * 7.60:.2f}"},
            {"Slab": "Above 200 units", "Rate (₹/kWh)": "8.75", "Monthly Cost": "Variable"}
        ])
        
        st.dataframe(tariff_df, use_container_width=True, hide_index=True)
        
        st.info(f"""
        **💡 Your Energy Stats:**
        - Monthly Production: **{savings['monthly_energy']:.2f} kWh**
        - Yearly Production: **{savings['yearly_energy']:.2f} kWh**
        - CO₂ Offset: **{savings['yearly_energy'] * 0.82:.2f} kg/year**
        - Trees Equivalent: **{(savings['yearly_energy'] * 0.82) / 21:.0f} trees/year**
        """)
    
    st.markdown("---")
    
    # ==================== SYSTEM INFO ====================
    st.markdown("## ⚙️ System Configuration & Statistics")
    
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    
    with info_col1:
        st.markdown("""
        **📍 Location**
        - Latitude: 12.9716°N
        - Longitude: 77.5946°E
        - City: Bengaluru
        """)
    
    with info_col2:
        st.markdown(f"""
        **⚡ System Specs**
        - Capacity: {SYSTEM_CAPACITY} kW
        - Efficiency: {SYSTEM_EFFICIENCY * 100}%
        - Panel Area: {PANEL_AREA} m²
        """)
    
    with info_col3:
        avg_power = data['power_output'].tail(7).mean()
        max_power = data['power_output'].max()
        st.markdown(f"""
        **📊 7-Day Stats**
        - Avg Power: {avg_power:.2f} kW
        - Max Power: {max_power:.2f} kW
        - Avg PR: {data['pr_ratio'].tail(7).mean():.1f}%
        """)
    
    with info_col4:
        total_energy_30d = data['daily_energy'].sum()
        st.markdown(f"""
        **🔋 30-Day Total**
        - Energy: {total_energy_30d:.2f} kWh
        - Savings: ₹{total_energy_30d * 6.5:.2f}
        - Capacity Factor: {(total_energy_30d / (SYSTEM_CAPACITY * 24 * 30)) * 100:.1f}%
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        🕐 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
        📡 Data Source: NASA POWER API | 
        💚 Powered by Streamlit
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
