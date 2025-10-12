import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Power Analyzer Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown('''
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
</style>
''', unsafe_allow_html=True)

st.markdown('<h1 class="main-title">⚡ Power Analyzer Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Solar System Monitoring & Analysis")
st.markdown("---")



# Feature overview
st.markdown("##  Application Features")

feat_col1, feat_col2 = st.columns(2)

with feat_col1:
    st.markdown('''
    <div class="feature-card">
        <h2> Root Cause Analysis</h2>
        <p style="font-size: 1.1rem; color: #666;">
            Real-time fault detection and diagnosis<br>
            • Harmonic distortion analysis<br>
            • Power factor monitoring<br>
            • Instant RCA recommendations<br>
            • Priority-based action plans
        </p>
    </div>
    ''', unsafe_allow_html=True)

with feat_col2:
    st.markdown('''
    <div class="feature-card">
        <h2> Power Forecasting</h2>
        <p style="font-size: 1.1rem; color: #666;">
            Predictive maintenance & energy forecasting<br>
            • RUL (Remaining Useful Life) predictions<br>
            • Yield loss analysis<br>
            • Trend visualization<br>
            • ML-powered forecasts
        </p>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)



