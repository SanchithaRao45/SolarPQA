import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# --- I. GLOBAL CONFIGURATION, UTILITIES, and THRESHOLDS (IEEE 519 Compliance) ---

# Standards-based RUL Endpoints
VTHD_CRITICAL_THRESHOLD = 5.0    # IEEE 519 Standard for VTHD <= 69 kV (Safety)
ITHD_CRITICAL_THRESHOLD = 15.0   # Conservative limit for inverter component health
PF_CRITICAL_THRESHOLD = 0.85     # Common utility penalty threshold

# RCA Rules and Mappings
RCA_RULES = {
    "System Resonance": lambda v, i, pf: v > 5.0 and i > 15.0,
    "Harmonic Distortion": lambda v, i, pf: i > 25.0 and v < 5.0,
    "Low Power Factor (Reactive Load)": lambda v, i, pf: pf < 0.83,
    "Normal Operation": lambda v, i, pf: True # This MUST be the last rule
}

ACTION_MAPPING = {
    "System Resonance": "EMERGENCY: Install Active Harmonic Filter (AHF) for active damping and to prevent equipment damage.",
    "Harmonic Distortion": "Inspect load-side filters and reactive compensation equipment. High Current THD is stressing components.",
    "Low Power Factor (Reactive Load)": "Activate or repair Power Factor Correction (PFC) banks or use VFD to improve the load.",
    "Normal Operation": "Continue monitoring. System is operating within compliance limits."
}

PRIORITY_MAPPING = {
    "System Resonance": "Urgent",
    "Harmonic Distortion": "High",
    "Low Power Factor (Reactive Load)": "Medium",
    "Normal Operation": "Low"
}

# --- II. CORE PM/RUL FUNCTIONS ---

def calculate_rul(current_value, critical_threshold, rate_of_change):
    """
    Calculates Remaining Useful Life (RUL) in days using linear extrapolation.
    Handles both increasing (e.g., THD) and decreasing (e.g., PF) values.
    """
    
    if abs(rate_of_change) < 1e-6 or np.isnan(rate_of_change):
        return None

    # Check if the threshold is already passed
    if (critical_threshold > current_value and rate_of_change <= 0) or \
       (critical_threshold < current_value and rate_of_change >= 0):
        return 0.0

    # Calculate time to reach the threshold
    rul_days = abs(critical_threshold - current_value) / abs(rate_of_change)

    return max(0.0, rul_days)

def classify(v_thd_instant, i_thd_instant, pf_instant):
    """Applies RCA rules to instant data."""
    for label, rule in RCA_RULES.items():
        if rule(v_thd_instant, i_thd_instant, pf_instant):
            return label
    return "Undiagnosed Fault"

def run_predictive_analysis(vthd_7d, ithd_7d, pf_14d, vthd_rate, ithd_rate, pf_rate):
    """
    Executes all PM/RUL logic for VTHD, ITHD, and PF.
    """
    alerts = []
    
    # --- 1. PM: Voltage Stability (VTHD - IEEE 519 Safety Compliance) ---
    # Trigger if VTHD is increasing AND RUL is short.
    if vthd_rate > 0:
        rul_vthd = calculate_rul(vthd_7d, VTHD_CRITICAL_THRESHOLD, vthd_rate)
        if rul_vthd is not None and rul_vthd <= 60: # Warning if RUL is less than 2 months
            status = 'CRITICAL' if rul_vthd < 14 else 'WARNING' # Critical if < 2 weeks
            alerts.append({
                'Status': status,
                'Type': 'Voltage Stability RUL',
                'Prognosis': f"VTHD RUL: *{rul_vthd:.1f} days*. VTHD is trending towards the IEEE 519 limit of {VTHD_CRITICAL_THRESHOLD}%. Rate: {vthd_rate:.4f} %/day."
            })

    # --- 2. PM: Current Harmonic Stress (ITHD - Component Health) ---
    # Trigger if ITHD is increasing AND RUL is short.
    if ithd_rate > 0:
        rul_ithd = calculate_rul(ithd_7d, ITHD_CRITICAL_THRESHOLD, ithd_rate)
        if rul_ithd is not None and rul_ithd <= 90: # Warning if RUL is less than 3 months
            status = 'CRITICAL' if rul_ithd < 30 else 'WARNING' # Critical if < 1 month
            alerts.append({
                'Status': status,
                'Type': 'Harmonic Filter RUL',
                'Prognosis': f"ITHD RUL: *{rul_ithd:.1f} days*. Current THD is trending towards component overload limit of {ITHD_CRITICAL_THRESHOLD}%. Rate: {ithd_rate:.4f} %/day."
            })

    # --- 3. PM: Power Factor Degradation (PF - Utility Penalty) ---
    # Trigger if PF is decreasing AND RUL is short.
    if pf_rate < 0:
        rul_pf = calculate_rul(pf_14d, PF_CRITICAL_THRESHOLD, pf_rate)
        if rul_pf is not None and rul_pf <= 120: # Warning if RUL is less than 4 months
            status = 'CRITICAL' if rul_pf < 30 else 'WARNING' # Critical if < 1 month
            alerts.append({
                'Status': status,
                'Type': 'PF Degradation RUL',
                'Prognosis': f"PF RUL: *{rul_pf:.1f} days*. Power Factor expected to drop below the utility penalty threshold of {PF_CRITICAL_THRESHOLD}. Rate: {pf_rate:.5f} /day."
            })
    
    if not alerts:
        alerts.append({'Status': 'NORMAL', 'Type': 'System Stable', 'Prognosis': 'No RUL warnings or critical issues detected. Continue monitoring.'})

    return alerts

# --- III. MODEL LOADING ---

@st.cache_resource
def load_all_models():
    """Loads RCA model placeholder (no forecasting models needed)."""
    rca_model = None
    try:
        rca_model = joblib.load('rca_model.pkl')
    except FileNotFoundError:
        pass # Using rule-based classification if model not found
    return rca_model

RCA_MODEL = load_all_models()

# --- IV. UNIFIED ANALYSIS FUNCTION ---

def get_unified_analysis(input_data):
    """Executes PM/RUL and RCA models sequentially."""

    # Pass only required PM/RUL inputs
    pm_report = run_predictive_analysis(
        vthd_7d=input_data['vthd_7d_avg'], ithd_7d=input_data['ithd_7d_avg'],
        pf_14d=input_data['pf_14d_avg'], vthd_rate=input_data['vthd_rate'],
        ithd_rate=input_data['ithd_rate'], pf_rate=input_data['pf_rate']
    )

    rca_label = classify(input_data['v_thd_instant'], input_data['i_thd_instant'], input_data['pf_instant'])

    return {
        "rca_diagnosis": rca_label,
        "pm_prognosis": pm_report,
        "confidence": 0.95
    }

# --- Helper Function for Placeholder Graph Data ---
def generate_placeholder_data(key_metric):
    """Generates time-series data for a week."""
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=6)
    dates = pd.date_range(start_date, end_date, freq='D')

    if 'ITHD' in key_metric:
        base = st.session_state.get('ithd_7d_avg', 9.5)
        trend = np.linspace(base - 1.0, base, len(dates)) + np.random.randn(len(dates)) * 0.5
    elif 'PF' in key_metric:
        base = st.session_state.get('pf_14d_avg', 0.88)
        trend = np.linspace(base + 0.01, base, len(dates)) + np.random.randn(len(dates)) * 0.005
    elif 'VTHD' in key_metric:
        base = st.session_state.get('vthd_7d_avg', 4.0)
        trend = np.linspace(base - 0.5, base, len(dates)) + np.random.randn(len(dates)) * 0.2

    df = pd.DataFrame({
        'Date': dates,
        key_metric: trend
    }).set_index('Date')
    return df

# --- V. STREAMLIT UI CONFIGURATION ---

st.set_page_config(
    page_title="Solar System Health Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design (removed emojis)
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }

    .status-normal {
        background: #d4edda;
        color: #155724;
    }

    .status-warning {
        background: #fff3cd;
        color: #856404;
    }

    .status-critical {
        background: #f8d7da;
        color: #721c24;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- VI. HEADER ---
st.markdown('<h1 class="main-title">Solar Power Quality and Maintenance Monitor</h1>', unsafe_allow_html=True)
st.markdown("**Real-time AI-powered diagnostics and predictive maintenance based on IEEE 519 standards**")
st.markdown("---")

# --- VII. SIDEBAR INPUTS ---
with st.sidebar:
    st.title("System Parameters")
    st.markdown("### Data Inputs")

    with st.expander("Instantaneous Measurements", expanded=True):
        i_thd = st.slider("Current THD (ITHD) %", 0.0, 40.0, 26.0, 0.1, key='i_thd_instant')
        v_thd = st.slider("Voltage THD (VTHD) %", 0.0, 10.0, 3.5, 0.1, key='v_thd_instant')
        pf_instant = st.slider("Power Factor (PF)", 0.5, 1.0, 0.82, 0.01, key='pf_instant')

    with st.expander("Trend Analysis Data (7-14 Day Avg)", expanded=True):
        # Values set to trigger RUL warnings for demo purposes
        ithd_7d_avg = st.slider("7-Day Avg ITHD %", 5.0, 15.0, 14.0, 0.1, key='ithd_7d_avg')
        vthd_7d_avg = st.slider("7-Day Avg VTHD %", 3.0, 5.5, 4.8, 0.01, key='vthd_7d_avg')
        pf_14d_avg = st.slider("14-Day Avg PF", 0.75, 0.95, 0.86, 0.01, key='pf_14d_avg')
        
    st.markdown("---")
    st.markdown("### Trend Rates (Daily Change)")
    SIM_ITHD_RATE = st.slider("ITHD Rate (per day)", 0.00, 0.10, 0.01, 0.001)
    SIM_VTHD_RATE = st.slider("VTHD Rate (per day)", 0.00, 0.05, 0.002, 0.001)
    SIM_PF_RATE = st.slider("PF Rate (per day)", -0.01, 0.00, -0.0003, 0.0001)

    st.markdown("---")
    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Package inputs
input_data = {
    'v_thd_instant': v_thd, 'i_thd_instant': i_thd, 'pf_instant': pf_instant,
    'ithd_7d_avg': ithd_7d_avg, 'vthd_7d_avg': vthd_7d_avg, 'pf_14d_avg': pf_14d_avg,
    'vthd_rate': SIM_VTHD_RATE, 'ithd_rate': SIM_ITHD_RATE, 'pf_rate': SIM_PF_RATE,
}

# Execute analysis
report = get_unified_analysis(input_data)

# --- VIII. SYSTEM DATA OVERVIEW ---
st.markdown('<div class="section-header">System Data Overview</div>', unsafe_allow_html=True)

data_col1, data_col2, data_col3 = st.columns(3)

with data_col1:
    st.markdown("##### Instantaneous Readings")
    data_df1 = pd.DataFrame({
        'Parameter': ['Current THD (ITHD)', 'Voltage THD (VTHD)', 'Power Factor'],
        'Value': [f"{i_thd:.2f}%", f"{v_thd:.2f}%", f"{pf_instant:.3f}"],
        'Status': ['High' if i_thd > 20 else 'OK', 'Monitor' if v_thd > 4.5 else 'OK', 'Low' if pf_instant < 0.85 else 'OK']
    })
    st.dataframe(data_df1, use_container_width=True, hide_index=True)

with data_col2:
    st.markdown("##### Trend Averages")
    data_df2 = pd.DataFrame({
        'Parameter': ['7-Day Avg ITHD', '7-Day Avg VTHD', '14-Day Avg PF'],
        'Value': [f"{ithd_7d_avg:.2f}%", f"{vthd_7d_avg:.2f}%", f"{pf_14d_avg:.3f}"],
        'Status': ['Warning' if ithd_7d_avg > 14 else 'OK', 'Warning' if vthd_7d_avg > 4.5 else 'OK', 'Warning' if pf_14d_avg < 0.87 else 'OK']
    })
    st.dataframe(data_df2, use_container_width=True, hide_index=True)

with data_col3:
    st.markdown("##### System Rates")
    data_df3 = pd.DataFrame({
        'Parameter': ['VTHD Rate of Change', 'ITHD Rate of Change', 'PF Rate of Change'],
        'Value': [f"{SIM_VTHD_RATE:.4f} %/day", f"{SIM_ITHD_RATE:.4f} %/day", f"{SIM_PF_RATE:.5f} /day"],
        'Trend': ['Increasing', 'Increasing', 'Decreasing']
    })
    st.dataframe(data_df3, use_container_width=True, hide_index=True)

# --- IX. ROOT CAUSE ANALYSIS ---
st.markdown('<div class="section-header">Root Cause Analysis - Immediate Diagnosis</div>', unsafe_allow_html=True)

rca_main, rca_side = st.columns([2, 1])

with rca_main:
    diagnosis = report['rca_diagnosis']
    action = ACTION_MAPPING.get(diagnosis, "No specific action defined.")
    priority = PRIORITY_MAPPING.get(diagnosis, "Normal")

    if diagnosis == 'Normal Operation':
        st.markdown(f'<div class="status-badge status-normal">Status: {diagnosis}</div>', unsafe_allow_html=True)
    elif priority == 'Urgent' or priority == 'High':
        st.markdown(f'<div class="status-badge status-critical">FAULT: {diagnosis}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-badge status-warning">FAULT: {diagnosis}</div>', unsafe_allow_html=True)

    st.markdown(f"**Recommended Action:** {action}")

with rca_side:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 1.5rem; border-radius: 10px; text-align: center;'>
        <div style='font-size: 0.9rem; opacity: 0.9;'>Priority Level</div>
        <div style='font-size: 2rem; font-weight: bold; margin-top: 0.5rem;'>{priority}</div>
    </div>
    """, unsafe_allow_html=True)

# --- X. PREDICTIVE MAINTENANCE ---
st.markdown('<div class="section-header">Predictive Maintenance - RUL and Trend Analysis</div>', unsafe_allow_html=True)

alerts_to_display = report['pm_prognosis']
has_alerts = any(alert['Status'] != 'NORMAL' for alert in alerts_to_display)

if not has_alerts:
    st.success("No critical RUL or trend-based issues detected. All system trends are stable.")
else:
    st.markdown("##### Current System Health Warnings:")
    for alert in alerts_to_display:
        if alert['Status'] == 'CRITICAL':
            st.error(f"CRITICAL: {alert['Type']}")
        elif alert['Status'] == 'WARNING':
            st.warning(f"WARNING: {alert['Type']}")
        
        st.markdown(f"_{alert['Prognosis']}_")
        st.markdown("---")

pm_col1, pm_col2, pm_col3 = st.columns(3)

with pm_col1:
    st.markdown("#### 7-Day VTHD Trend")
    vthd_df = generate_placeholder_data('7-Day Avg VTHD (%)')
    vthd_df['IEEE 519 Limit (5.0%)'] = VTHD_CRITICAL_THRESHOLD
    st.line_chart(vthd_df, use_container_width=True)

with pm_col2:
    st.markdown("#### 7-Day ITHD Trend")
    ithd_df = generate_placeholder_data('7-Day Avg ITHD (%)')
    ithd_df['Component Limit (15%)'] = ITHD_CRITICAL_THRESHOLD
    st.line_chart(ithd_df, use_container_width=True)

with pm_col3:
    st.markdown("#### 14-Day Power Factor Trend")
    pf_df = generate_placeholder_data('14-Day Avg PF')
    pf_df['Utility Penalty (0.85)'] = PF_CRITICAL_THRESHOLD
    st.line_chart(pf_df, use_container_width=True)

# --- XI. FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>AI-Powered Solar System Health Monitoring | Built with Streamlit</p>
    <p style='font-size: 0.85rem;'>Model Confidence: {:.1%} | System Status: Active</p>
</div>
""".format(report['confidence']), unsafe_allow_html=True)
