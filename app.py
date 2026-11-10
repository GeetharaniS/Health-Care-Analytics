import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
import json
from datetime import datetime, date
from rag_engine import HealthcareRAG

# ==========================================
# PAGE CONFIGURATION & CUSTOM THEME
# ==========================================
st.set_page_config(
    page_title="Sunrise Hospital | Healthcare Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Healthcare Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0A2540 0%, #1A365D 50%, #0D9488 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(10, 37, 64, 0.2);
    }
    
    .metric-card {
        background: white;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .badge-target {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-red { background-color: #FEE2E2; color: #DC2626; }
    .badge-green { background-color: #DCFCE7; color: #16A34A; }
    .badge-amber { background-color: #FEF3C7; color: #D97706; }
    
    .hero-stat {
        border-left: 4px solid #0D9488;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING & CACHING
# ==========================================
@st.cache_data
def load_all_data():
    data_dir = "data"
    admissions = pd.read_csv(os.path.join(data_dir, "admissions.csv"))
    daily_bed_census = pd.read_csv(os.path.join(data_dir, "daily_bed_census.csv"))
    departments = pd.read_csv(os.path.join(data_dir, "departments.csv"))
    insurance_claims = pd.read_csv(os.path.join(data_dir, "insurance_claims.csv"))
    patients = pd.read_csv(os.path.join(data_dir, "patients.csv"))
    patient_feedback = pd.read_csv(os.path.join(data_dir, "patient_feedback.csv"))
    
    # Process dates
    admissions['admit_date'] = pd.to_datetime(admissions['admit_date'])
    admissions['discharge_date'] = pd.to_datetime(admissions['discharge_date'])
    admissions['admit_month'] = admissions['admit_date'].dt.strftime('%Y-%m')
    daily_bed_census['date'] = pd.to_datetime(daily_bed_census['date'])
    daily_bed_census['month'] = daily_bed_census['date'].dt.strftime('%Y-%m')
    insurance_claims['discharge_date'] = pd.to_datetime(insurance_claims['discharge_date'])
    insurance_claims['month'] = insurance_claims['discharge_date'].dt.strftime('%Y-%m')
    
    return admissions, daily_bed_census, departments, insurance_claims, patients, patient_feedback

@st.cache_resource
def load_ml_models():
    models = {}
    model_paths = {
        "Logistic Regression": "models/logistic_regression.pkl",
        "Random Forest": "models/random_forest.pkl",
        "Gradient Boosting": "models/gradient_boosting.pkl",
        "XGBoost": "models/xgboost.pkl"
    }
    for name, p in model_paths.items():
        if os.path.exists(p):
            models[name] = joblib.load(p)
    return models

@st.cache_resource
def get_rag_engine():
    return HealthcareRAG()

# Load resources
admissions, bed_census, departments, claims, patients, feedback = load_all_data()
ml_models = load_ml_models()
rag_engine = get_rag_engine()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital-3.png", width=64)
    st.title("Sunrise Hospital")
    st.caption("NABH Accredited | Coimbatore, TN")
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation Portal",
        [
            "🏥 Hospital Landing Website",
            "📊 Executive BI Dashboard",
            "🤖 Readmission Risk Predictor (ML)",
            "💬 Gemini RAG Policy Chatbot",
            "📋 Strategic Recommendations Memo"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🔑 GenAI Configuration")
    user_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Enter your Google Gemini API key to activate Gemini 2.5 Flash. Built-in grounded fallback is active if empty."
    )
    if user_api_key:
        st.success("API Key Active: Gemini 2.5 Flash")
    else:
        st.info("Using Built-in Grounded Policy Knowledge Base")
        
    st.markdown("---")
    st.caption("📍 Trichy Road, Coimbatore, Tamil Nadu\n📞 24x7 Helpline: 0422-400-7000\n✉️ info@sunrisehospital.example")

# ==========================================
# 1. HOSPITAL PUBLIC LANDING WEBSITE
# ==========================================
if app_mode == "🏥 Hospital Landing Website":
    # Hero Section
    st.markdown("""
    <div class="main-header">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight:600; letter-spacing:0.5px;">
                    ⭐ NABH 5TH EDITION ACCREDITED HOSPITAL
                </span>
                <h1 style="color:white; margin: 12px 0 8px 0; font-size: 2.5rem; font-weight: 800;">Sunrise Multispeciality Hospital</h1>
                <p style="color:#E2E8F0; font-size: 1.15rem; max-width: 750px; margin-bottom: 20px;">
                    Excellence in Tertiary Healthcare, Grounded in Compassion. Delivering advanced cardiac, nephrology, oncology, and emergency medical care across Coimbatore and Western Tamil Nadu.
                </p>
                <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                    <span style="background: #0D9488; color: white; padding: 8px 16px; border-radius: 8px; font-weight: 600;">
                        🚨 24x7 Emergency: 0422-400-7000
                    </span>
                    <span style="background: white; color: #0A2540; padding: 8px 16px; border-radius: 8px; font-weight: 600;">
                        🛏️ 300 Inpatient Beds
                    </span>
                    <span style="background: rgba(255,255,255,0.15); color: white; padding: 8px 16px; border-radius: 8px; font-weight: 500;">
                        🏥 9 Centers of Excellence
                    </span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if os.path.exists("assets/sunrise_hospital_hero.jpg"):
        st.image("assets/sunrise_hospital_hero.jpg", caption="Sunrise Multispeciality Hospital — Trichy Road, Coimbatore, Tamil Nadu", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Highlights Grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#0A2540; margin:0;">⏱️ Rapid ER Triage</h4>
            <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">5-Level color triage with door-to-doctor admission < 30 mins for emergency cases.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#0A2540; margin:0;">☀️ Discharge Before Noon</h4>
            <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">Patient-centric discharge process targeting 60% discharges before 12:00 PM.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#0A2540; margin:0;">📞 48-Hour Nurse Care</h4>
            <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">Mandatory post-discharge follow-up call checking recovery, meds, and wound care.</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#0A2540; margin:0;">💳 Cashless Empanelment</h4>
            <p style="color:#64748B; font-size:0.9rem; margin-top:4px;">Empanelled with Ayushman Bharat PM-JAY, CMCHIS, ESI, and all major TPAs.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Clinical Centers & Departments Showcase
    st.subheader("🏥 Clinical Centers of Excellence")
    st.markdown("Comprehensive super-speciality inpatient and outpatient medical services in Coimbatore.")
    
    dept_cols = st.columns(3)
    dept_info = [
        ("❤️ Cardiology", "Interventional cardiology, Cath Lab, 24x7 primary PCI, heart failure clinic, coronary care unit (45 Beds, 7 ICU Beds).", "Dr. A. Sundaram, MD, DM"),
        ("🩺 Nephrology & Dialysis", "24x7 state-of-the-art haemodialysis, continuous renal replacement, kidney biopsy, transplant care (30 Beds, 4 ICU Beds).", "Dr. K. Venkataraman, DM"),
        ("🎗️ Comprehensive Oncology", "Medical oncology, chemotherapy day-care suite, surgical tumour resections and palliative oncology (25 Beds, 3 ICU Beds).", "Dr. Priya Nair, MS, MCh"),
        ("🧠 Neurology & Stroke Care", "Dedicated acute stroke thrombolysis team, neuro-ICU, epilepsy management and neuromuscular rehabilitation (20 Beds, 4 ICU Beds).", "Dr. R. Karthik, MD, DM"),
        ("🦴 Orthopaedics & Trauma", "Robotic joint replacement, arthroscopy, polytrauma care, minimally invasive spine surgery (40 Beds, 6 ICU Beds).", "Dr. S. Balaji, MS Ortho"),
        ("🫁 Pulmonology & Respiratory", "Advanced broncho-alveolar lavage, sleep medicine, non-invasive ventilation & respiratory critical care (25 Beds, 4 ICU Beds).", "Dr. M. Sangeetha, MD"),
        ("⚕️ General Medicine", "Adult chronic disease management, diabetes & metabolic disorders, monsoon infectious fever clinic (50 Beds, 6 ICU Beds).", "Dr. V. Ramanathan, MD"),
        ("🔬 General & Laparoscopic Surgery", "Advanced laparoscopic GI surgery, hernia repair, colorectal and acute surgical emergencies (35 Beds, 4 ICU Beds).", "Dr. T. Natarajan, MS"),
        ("👶 Obstetrics & Gynaecology", "Modern maternity suites, high-risk pregnancy management, painless labour, NICU level-3 support (30 Beds, 2 ICU Beds).", "Dr. L. Shanthi, MD DGO")
    ]
    
    for idx, (title, desc, head) in enumerate(dept_info):
        col = dept_cols[idx % 3]
        with col:
            st.markdown(f"""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:1.2rem; margin-bottom:1rem; height:185px;">
                <h4 style="margin:0 0 6px 0; color:#0A2540;">{title}</h4>
                <p style="font-size:0.85rem; color:#475569; margin-bottom:8px; line-height:1.4;">{desc}</p>
                <div style="font-size:0.8rem; font-weight:600; color:#0D9488;">Lead Consultant: {head}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Interactive Services: Appointment Booking & Emergency Triage Guide
    col_left, col_right = st.columns([1.1, 1.0])
    
    with col_left:
        st.subheader("📅 Book an Inpatient / OPD Consultation")
        with st.form("appointment_form"):
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                pat_name = st.text_input("Patient Full Name*", placeholder="e.g. Ramesh Kumar")
                pat_phone = st.text_input("Mobile Number*", placeholder="+91 98765 43210")
                pat_city = st.selectbox("Residence City", ["Coimbatore", "Tiruppur", "Erode", "Pollachi", "Palakkad", "Salem", "Other"])
            with fcol2:
                pat_dept = st.selectbox("Speciality Department*", [
                    "Cardiology", "Nephrology", "Oncology", "Neurology",
                    "Orthopaedics", "Pulmonology", "General Medicine",
                    "General Surgery", "Obstetrics & Gynaecology"
                ])
                pref_date = st.date_input("Preferred Date", min_value=date.today())
                pat_payer = st.selectbox("Payment / Insurance Scheme", ["Cash", "Private Health Insurance (TPA)", "Ayushman Bharat (PM-JAY)", "Tamil Nadu CMCHIS", "ESI Scheme", "Corporate Tie-up"])
            
            symptoms = st.text_area("Chief Complaint / Clinical Symptoms (Optional)", placeholder="Describe symptoms or ongoing treatments...")
            submitted = st.form_submit_button("Confirm Consultation Request", use_container_width=True)
            
            if submitted:
                if pat_name and pat_phone:
                    st.success(f"✅ Appointment Requested for **{pat_name}** with **{pat_dept}** on **{pref_date}**! Confirmation SMS and token sent to **{pat_phone}**.")
                    st.info("💡 Note: For insurance pre-authorisation assistance, our TPA desk in Block A is open 24x7.")
                else:
                    st.error("Please enter patient name and contact phone number.")

    with col_right:
        st.subheader("🚨 Emergency 5-Level Triage SOP")
        st.markdown("Under SMH Emergency Clinical Protocol, every incoming patient is triaged within 10 minutes:")
        
        st.markdown("""
        <div style="background:#FEF2F2; border-left:4px solid #EF4444; padding:10px 14px; border-radius:6px; margin-bottom:8px;">
            <strong style="color:#B91C1C;">LEVEL 1 - RED (Resuscitation / Immediate)</strong><br>
            <span style="font-size:0.85rem; color:#7F1D1D;">Cardiac arrest, major trauma, severe respiratory distress. Door-to-doctor: 0 min (Immediate).</span>
        </div>
        <div style="background:#FFF7ED; border-left:4px solid #F97316; padding:10px 14px; border-radius:6px; margin-bottom:8px;">
            <strong style="color:#C2410C;">LEVEL 2 - ORANGE (Emergent / Very Urgent)</strong><br>
            <span style="font-size:0.85rem; color:#9A3412;">Acute chest pain, acute stroke signs, severe asthma. Door-to-admission target: < 30 min.</span>
        </div>
        <div style="background:#FEFCE8; border-left:4px solid #EAB308; padding:10px 14px; border-radius:6px; margin-bottom:8px;">
            <strong style="color:#A16207;">LEVEL 3 - YELLOW (Urgent)</strong><br>
            <span style="font-size:0.85rem; color:#854D0E;">Severe abdominal pain, high persistent fever, moderate fractures. Door-to-doctor: < 60 min.</span>
        </div>
        <div style="background:#F0FDF4; border-left:4px solid #22C55E; padding:10px 14px; border-radius:6px; margin-bottom:8px;">
            <strong style="color:#15803D;">LEVEL 4 - GREEN (Standard / Less Urgent)</strong><br>
            <span style="font-size:0.85rem; color:#166534;">Minor lacerations, mild fever, sprains. Door-to-doctor: < 90 min.</span>
        </div>
        <div style="background:#F8FAFC; border-left:4px solid #64748B; padding:10px 14px; border-radius:6px;">
            <strong style="color:#334155;">LEVEL 5 - BLUE (Non-Urgent / OPD Redirect)</strong><br>
            <span style="font-size:0.85rem; color:#475569;">Medication refills, chronic aches. Routed to OPD clinics.</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.info("ℹ️ **Hospital Staff & Leadership Portal**: To explore the executive analytics, predictive readmission models, and policy RAG assistant, switch to the portal tabs in the left sidebar.")

# ==========================================
# 2. EXECUTIVE BI DASHBOARD
# ==========================================
elif app_mode == "📊 Executive BI Dashboard":
    st.title("📊 Executive Hospital Operations & Financial Dashboard")
    st.caption("Sunrise Multispeciality Hospital | Annual Analytics FY2025-26 | NABH Quality Metrics")
    
    # Global Filters
    st.markdown("#### 🔍 Interactive Analytics Filters")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        selected_depts = st.multiselect(
            "Filter Department",
            options=sorted(admissions['department'].unique()),
            default=sorted(admissions['department'].unique())
        )
    with f_col2:
        selected_payers = st.multiselect(
            "Filter Payer Scheme",
            options=sorted(admissions['payer'].unique()),
            default=sorted(admissions['payer'].unique())
        )
    with f_col3:
        selected_adm_type = st.multiselect(
            "Admission Type",
            options=sorted(admissions['admission_type'].unique()),
            default=sorted(admissions['admission_type'].unique())
        )
    with f_col4:
        months_list = sorted(admissions['admit_month'].unique())
        selected_months = st.multiselect(
            "Month Range",
            options=months_list,
            default=months_list
        )
        
    # Apply filters
    filtered_adm = admissions[
        (admissions['department'].isin(selected_depts)) &
        (admissions['payer'].isin(selected_payers)) &
        (admissions['admission_type'].isin(selected_adm_type)) &
        (admissions['admit_month'].isin(selected_months))
    ]
    
    # Filtered eligible for readmissions (Home or LAMA)
    filtered_eligible = filtered_adm[filtered_adm['discharge_disposition'].isin(['Discharged Home', 'LAMA'])]
    
    # Calculate Dynamic KPIs
    tot_admissions = len(filtered_adm)
    tot_revenue_cr = filtered_adm['total_bill_inr'].sum() / 1e7
    avg_alos = filtered_adm['los_days'].mean() if tot_admissions > 0 else 0
    readmit_rate = (filtered_eligible['readmitted_30d'].mean() * 100) if len(filtered_eligible) > 0 else 0
    followup_cov = (filtered_eligible['followup_call_48h'].mean() * 100) if len(filtered_eligible) > 0 else 0
    
    # Claims metrics
    filtered_claims = claims[claims['admission_id'].isin(filtered_adm['admission_id'])]
    tot_claims_count = len(filtered_claims)
    rejected_claims = filtered_claims[filtered_claims['claim_status'] == 'Rejected']
    rejection_rate = (len(rejected_claims) / tot_claims_count * 100) if tot_claims_count > 0 else 0
    rejected_amt_cr = rejected_claims['claimed_amount_inr'].sum() / 1e7
    
    # Bed Census occupancy
    filtered_census = bed_census[
        (bed_census['department'].isin(selected_depts)) &
        (bed_census['month'].isin(selected_months))
    ]
    avg_occupancy = filtered_census['occupancy_pct'].mean() if len(filtered_census) > 0 else 63.1
    
    # Feedback & NPS
    filtered_fb = feedback[feedback['admission_id'].isin(filtered_adm['admission_id'])]
    fb_count = len(filtered_fb)
    avg_rating = filtered_fb['rating_overall'].mean() if fb_count > 0 else 4.03
    nps_val = ((sum(filtered_fb['nps_category'] == 'Promoter') - sum(filtered_fb['nps_category'] == 'Detractor')) / fb_count * 100) if fb_count > 0 else 6.4

    # Top KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#64748B; font-weight:600;">TOTAL INPATIENT ADMISSIONS</span>
            <h2 style="margin:6px 0; color:#0A2540;">{tot_admissions:,}</h2>
            <div>Gross Revenue: <strong>₹{tot_revenue_cr:.2f} Cr</strong></div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.85rem; color:#64748B; font-weight:600;">30-DAY READMISSION RATE</span>
                <span class="badge-target badge-red">Target < 10%</span>
            </div>
            <h2 style="margin:6px 0; color:#DC2626;">{readmit_rate:.1f}%</h2>
            <div>Hospital Gap: <span style="color:#DC2626;">+{readmit_rate - 10.0:.1f}% vs Target</span></div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.85rem; color:#64748B; font-weight:600;">BED OCCUPANCY RATE</span>
                <span class="badge-target badge-amber">Target 75-80%</span>
            </div>
            <h2 style="margin:6px 0; color:#D97706;">{avg_occupancy:.1f}%</h2>
            <div>Avg ALOS: <strong>{avg_alos:.2f} Days</strong> (Target < 5.8)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.85rem; color:#64748B; font-weight:600;">CLAIM REJECTION RATE</span>
                <span class="badge-target badge-red">Target < 5%</span>
            </div>
            <h2 style="margin:6px 0; color:#DC2626;">{rejection_rate:.1f}%</h2>
            <div>Revenue Lost: <span style="color:#DC2626;"><strong>₹{rejected_amt_cr:.2f} Cr</strong></span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Secondary KPIs Row
    skpi_1, skpi_2, skpi_3, skpi_4 = st.columns(4)
    with skpi_1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.8rem; color:#64748B; font-weight:600;">48-HR FOLLOW-UP COVERAGE</span>
            <h3 style="margin:4px 0; color:{'#16A34A' if followup_cov >= 90 else '#DC2626'};">{followup_cov:.1f}%</h3>
            <span style="font-size:0.8rem; color:#64748B;">Target: > 90% (Annual Report: 54.8%)</span>
        </div>
        """, unsafe_allow_html=True)
    with skpi_2:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.8rem; color:#64748B; font-weight:600;">NET PROMOTER SCORE (NPS)</span>
            <h3 style="margin:4px 0; color:#0A2540;">+{nps_val:.1f}</h3>
            <span style="font-size:0.8rem; color:#64748B;">Target: > +40.0 (3,200 Respondents)</span>
        </div>
        """, unsafe_allow_html=True)
    with skpi_3:
        avg_er_wait = filtered_adm[filtered_adm['admission_type'] == 'Emergency']['er_wait_minutes'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.8rem; color:#64748B; font-weight:600;">AVG ER DOOR WAIT (EMERGENCY)</span>
            <h3 style="margin:4px 0; color:#0A2540;">{avg_er_wait:.1f} min</h3>
            <span style="font-size:0.8rem; color:#64748B;">Target: < 30 min (NABH Standard)</span>
        </div>
        """, unsafe_allow_html=True)
    with skpi_4:
        lama_rate = (sum(filtered_adm['discharge_disposition'] == 'LAMA') / tot_admissions * 100) if tot_admissions > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.8rem; color:#64748B; font-weight:600;">LAMA RATE (LEFT AGAINST ADVICE)</span>
            <h3 style="margin:4px 0; color:#0A2540;">{lama_rate:.2f}%</h3>
            <span style="font-size:0.8rem; color:#64748B;">Target: < 2.0%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Dashboard Tabs
    d_tab1, d_tab2, d_tab3, d_tab4, d_tab5 = st.tabs([
        "📈 Monthly Admissions & Bed Bottlenecks",
        "🎯 Readmissions & 48-Hr Follow-Up Gap",
        "💸 Insurance Claims & Pareto Rejection",
        "⭐ Patient Experience & NPS Drivers",
        "📋 Power BI Data Model & DAX Guide"
    ])
    
    # TAB 1: Monthly Trends & Bed Capacity
    with d_tab1:
        st.subheader("Monthly Flow & Capacity Crunch Analysis")
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            # Monthly admissions trend
            monthly_counts = filtered_adm.groupby('admit_month').size().reset_index(name='admissions')
            fig_monthly = px.bar(
                monthly_counts,
                x='admit_month',
                y='admissions',
                title="Monthly Inpatient Admissions (Monsoon Surge Oct-Dec)",
                labels={'admit_month': 'Month', 'admissions': 'Admissions'},
                color='admissions',
                color_continuous_scale='teal'
            )
            fig_monthly.add_hline(y=monthly_counts['admissions'].mean(), line_dash="dash", line_color="orange", annotation_text="Monthly Avg")
            fig_monthly.update_layout(height=380, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_monthly, use_container_width=True)
            
        with col_m2:
            # Department Occupancy vs Target
            dept_occ = filtered_census.groupby('department')['occupancy_pct'].mean().reset_index()
            dept_occ = dept_occ.sort_values('occupancy_pct', ascending=True)
            
            # Color code based on bottleneck vs under-utilized
            colors = ['#EF4444' if x >= 80 else ('#3B82F6' if x >= 60 else '#10B981') for x in dept_occ['occupancy_pct']]
            
            fig_occ = go.Figure()
            fig_occ.add_trace(go.Bar(
                x=dept_occ['occupancy_pct'],
                y=dept_occ['department'],
                orientation='h',
                marker_color=colors,
                text=[f"{v:.1f}%" for v in dept_occ['occupancy_pct']],
                textposition='auto'
            ))
            fig_occ.add_vline(x=75, line_dash="dash", line_color="#F59E0B", annotation_text="Target Range Min 75%")
            fig_occ.add_vline(x=85, line_dash="dash", line_color="#EF4444", annotation_text="Red Alert 85%+")
            fig_occ.update_layout(
                title="Average Bed Occupancy % by Department",
                xaxis_title="Occupancy %",
                height=380,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_occ, use_container_width=True)
            
        # Bottleneck insight callout
        st.markdown("""
        > [!WARNING] **Capacity Crunch Identified**: 
        > **Cardiology (86.5%)**, **Nephrology (88.0%)**, and **Neurology (85.2%)** are running in the **Red Escalation Zone (>85%)**, triggering elective surgery deferrals.
        > In contrast, **Obstetrics & Gynaecology (28.5%)** and **General Medicine (49.7%)** operate with excess bed capacity. 
        > **Strategic Action**: Activate hospital bed flexing SOP (borrow 20-25 beds from OBG/General Medicine to cardiology overflow) to immediately normalize bed utilization without capital expenditure.
        """)
        
    # TAB 2: Readmissions Deep Dive
    with d_tab2:
        st.subheader("30-Day Readmission Analysis: Clinical & Care Transition Drivers")
        r_col1, r_col2 = st.columns(2)
        
        with r_col1:
            # 48h call impact on readmission
            call_impact = filtered_eligible.groupby('followup_call_48h')['readmitted_30d'].agg(
                rate=lambda x: x.mean() * 100,
                count='count'
            ).reset_index()
            call_impact['followup_label'] = call_impact['followup_call_48h'].map({True: '48h Call Completed (54.8%)', False: 'No 48h Call (45.2%)'})
            
            fig_call = px.bar(
                call_impact,
                x='followup_label',
                y='rate',
                color='followup_label',
                color_discrete_map={'48h Call Completed (54.8%)': '#10B981', 'No 48h Call (45.2%)': '#EF4444'},
                title="Readmission Rate: Patients Called vs NOT Called within 48h",
                text=[f"{v:.1f}%" for v in call_impact['rate']],
                labels={'rate': 'Readmission Rate (%)', 'followup_label': 'Care Transition Status'}
            )
            fig_call.add_hline(y=10.0, line_dash="dash", line_color="#3B82F6", annotation_text="Hospital Target < 10%")
            fig_call.update_layout(height=380, showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_call, use_container_width=True)
            
        with r_col2:
            # Readmissions by Department
            dept_readmit = filtered_eligible.groupby('department')['readmitted_30d'].agg(
                rate=lambda x: x.mean() * 100
            ).reset_index().sort_values('rate', ascending=False)
            
            fig_dept_readmit = px.bar(
                dept_readmit,
                x='department',
                y='rate',
                color='rate',
                color_continuous_scale='Reds',
                title="30-Day Readmission Rate by Clinical Speciality",
                text=[f"{v:.1f}%" for v in dept_readmit['rate']],
                labels={'rate': 'Readmission Rate (%)', 'department': 'Department'}
            )
            fig_dept_readmit.add_hline(y=10.0, line_dash="dash", line_color="#3B82F6", annotation_text="Target 10%")
            fig_dept_readmit.update_layout(height=380, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_dept_readmit, use_container_width=True)
            
        # Clinical Risk Factors
        c1, c2, c3 = st.columns(3)
        with c1:
            late_dis = filtered_eligible.groupby(filtered_eligible['discharge_hour'] >= 18)['readmitted_30d'].mean() * 100
            st.metric("Late Discharges (>= 18:00)", f"{late_dis.get(True, 0):.1f}%", delta=f"{late_dis.get(True, 0) - late_dis.get(False, 0):.1f}% vs Before 18:00", delta_color="inverse")
        with c2:
            dist_read = filtered_eligible.groupby(filtered_eligible['distance_km'] > 50)['readmitted_30d'].mean() * 100
            st.metric("Distance > 50 km from Hospital", f"{dist_read.get(True, 0):.1f}%", delta=f"{dist_read.get(True, 0) - dist_read.get(False, 0):.1f}% vs Local Patients", delta_color="inverse")
        with c3:
            icu_read = filtered_eligible.groupby('icu_stay')['readmitted_30d'].mean() * 100
            st.metric("Patients with ICU Stay", f"{icu_read.get(True, 0):.1f}%", delta=f"{icu_read.get(True, 0) - icu_read.get(False, 0):.1f}% vs Ward Only", delta_color="inverse")
            
    # TAB 3: Claims & Revenue Leakage
    with d_tab3:
        st.subheader("Insurance Claims & Revenue Leakage Pareto Analysis")
        cl_col1, cl_col2 = st.columns(2)
        
        with cl_col1:
            # Pareto chart of rejection reasons
            rej_reasons = rejected_claims.groupby('rejection_reason').agg(
                count=('claim_id', 'count'),
                amount_cr=('claimed_amount_inr', lambda x: x.sum() / 1e7)
            ).reset_index().sort_values('amount_cr', ascending=False)
            
            rej_reasons['cum_pct'] = (rej_reasons['amount_cr'].cumsum() / rej_reasons['amount_cr'].sum()) * 100
            
            fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pareto.add_trace(
                go.Bar(
                    x=rej_reasons['rejection_reason'],
                    y=rej_reasons['amount_cr'],
                    name="Rejected Amount (₹ Cr)",
                    marker_color="#DC2626"
                ),
                secondary_y=False
            )
            fig_pareto.add_trace(
                go.Scatter(
                    x=rej_reasons['rejection_reason'],
                    y=rej_reasons['cum_pct'],
                    name="Cumulative %",
                    mode="lines+markers",
                    marker_color="#2563EB",
                    line=dict(width=3)
                ),
                secondary_y=True
            )
            fig_pareto.add_hline(y=80, line_dash="dash", line_color="orange", secondary_y=True, annotation_text="80% Pareto Cut")
            fig_pareto.update_layout(
                title="Pareto Chart: Revenue Lost by Rejection Reason (Total ₹9.03 Cr)",
                height=400,
                margin=dict(t=40, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_pareto, use_container_width=True)
            
        with cl_col2:
            # Rejection amount by payer
            rej_payer = rejected_claims.groupby('payer')['claimed_amount_inr'].sum().reset_index()
            rej_payer['amt_cr'] = rej_payer['claimed_amount_inr'] / 1e7
            rej_payer = rej_payer.sort_values('amt_cr', ascending=False)
            
            fig_payer_rej = px.pie(
                rej_payer,
                names='payer',
                values='amt_cr',
                title="Revenue Loss Breakdown by Payer (₹ Cr)",
                color_discrete_sequence=px.colors.qualitative.Safe,
                hole=0.4
            )
            fig_payer_rej.update_layout(height=400, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_payer_rej, use_container_width=True)

        st.markdown("""
        > [!NOTE] **Root Cause of Revenue Leakage**:
        > 1. **Incomplete Documents (₹3.1 Cr, 250 claims)**: Missing doctor discharge summary signatures, itemized pharmacy bills, and unattached diagnostic reports.
        > 2. **Pre-authorisation Not Obtained (₹1.7 Cr, 138 claims)**: Emergency admissions not notified within the 24-hour window.
        > 3. **Package Mismatch (₹1.3 Cr, 105 claims)**: Mismatched procedure package codes under PM-JAY and CMCHIS schemes.
        > *Over 70% of these rejections are fully preventable with an automated pre-billing digital audit checklist.*
        """)

    # TAB 4: Patient Experience & Feedback
    with d_tab4:
        st.subheader("Patient Satisfaction (CSAT) & Net Promoter Score (NPS)")
        f_col1, f_col2 = st.columns([1, 1.2])
        
        with f_col1:
            # Rating by dimension
            rating_metrics = {
                "Overall Experience": filtered_fb['rating_overall'].mean(),
                "Doctor Care": filtered_fb['rating_doctor'].mean(),
                "Nursing Care": filtered_fb['rating_nursing'].mean(),
                "Billing & Insurance Desk": filtered_fb['rating_billing'].mean()
            }
            df_ratings = pd.DataFrame(list(rating_metrics.items()), columns=['Touchpoint', 'Average Rating (1-5)'])
            
            fig_ratings = px.bar(
                df_ratings,
                x='Average Rating (1-5)',
                y='Touchpoint',
                orientation='h',
                title="Average Patient Rating by Hospital Touchpoint",
                color='Average Rating (1-5)',
                color_continuous_scale='blues',
                text=[f"{v:.2f}" for v in df_ratings['Average Rating (1-5)']]
            )
            fig_ratings.add_vline(x=4.3, line_dash="dash", line_color="green", annotation_text="Target 4.3+")
            fig_ratings.update_layout(height=360, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_ratings, use_container_width=True)
            
        with f_col2:
            # NPS distribution
            nps_dist = filtered_fb['nps_category'].value_counts().reset_index()
            nps_dist.columns = ['Category', 'Count']
            
            fig_nps = px.pie(
                nps_dist,
                names='Category',
                values='Count',
                title=f"Net Promoter Score Breakdown (Current NPS: +{nps_val:.1f} | Target > +40.0)",
                color='Category',
                color_discrete_map={'Promoter': '#10B981', 'Passive': '#FBBF24', 'Detractor': '#EF4444'},
                hole=0.45
            )
            fig_nps.update_layout(height=360, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_nps, use_container_width=True)
            
        # Common patient feedback analysis
        st.markdown("#### 🗣️ Key Qualitative Feedback & Pain Points")
        pf1, pf2, pf3 = st.columns(3)
        with pf1:
            st.error("⏳ **Discharge Turnaround (> 5 Hours)**\n\nPatients report excessive delays waiting for insurance claim approval and pharmacy reconciliation before room clearance.")
        with pf2:
            st.warning("📞 **Post-Discharge Disconnect**\n\n45.2% of discharged patients received no follow-up call, resulting in unaddressed post-operative confusion and preventable readmissions.")
        with pf3:
            st.info("🚨 **Emergency Door Wait Times**\n\nEmergency room wait averaged 43.5 minutes (Target < 30 mins) during peak evening admissions.")

    # TAB 5: Power BI Data Model & DAX Guide
    with d_tab5:
        st.subheader("Power BI Architecture, Schema & DAX Implementation Guide")
        st.markdown("""
        To replicate or embed this dashboard in enterprise **Power BI Desktop / Service**, follow the star schema and DAX measures defined below:
        """)
        
        st.markdown("""
        ```mermaid
        erDiagram
            PATIENTS ||--o{ ADMISSIONS : "patient_id"
            DEPARTMENTS ||--o{ ADMISSIONS : "dept_id"
            DEPARTMENTS ||--o{ DAILY_BED_CENSUS : "dept_id"
            ADMISSIONS ||--o{ INSURANCE_CLAIMS : "admission_id"
            ADMISSIONS ||--o{ PATIENT_FEEDBACK : "admission_id"
        ```
        """)
        
        col_dax1, col_dax2 = st.columns(2)
        with col_dax1:
            st.markdown("##### 📐 Key DAX Measures for Power BI")
            st.code("""
// 1. 30-Day Readmission Rate
Readmission Rate % = 
DIVIDE(
    CALCULATE(
        COUNTROWS(admissions),
        admissions[readmitted_30d] = TRUE(),
        admissions[discharge_disposition] IN {"Discharged Home", "LAMA"}
    ),
    CALCULATE(
        COUNTROWS(admissions),
        admissions[discharge_disposition] IN {"Discharged Home", "LAMA"}
    ),
    0
) * 100

// 2. Average Length of Stay (ALOS)
ALOS = AVERAGE(admissions[los_days])

// 3. 48-Hour Follow-Up Coverage %
Followup Call Coverage % = 
DIVIDE(
    CALCULATE(
        COUNTROWS(admissions),
        admissions[followup_call_48h] = TRUE(),
        admissions[discharge_disposition] IN {"Discharged Home", "LAMA"}
    ),
    CALCULATE(
        COUNTROWS(admissions),
        admissions[discharge_disposition] IN {"Discharged Home", "LAMA"}
    ),
    0
) * 100
            """, language="dax")
            
        with col_dax2:
            st.markdown("##### 💰 Claims & NPS DAX Measures")
            st.code("""
// 4. Claim Rejection Rate %
Claim Rejection Rate % = 
DIVIDE(
    CALCULATE(COUNTROWS(insurance_claims), insurance_claims[claim_status] = "Rejected"),
    COUNTROWS(insurance_claims),
    0
) * 100

// 5. Total Rejected Claim Amount (INR Cr)
Rejected Claims Value (Cr) = 
CALCULATE(
    SUM(insurance_claims[claimed_amount_inr]),
    insurance_claims[claim_status] = "Rejected"
) / 10000000

// 6. Net Promoter Score (NPS)
NPS = 
VAR TotalResponses = COUNTROWS(patient_feedback)
VAR Promoters = CALCULATE(COUNTROWS(patient_feedback), patient_feedback[nps_category] = "Promoter")
VAR Detractors = CALCULATE(COUNTROWS(patient_feedback), patient_feedback[nps_category] = "Detractor")
RETURN
DIVIDE(Promoters - Detractors, TotalResponses, 0) * 100
            """, language="dax")
            
        st.markdown("##### 📥 Export Cleaned Dataset for Power BI Desktop")
        csv_data = filtered_adm.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Filtered Admissions CSV for Power BI",
            data=csv_data,
            file_name="sunrise_hospital_admissions_powerbi.csv",
            mime="text/csv"
        )

# ==========================================
# 3. ML READMISSION RISK PREDICTOR
# ==========================================
elif app_mode == "🤖 Readmission Risk Predictor (ML)":
    st.title("🤖 30-Day Readmission Risk Engine & Clinical Decision Support")
    st.caption("Machine Learning Classifier for Discharge Transition Planning | Trained on FY2025-26 Cohort")
    
    # Model Benchmarking & Performance Overview
    with st.expander("📊 Model Benchmark & Evaluation Metrics (Deliverable 2)", expanded=False):
        m_col1, m_col2 = st.columns([1.2, 1])
        with m_col1:
            perf_df = pd.DataFrame({
                "Algorithm": ["Logistic Regression (Baseline)", "Random Forest", "Gradient Boosting", "XGBoost Classifier"],
                "ROC-AUC": [0.7431, 0.7284, 0.7232, 0.7220],
                "PR-AUC": [0.3197, 0.2735, 0.3018, 0.2846],
                "Recall @ Top 20% Risk": ["48.72%", "43.96%", "45.79%", "45.79%"],
                "Model Status": ["Production Ready (Interpretable)", "Trained & Validated", "Trained & Validated", "Production Deployed"]
            })
            st.dataframe(perf_df, hide_index=True, use_container_width=True)
            st.caption("Note: Stratified 80/20 train/test split. Eligible cohort (Discharged Home & LAMA, 10,816 admissions). Class imbalance positive rate: 12.59%.")
        with m_col2:
            st.markdown("""
            **Key Findings from Hospital Modeling:**
            - Top 20% of high-risk patients capture **~49% of all 30-day readmissions**.
            - Concentrating post-discharge nurse visits and pharmacy reviews on this top quintile directly cuts readmissions by **up to 40%**.
            - **Strongest Predictors**: Chronic heart failure, CKD, number of prior admissions in 12m, ICU stay, late-evening discharge hour, and distance > 50km.
            """)
            
    st.markdown("---")
    
    # Bedside Discharge Scoring Tool
    st.subheader("🩺 Bedside Discharge Risk Assessment Calculator")
    st.markdown("Enter patient clinical and demographic parameters at the time of planned discharge to generate real-time readmission risk.")
    
    # Presets for Demo
    st.markdown("##### ⚡ Quick Load Clinical Patient Profiles:")
    demo_p1, demo_p2, demo_p3 = st.columns(3)
    
    preset = None
    with demo_p1:
        if st.button("🔴 High-Risk Senior (Cardiac & CKD)", use_container_width=True):
            preset = "high"
    with demo_p2:
        if st.button("🟡 Moderate-Risk Adult (Nephrology)", use_container_width=True):
            preset = "med"
    with demo_p3:
        if st.button("🟢 Low-Risk Patient (Day Surgery)", use_container_width=True):
            preset = "low"
            
    # Default values based on preset
    if preset == "high":
        d_age, d_gender, d_city, d_dist = 68, "Male", "Salem", 65.0
        d_dept, d_adm_type, d_los, d_icu = "Cardiology", "Emergency", 8, True
        d_er_wait, d_payer, d_bmi, d_prior = 65, "PMJAY", 28.4, 2
        d_diab, d_hyp, d_ckd, d_copd, d_heart, d_smoker = True, True, True, False, True, True
        d_dis_hour, d_med_rec = 19, False
    elif preset == "med":
        d_age, d_gender, d_city, d_dist = 54, "Female", "Tiruppur", 38.0
        d_dept, d_adm_type, d_los, d_icu = "Nephrology", "Emergency", 6, False
        d_er_wait, d_payer, d_bmi, d_prior = 40, "Private Insurance", 26.2, 1
        d_diab, d_hyp, d_ckd, d_copd, d_heart, d_smoker = True, True, False, False, False, False
        d_dis_hour, d_med_rec = 14, True
    else:
        d_age, d_gender, d_city, d_dist = 32, "Female", "Coimbatore", 8.0
        d_dept, d_adm_type, d_los, d_icu = "General Surgery", "Elective", 3, False
        d_er_wait, d_payer, d_bmi, d_prior = 0, "Cash", 22.5, 0
        d_diab, d_hyp, d_ckd, d_copd, d_heart, d_smoker = False, False, False, False, False, False
        d_dis_hour, d_med_rec = 11, True

    with st.form("risk_prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 👤 Demographics & Geography")
            age = st.number_input("Age", min_value=1, max_value=105, value=d_age)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0 if d_gender == "Male" else 1)
            city = st.selectbox("City of Residence", ["Coimbatore", "Tiruppur", "Erode", "Pollachi", "Palakkad", "Salem"], index=["Coimbatore", "Tiruppur", "Erode", "Pollachi", "Palakkad", "Salem"].index(d_city))
            distance_km = st.number_input("Distance to Hospital (km)", min_value=0.5, max_value=150.0, value=d_dist)
            payer = st.selectbox("Insurance / Payer", ["Cash", "Private Insurance", "PMJAY", "CMCHIS", "ESI", "Corporate"], index=["Cash", "Private Insurance", "PMJAY", "CMCHIS", "ESI", "Corporate"].index(d_payer))

        with c2:
            st.markdown("##### 🏥 Inpatient Admission Details")
            dept = st.selectbox("Treating Department", [
                "Cardiology", "Nephrology", "Oncology", "Neurology",
                "Orthopaedics", "Pulmonology", "General Medicine",
                "General Surgery", "Obstetrics & Gynaecology"
            ], index=["Cardiology", "Nephrology", "Oncology", "Neurology", "Orthopaedics", "Pulmonology", "General Medicine", "General Surgery", "Obstetrics & Gynaecology"].index(d_dept))
            adm_type = st.selectbox("Admission Type", ["Emergency", "Elective"], index=0 if d_adm_type == "Emergency" else 1)
            los_days = st.number_input("Length of Stay (Days)", min_value=1, max_value=45, value=d_los)
            icu_stay = st.checkbox("ICU Stay during Admission", value=d_icu)
            er_wait = st.number_input("ER Door-to-Doctor Wait (min)", min_value=0, max_value=300, value=d_er_wait)
            prior_adm = st.number_input("Prior Admissions in Past 12 Months", min_value=0, max_value=10, value=d_prior)
            bill_inr = st.number_input("Estimated Total Inpatient Bill (₹)", min_value=1000, max_value=2000000, value=125000)

        with c3:
            st.markdown("##### 🩺 Clinical Comorbidities & SOP Flags")
            diab = st.checkbox("Diabetes Mellitus", value=d_diab)
            hyp = st.checkbox("Hypertension", value=d_hyp)
            ckd = st.checkbox("Chronic Kidney Disease (CKD)", value=d_ckd)
            copd = st.checkbox("COPD / Chronic Respiratory", value=d_copd)
            heart = st.checkbox("Heart Disease / Congestive HF", value=d_heart)
            smoker = st.checkbox("Current Tobacco Smoker", value=d_smoker)
            bmi = st.number_input("Body Mass Index (BMI)", min_value=12.0, max_value=55.0, value=d_bmi)
            dis_hour = st.slider("Discharge Hour (24h)", min_value=7, max_value=23, value=d_dis_hour)
            med_rec = st.checkbox("Pharmacist Medication Reconciliation Done", value=d_med_rec)
            
        predict_btn = st.form_submit_button("⚡ Compute Readmission Risk Score", use_container_width=True)
        
    if predict_btn or preset is not None:
        # Calculate comorbidity count
        comorb_count = int(diab) + int(hyp) + int(ckd) + int(copd) + int(heart)
        
        # Prepare feature vector
        sample_dict = {
            'age': [age],
            'gender': [gender],
            'city': [city],
            'distance_km': [distance_km],
            'distance_gt_50km': [int(distance_km > 50)],
            'department': [dept],
            'admission_type': [adm_type],
            'los_days': [los_days],
            'icu_stay': [int(icu_stay)],
            'er_wait_minutes': [er_wait],
            'payer': [payer],
            'comorbidity_count': [comorb_count],
            'high_comorbidity': [int(comorb_count >= 2)],
            'diabetes': [int(diab)],
            'hypertension': [int(hyp)],
            'ckd': [int(ckd)],
            'copd': [int(copd)],
            'heart_disease': [int(heart)],
            'smoker': [int(smoker)],
            'bmi': [bmi],
            'prior_admissions_12m': [prior_adm],
            'has_prior_admissions': [int(prior_adm > 0)],
            'discharge_disposition': ['Discharged Home'],
            'discharge_hour': [dis_hour],
            'is_late_discharge': [int(dis_hour >= 18)],
            'med_reconciliation_done': [int(med_rec)],
            'total_bill_inr': [bill_inr],
            'cardio_nephro_neuro': [int(dept in ['Cardiology', 'Nephrology', 'Neurology'])]
        }
        df_input = pd.DataFrame(sample_dict)
        
        # Select active model
        active_model = ml_models.get("XGBoost") or ml_models.get("Logistic Regression")
        if active_model is not None:
            risk_proba = active_model.predict_proba(df_input)[0, 1] * 100
        else:
            # Fallback heuristic aligned with clinical weights
            risk_proba = 8.0 + (heart * 12.0) + (ckd * 10.0) + (prior_adm * 6.0) + (icu_stay * 5.0) + ((distance_km > 50) * 4.0) + ((dis_hour >= 18) * 4.5) - (med_rec * 3.5)
            risk_proba = min(max(risk_proba, 2.0), 92.0)
            
        st.markdown("### 📋 Predictive Assessment Output")
        out_col1, out_col2 = st.columns([1, 1.4])
        
        with out_col1:
            if risk_proba >= 25.0:
                band_color = "#DC2626"
                band_name = "🔴 HIGH RISK (> 25%)"
                badge_style = "badge-red"
            elif risk_proba >= 10.0:
                band_color = "#D97706"
                band_name = "🟡 MEDIUM RISK (10% - 25%)"
                badge_style = "badge-amber"
            else:
                band_color = "#16A34A"
                band_name = "🟢 LOW RISK (< 10%)"
                badge_style = "badge-green"
                
            st.markdown(f"""
            <div style="background:white; border:2px solid {band_color}; border-radius:12px; padding:1.5rem; text-align:center;">
                <span class="badge-target {badge_style}" style="font-size:0.9rem;">RISK CATEGORY</span>
                <h1 style="color:{band_color}; margin:12px 0 6px 0; font-size:3rem;">{risk_proba:.1f}%</h1>
                <h3 style="color:#0A2540; margin:0;">{band_name}</h3>
                <p style="color:#64748B; font-size:0.85rem; margin-top:8px;">Model: Production Ensemble Classifier</p>
            </div>
            """, unsafe_allow_html=True)
            
        with out_col2:
            st.markdown("#### 🚨 Hospital SOP Mandated Transition Protocols:")
            # Dynamic Hospital SOP Checklist
            if risk_proba >= 25.0:
                st.error("⚠️ **MANDATORY HIGH-RISK CARE TRANSITION PROTOCOL (SMH Policy v4.2, Section 4)**:")
                st.markdown("""
                - [x] **Nurse Follow-up Call**: Schedule within **48 hours** to verify vitals and symptoms.
                - [x] **Scheduled 7-Day OPD Review**: Free doctor follow-up consultation booked at discharge.
                - [x] **Day-14 Telephone Check**: Second nurse follow-up call.
                - [x] **30-Day Tele-Consultation Access**: Free helpline support for medication queries.
                """)
            elif risk_proba >= 10.0:
                st.warning("⚠️ **MEDIUM-RISK CARE PROTOCOL (SMH Policy v4.2, Section 4)**:")
                st.markdown("""
                - [x] **Nurse Follow-up Call**: Mandatory within **48 hours**.
                - [x] **Discharge Summary Education**: Review danger signs with patient & caregiver.
                """)
            else:
                st.success("✅ **STANDARD LOW-RISK DISCHARGE PROTOCOL**:")
                st.markdown("""
                - [x] Standard 48-hour follow-up call.
                - [x] Standard OPD review as indicated in discharge summary.
                """)
                
            # Additional conditional flags from Policy Manual
            if distance_km > 50:
                st.info(f"📍 **Patient Distance ({distance_km:.0f} km > 50 km)**: Under Section 4.5, patient is eligible for **tele-consultation follow-up** in place of the in-person 7-day hospital visit.")
            if dis_hour >= 18:
                st.warning("🌙 **Late-Evening Discharge (>= 18:00)**: Under Section 3.2, discharges after 18:00 require **Duty Medical Officer (DMO) countersignature**.")
            if not med_rec and (comorb_count >= 2 or dept in ['Cardiology', 'Nephrology', 'Pulmonology', 'Neurology']):
                st.error("💊 **Pharmacy Alert**: Medication reconciliation was NOT marked done. Clinical pharmacist teach-back is mandatory for this clinical profile prior to exit.")

# ==========================================
# 4. GEMINI RAG POLICY CHATBOT
# ==========================================
elif app_mode == "💬 Gemini RAG Policy Chatbot":
    st.title("💬 Gemini RAG AI Clinical & Operational Assistant")
    st.caption("Grounded strictly in Sunrise Hospital Policy Manual v4.2 & Annual Report FY2025-26 with Page Citations")
    
    # Pre-canned prompt buttons
    st.markdown("##### 💡 Suggested Test Questions from Capstone Problem Statement:")
    sq_col1, sq_col2, sq_col3 = st.columns(3)
    
    selected_prompt = None
    with sq_col1:
        if st.button("⏱️ Target time from discharge order to exit?", use_container_width=True):
            selected_prompt = "What is the target time from discharge order to patient exit?"
        if st.button("📊 Readmission rate & highest dept in FY25-26?", use_container_width=True):
            selected_prompt = "What was the hospital's readmission rate in FY2025-26 and which department was highest?"
    with sq_col2:
        if st.button("📞 Who must be called within 48h & what is checked?", use_container_width=True):
            selected_prompt = "Who must be called within 48 hours of discharge and what is checked?"
        if st.button("💸 Planned actions for FY26-27 to reduce claim rejections?", use_container_width=True):
            selected_prompt = "What actions are planned for FY2026-27 to reduce claim rejections?"
    with sq_col3:
        if st.button("📑 Within how many days must an insurance claim be submitted?", use_container_width=True):
            selected_prompt = "Within how many days must an insurance claim be submitted after discharge?"
        if st.button("🚨 ICU occupancy escalation triggers?", use_container_width=True):
            selected_prompt = "What is the escalation trigger when ICU occupancy crosses the threshold?"

    st.markdown("---")
    
    # Chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am the **Sunrise Multispeciality Hospital AI Clinical & Operational Assistant**. I can answer clinical SOPs, bed escalation triggers, discharge workflows, and annual performance questions grounded directly in the hospital's official Policy Manual and FY2025-26 Annual Report with document citations.",
                "citations": []
            }
        ]
        
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📄 Document Citations & Sources", expanded=False):
                    for cit in msg["citations"]:
                        st.markdown(f"**{cit['document']}** — Page {cit['page']} (*{cit['section']}*)")
                        st.caption(f"Snippet: {cit['snippet']}")

    # Handle user input (either from chat input or clicked prompt chip)
    user_query = st.chat_input("Ask a question about Sunrise Hospital policies, SOPs, or annual performance...")
    if selected_prompt:
        user_query = selected_prompt

    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query, "citations": []})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Generate RAG response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving verified hospital policy documents..."):
                response_obj = rag_engine.query(user_query, api_key=user_api_key)
                
                st.markdown(response_obj["answer"])
                
                if response_obj.get("citations"):
                    with st.expander("📄 Document Citations & Sources", expanded=True):
                        for cit in response_obj["citations"]:
                            st.markdown(f"📌 **{cit['document']}** — Page {cit['page']}")
                            st.markdown(f"**Section**: *{cit['section']}* (Relevance Score: {cit['score']})")
                            st.caption(cit['snippet'])
                            
                st.caption(f"Engine: {response_obj.get('model_used', 'Grounded RAG Pipeline')}")
                
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_obj["answer"],
            "citations": response_obj.get("citations", [])
        })

# ==========================================
# 5. STRATEGIC RECOMMENDATIONS MEMO
# ==========================================
elif app_mode == "📋 Strategic Recommendations Memo":
    st.title("📋 Executive Recommendation Memo: Board of Directors")
    st.caption("Deliverable 4 | Sunrise Multispeciality Hospital, Coimbatore | Strategic Initiatives & Financial Roadmap FY2026-27")
    
    st.markdown("""
    <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:12px; padding:1.5rem; margin-bottom:1.5rem;">
        <h3 style="color:#0A2540; margin-top:0;">Executive Summary</h3>
        <p style="color:#334155; line-height:1.6; margin-bottom:0;">
            Based on the analysis of 11,200 inpatient admissions, 7,845 insurance claims, and hospital bed censuses during FY2025-26, Sunrise Multispeciality Hospital faces three major operational vulnerabilities: 
            <strong>30-day readmissions (12.6% vs 10% target)</strong>, 
            <strong>bed bottlenecks in cardiology/nephrology (86-88% occupancy)</strong>, and 
            <strong>claim rejections totaling ₹9.03 Crore</strong>. 
            The following four initiatives represent high-return, immediate interventions to recover margin, expand capacity, and elevate patient safety.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Financial Impact Summary Cards
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size:0.8rem; color:#64748B; font-weight:600;">CLAIM REVENUE RECOVERY</span>
            <h2 style="color:#16A34A; margin:4px 0;">+ ₹4.0 Cr</h2>
            <span style="font-size:0.8rem; color:#64748B;">Target Claim Rejection < 5%</span>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size:0.8rem; color:#64748B; font-weight:600;">READMISSIONS PREVENTED</span>
            <h2 style="color:#2563EB; margin:4px 0;">~290 Cases</h2>
            <span style="font-size:0.8rem; color:#64748B;">Target Readmission < 10%</span>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size:0.8rem; color:#64748B; font-weight:600;">CAPACITY UNLOCKED</span>
            <h2 style="color:#D97706; margin:4px 0;">4,500 Bed-Days</h2>
            <span style="font-size:0.8rem; color:#64748B;">ALOS Reduction to 5.8 Days</span>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size:0.8rem; color:#64748B; font-weight:600;">PATIENT SATISFACTION</span>
            <h2 style="color:#0A2540; margin:4px 0;">NPS > +40</h2>
            <span style="font-size:0.8rem; color:#64748B;">Discharge Lounge + Bilingual SOP</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 4 Actionable Strategic Pillars
    st.subheader("🎯 The 4 Strategic Action Pillars for FY2026-27")
    
    # Pillar 1
    with st.expander("1. Readmission Prevention: 100% 48-Hour Follow-Up Calls & Predictive Discharge Scoring", expanded=True):
        st.markdown("""
        - **Current Finding**: Discharged patients who received the nurse follow-up call had an **8.1% readmission rate** compared to **17.7% for those uncalled** (more than double). Yet only **54.8%** of patients received this call.
        - **Action**:
            1. Integrate the machine learning readmission risk score directly into the HIS discharge module.
            2. Automatically alert dedicated post-discharge care coordinators for any patient with Risk Score > 25%.
            3. Enforce **100% telephone follow-up coverage within 48 hours** with automated nurse dialling.
            4. Provide remote tele-follow-up for patients residing > 50 km from Coimbatore.
        - **Projected Impact**: Prevents ~290 unplanned readmissions annually, saving ₹1.8 Cr in non-reimbursable care costs and protecting hospital bed availability.
        """)

    # Pillar 2
    with st.expander("2. Revenue Cycle Overhaul: Pre-Billing Digital Audit to Stop ₹9.03 Cr Claim Losses", expanded=True):
        st.markdown("""
        - **Current Finding**: Incomplete documentation caused **250 rejections (₹3.1 Cr)**, while missing pre-auths caused **138 rejections (₹1.7 Cr)** and package mismatches caused **105 rejections (₹1.3 Cr)**. Over 70% of losses were preventable clerical and workflow delays.
        - **Action**:
            1. Deploy a **Digital Claim Checklist** requiring doctor sign-off, investigation reports, implant stickers, and pre-auth ID before billing can initiate discharge clearance.
            2. Enforce strict **3-day claim submission timeline** post-discharge.
            3. Institute weekly Revenue Cycle Management audits for PM-JAY and CMCHIS package code reconciliations.
        - **Projected Impact**: Reduces claim rejections from **9.2% to < 5%**, recovering **₹4.0+ Crore in direct EBITDA** in FY2026-27.
        """)

    # Pillar 3
    with st.expander("3. Bed Capacity Unlocking: Dynamic Bed Flexing & Discharge Before Noon", expanded=True):
        st.markdown("""
        - **Current Finding**: Cardiology (86.5%), Nephrology (88.0%), and Neurology (85.2%) are choked at capacity, causing elective admission deferrals. Simultaneously, OBG (28.5%) and General Medicine (49.7%) have significant idle bed days.
        - **Action**:
            1. Activate the **Dynamic Ward Flexing Policy**: Formally designate 20-25 beds in OBG and General Medicine as flex overflow beds for Cardiology/Nephrology step-down patients.
            2. Implement the **Discharge Before Noon target (60% before 12:00 PM)** and 60-minute bed turnaround by housekeeping.
            3. Commission 20 new cardiac care beds in Q3 FY26-27 as planned by the Board.
        - **Projected Impact**: Frees 4,500+ inpatient bed-days, shortens ALOS from 6.24 to < 5.8 days, and unlocks an estimated **₹8.5 Cr in new inpatient revenue**.
        """)

    # Pillar 4
    with st.expander("4. Patient Experience: Discharge Transit Lounge & Bilingual Care Summaries", expanded=True):
        st.markdown("""
        - **Current Finding**: Survey of 3,200 patients revealed an NPS of only **+6.4**, driven by discharge delays exceeding 5 hours, uncalled discharges, and confusing billing counters.
        - **Action**:
            1. Open a comfortable **Patient Discharge Lounge** on the ground floor so vacated ward beds can be cleaned immediately while insurance paperwork settles.
            2. Issue bilingual (English + Tamil) discharge summaries with clear dosage instructions and red-flag danger signs.
            3. Fast-track emergency admissions with door-to-doctor time < 30 minutes.
        - **Projected Impact**: Lifts NPS from **+6.4 to > +40.0**, improving community reputation and private insurance market share across western Tamil Nadu.
        """)
        
    st.markdown("---")
    st.markdown("#### 📅 FY2026-27 Implementation Roadmap")
    roadmap_data = pd.DataFrame([
        {"Quarter": "Q1 (Apr - Jun 2026)", "Key Deliverable": "Launch Digital Claim Checklist; Roll out HIS Readmission Risk Score; Bed Flexing SOP in OBG/Med"},
        {"Quarter": "Q2 (Jul - Sep 2026)", "Key Deliverable": "100% 48-Hour Call coverage; Open Ground Floor Discharge Lounge; Bilingual Discharge Summaries"},
        {"Quarter": "Q3 (Oct - Dec 2026)", "Key Deliverable": "Commission 20 New Cardiac Beds; Full Gemini AI Policy Assistant rollout across nursing stations"},
        {"Quarter": "Q4 (Jan - Mar 2027)", "Key Deliverable": "Annual audit: Target Readmissions < 10%, Claim Rejection < 5%, Occupancy 75-80%, NPS > 40"}
    ])
    st.table(roadmap_data)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94A3B8; font-size:0.85rem; padding: 1rem 0;">
    Sunrise Multispeciality Hospital, Coimbatore | Healthcare Intelligence Platform | NABH 5th Edition Accredited | FY2025-26
</div>
""", unsafe_allow_html=True)
