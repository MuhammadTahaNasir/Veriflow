import os
import sys
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src folder to system path for clean imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from database import load_integrated_audit_data
    from preprocess import engineer_audit_features
except ImportError:
    sys.path.append("src")
    from database import load_integrated_audit_data
    from preprocess import engineer_audit_features

# ----------------------------------------------------
# MAIN CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Veriflow",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# HIGH-CONTRAST "FLUX" SIDEBAR & MAIN COMPLIANCE CSS
# ----------------------------------------------------
flux_theme_css = """
<style>
/* Load Professional Fonts and Bootstrap Vector Icons */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
@import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css');

/* Main Canvas Body Overrides - Forces High Contrast Deep Slate Text */
html, body {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #0f172a !important; /* Deep Slate Black for Maximum Contrast */
}

/* Ensure all text inside main canvas cards is highly visible */
.flux-card, .dashboard-title, .dashboard-subtitle, .form-section-title {
    color: #0f172a !important;
}

.stMarkdown, .stText, p, span, li, label {
    color: #0f172a;
}

/* Premium Light Grey Canvas on the right side */
.stApp {
    background-color: #f1f5f9; /* Slightly darker slate grey for premium contrast */
    background-attachment: fixed;
}

/* Style the Streamlit Sidebar to match the deep flat obsidian sidebar in flux */
[data-testid="stSidebar"] {
    background-color: #0f172a !important; /* Rich Dark Charcoal */
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Force all sidebar text to be bright slate white for high contrast */
[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

/* Brand logo styling in sidebar */
.brand-container {
    padding: 20px 10px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 25px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.brand-icon {
    font-size: 26px;
    color: #c7f70b;
    text-shadow: 0 0 12px rgba(199, 247, 11, 0.4);
}

.brand-name {
    font-family: 'Outfit', sans-serif;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff !important;
}

/* Custom Modern SaaS Sidebar Navigation Card List (no circles) */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    display: none !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    background: transparent !important;
    gap: 10px !important;
    padding: 0 10px !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 12px 18px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-bottom: 4px !important;
    display: flex !important;
    align-items: center !important;
    cursor: pointer !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}

/* Hides standard Streamlit radio circles and indicators in the sidebar navigation */
[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stVisualTests"],
[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stVisualTests"],
[data-testid="stSidebar"] div[role="radiogroup"] label div[dir="ltr"],
[data-testid="stSidebar"] div[role="radiogroup"] label [class*="StyledRadio"],
[data-testid="stSidebar"] div[role="radiogroup"] label [class*="RadioIndicator"],
[data-testid="stSidebar"] div[role="radiogroup"] label span,
[data-testid="stSidebar"] div[role="radiogroup"] label svg {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #94a3b8 !important; /* Muted Slate */
    font-size: 15px !important;
    font-weight: 600 !important;
    margin: 0 !important;
    letter-spacing: 0.010em !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover p {
    color: #ffffff !important;
}

/* Selected active tab highlight - SaaS Glow Card */
[data-testid="stSidebar"] div[role="radiogroup"] [data-checked="true"] {
    background-color: rgba(199, 247, 11, 0.15) !important;
    border: 1px solid rgba(199, 247, 11, 0.4) !important;
    box-shadow: 0 0 15px rgba(199, 247, 11, 0.1) !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] [data-checked="true"] label p {
    color: #c7f70b !important; /* Glowing Neon Green */
    font-weight: 700 !important;
}

/* Main Dashboard Header */
.dashboard-header {
    margin-bottom: 25px;
    margin-top: 15px;
}

.dashboard-title {
    font-family: 'Outfit', sans-serif;
    font-size: 38px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}

.dashboard-subtitle {
    font-size: 15px;
    color: #475569;
}

/* Crisp Premium White Cards */
.flux-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.03);
    margin-bottom: 20px;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.flux-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}

.flux-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}

.flux-card-icon-container {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background-color: #f1f5f9;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #0f172a;
    font-size: 18px;
}

.flux-card-title {
    font-size: 13px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-left: 12px;
}

.flux-card-value {
    font-family: 'Outfit', sans-serif;
    font-size: 34px;
    font-weight: 700;
    color: #0f172a;
}

.flux-card-change {
    background-color: rgba(199, 247, 11, 0.25);
    color: #0f172a;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

.flux-card-footer {
    font-size: 13px;
    color: #475569;
    margin-top: 10px;
    font-weight: 500;
}

/* Contrast Obsidian Card (Bottom Right widget) */
.obsidian-card {
    background-color: #0f172a !important; /* Flat Obsidian Charcoal */
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}

.obsidian-card p, .obsidian-card span, .obsidian-card b, .obsidian-card div {
    color: #f8fafc !important; /* Enforces bright white inside dark card */
}

/* Elegant High-Contrast Vector-Glow AP Badges */
.ap-badge {
    padding: 20px 24px;
    border-radius: 16px;
    margin-bottom: 24px;
    border-left: 6px solid;
}

.ap-badge-approved {
    background-color: #ecfdf5;
    border-color: #059669;
    color: #065f46;
}

.ap-badge-approved * {
    color: #065f46 !important;
}

.ap-badge-audit {
    background-color: #fff5f5;
    border-color: #dc2626;
    color: #7f1d1d;
}

.ap-badge-audit * {
    color: #7f1d1d !important;
}

.ap-badge-title {
    font-family: 'Outfit', sans-serif;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Premium Structured B2B Table - High Visibility */
.flux-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 14px;
    font-size: 14px;
}

.flux-table th {
    color: #475569 !important;
    text-align: left;
    padding: 14px 16px;
    font-weight: 700;
    border-bottom: 2px solid #cbd5e1;
}

.flux-table td {
    padding: 14px 16px;
    border-bottom: 1px solid #e2e8f0;
    color: #334155 !important;
    font-weight: 550;
}

.flux-dark-table th {
    color: #94a3b8 !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.1) !important;
}

.flux-dark-table td {
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    color: #cbd5e1 !important;
}

/* High Contrast Input Select Boxes and Input Fields */
div[data-baseweb="select"] * {
    color: #0f172a !important; /* Enforce readable black text inside dropdowns */
}

input {
    color: #0f172a !important; /* Enforce readable black text in inputs */
    font-weight: 600 !important;
}

/* Form Section Title */
.form-section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 12px;
}

/* Tab Overrides */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #cbd5e1;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #475569;
}

.stTabs [aria-selected="true"] {
    color: #0f172a !important;
    border-bottom: 2.5px solid #0f172a !important;
}

/* Fix Streamlit standard metrics display padding */
[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}
</style>
"""
st.markdown(flux_theme_css, unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD MODELS & CACHED DATA
# ----------------------------------------------------
@st.cache_resource
def load_ml_pipelines():
    reg_path = "models/price_regressor.pkl"
    clf_path = "models/audit_classifier.pkl"
    pipelines = {}
    if os.path.exists(reg_path):
        with open(reg_path, 'rb') as f:
            pipelines['regressor'] = pickle.load(f)
    if os.path.exists(clf_path):
        with open(clf_path, 'rb') as f:
            pipelines['classifier'] = pickle.load(f)
    return pipelines

@st.cache_data
def load_transaction_data():
    cache_path = "data/audited_transactions_cache.csv"
    db_path = "data/inventory.db"
    
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        for col in ['PODate', 'ReceivingDate', 'InvoiceDate', 'PayDate']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df
    elif os.path.exists(db_path):
        df_raw = load_integrated_audit_data(db_path)
        df_engineered = engineer_audit_features(df_raw)
        return df_engineered
    else:
        st.error("No database cache found. Please run 'python src/train.py' first.")
        st.stop()

pipelines = load_ml_pipelines()
df_audited = load_transaction_data()

# Define historical_vendors globally so it is available to ALL pages!
historical_vendors = sorted(df_audited["VendorName"].unique())

# ----------------------------------------------------
# SIDEBAR NAVIGATION (flux menu style)
# ----------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand-container">
        <div class="brand-icon"><i class="bi bi-shield-check"></i></div>
        <div class="brand-name">Veriflow</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Elegant custom styled modern sidebar radio selector
    page = st.sidebar.radio(
        "Navigation",
        options=["Overview Dashboard", "Invoice Simulator", "Database Audit Ledger"],
        index=0
    )

# ----------------------------------------------------
# PAGE 1: OVERVIEW DASHBOARD
# ----------------------------------------------------
if page == "Overview Dashboard":
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">Compliance Dashboard</div>
        <div class="dashboard-subtitle">Audited financial volume, transaction safety rates, and B2B pricing leakages.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate metrics
    total_billed = df_audited['InvoicedDollars'].sum()
    price_leakage = df_audited[df_audited['price_variance'] > 0]['price_variance'].sum()
    freight_leakage = df_audited[df_audited['freight_to_dollar_ratio'] > 0.008]['InvoicedFreight'].sum()
    total_leakage = price_leakage + (freight_leakage * 0.4)
    
    total_anomalies = df_audited['AuditRequired'].sum()
    anomaly_rate = (total_anomalies / len(df_audited)) * 100
    auto_approve_rate = 100 - anomaly_rate
    
    # KPI cards layout mirroring the flux screenshot exactly
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.markdown(f"""
        <div class="flux-card">
            <div class="flux-card-header">
                <div style="display: flex; align-items: center;">
                    <div class="flux-card-icon-container"><i class="bi bi-wallet2"></i></div>
                    <div class="flux-card-title">Audited Volume</div>
                </div>
            </div>
            <div class="flux-card-value">${total_billed/1e6:.2f}M</div>
            <div class="flux-card-footer">📄 {len(df_audited):,} invoices audited</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
        <div class="flux-card">
            <div class="flux-card-header">
                <div style="display: flex; align-items: center;">
                    <div class="flux-card-icon-container"><i class="bi bi-patch-check"></i></div>
                    <div class="flux-card-title">Compliance Rate</div>
                </div>
                <div class="flux-card-change">+1.2%</div>
            </div>
            <div class="flux-card-value">{auto_approve_rate:.1f}%</div>
            <div class="flux-card-footer">✔ {len(df_audited) - total_anomalies:,} invoices safe</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        st.markdown(f"""
        <div class="flux-card">
            <div class="flux-card-header">
                <div style="display: flex; align-items: center;">
                    <div class="flux-card-icon-container" style="background-color:#fee2e2; color:#ef4444;"><i class="bi bi-exclamation-triangle"></i></div>
                    <div class="flux-card-title">Billing Leakage</div>
                </div>
            </div>
            <div class="flux-card-value">${total_leakage/1e3:.1f}k</div>
            <div class="flux-card-footer">✗ {total_anomalies:,} anomalies flagged</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row with high visibility
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("##### Cumulative Billing Mismatches by Vendor")
        vendor_leakage = df_audited[df_audited['price_variance'] > 0].groupby("VendorName")["price_variance"].sum().reset_index()
        vendor_leakage = vendor_leakage.sort_values(by="price_variance", ascending=False).head(5)
        
        if len(vendor_leakage) > 0:
            fig_leakage = px.bar(
                vendor_leakage,
                x="price_variance",
                y="VendorName",
                orientation="h",
                color_discrete_sequence=["#b4b6f9"], # elegant lavender pastel
                labels={"price_variance": "Billing Mismatch ($)", "VendorName": "Vendor"},
                template="plotly_white"
            )
        else:
            # Fallback when real database price_variance has exactly 0 mismatches
            # Display real vendors with their audited freight variances instead
            freight_by_vendor = df_audited.groupby("VendorName")["InvoicedFreight"].sum().reset_index().sort_values(by="InvoicedFreight", ascending=False).head(5)
            fig_leakage = px.bar(
                freight_by_vendor,
                x="InvoicedFreight",
                y="VendorName",
                orientation="h",
                color_discrete_sequence=["#b4b6f9"],
                labels={"InvoicedFreight": "Audited Freight Volume ($)", "VendorName": "Vendor"},
                template="plotly_white"
            )
            
        fig_leakage.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(
                gridcolor="#cbd5e1",
                title_font=dict(color="#0f172a", size=12, family="'Plus Jakarta Sans', sans-serif"),
                tickfont=dict(color="#0f172a", size=10, family="'Plus Jakarta Sans', sans-serif")
            ),
            yaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                title_font=dict(color="#0f172a", size=12, family="'Plus Jakarta Sans', sans-serif"),
                tickfont=dict(color="#0f172a", size=10, family="'Plus Jakarta Sans', sans-serif")
            )
        )
        st.plotly_chart(fig_leakage, use_container_width=True)
        
    with chart_col2:
        st.markdown("##### Billed Freight Fees vs. Invoice Dollars")
        fig_freight = px.scatter(
            df_audited,
            x="InvoicedDollars",
            y="InvoicedFreight",
            color="AuditRequired",
            color_discrete_map={0: "#b4b6f9", 1: "#ef4444"},
            labels={"InvoicedDollars": "Invoice Dollars", "InvoicedFreight": "Freight Charge ($)", "AuditRequired": "Audit required"},
            template="plotly_white",
            opacity=0.6
        )
        fig_freight.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(
                gridcolor="#cbd5e1",
                title_font=dict(color="#0f172a", size=12, family="'Plus Jakarta Sans', sans-serif"),
                tickfont=dict(color="#0f172a", size=10, family="'Plus Jakarta Sans', sans-serif")
            ),
            yaxis=dict(
                gridcolor="#cbd5e1",
                title_font=dict(color="#0f172a", size=12, family="'Plus Jakarta Sans', sans-serif"),
                tickfont=dict(color="#0f172a", size=10, family="'Plus Jakarta Sans', sans-serif")
            )
        )
        st.plotly_chart(fig_freight, use_container_width=True)

# ====================================================
# PAGE 2: INVOICE SIMULATOR
# ====================================================
elif page == "Invoice Simulator":
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">Compliance Simulator</div>
        <div class="dashboard-subtitle">Simulate single invoice parameters to predict shipping and pricing contract compliance.</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_inputs, col_results = st.columns([4, 6])
    
    with col_inputs:
        st.markdown("<div class='form-section-title'><i class='bi bi-input-cursor-text'></i> Single Invoice Parameters</div>", unsafe_allow_html=True)
        
        # Bordered Container inside Crisp White Card - Clean, premium native inputs
        with st.container(border=True):
            vendor_select = st.selectbox("Select B2B Vendor Name", historical_vendors)
            qty_input = st.number_input("Invoiced Item Quantity", min_value=1, value=1200, step=100)
            dollars_input = st.number_input("Invoiced Total Amount ($)", min_value=1.0, value=8500.0, step=100.0)
            po_dollars_input = st.number_input("Agreed PO Amount ($)", min_value=1.0, value=8500.0, step=100.0)
            brands_input = st.number_input("Unique Brands Billed", min_value=1, value=4, step=1)
            days_input = st.number_input("Days PO to Invoice Date", min_value=0, value=5, step=1)
            freight_input = st.number_input("Billed Freight Cost ($)", min_value=0.0, value=120.0, step=10.0)
            
        run_audit = st.button("Run Compliance Audit", type="primary")
        
    with col_results:
        st.markdown("<div class='form-section-title'><i class='bi bi-receipt-cutoff'></i> Compliance Audit Report</div>", unsafe_allow_html=True)
        
        if run_audit:
            # Structure inputs
            input_df = pd.DataFrame([{
                'InvoicedQuantity': qty_input,
                'InvoicedDollars': dollars_input,
                'UniqueBrandsOrdered': brands_input,
                'days_to_invoice': days_input,
                'VendorName': vendor_select
            }])
            
            # Predict using pipeline
            if 'regressor' not in pipelines or 'classifier' not in pipelines:
                st.error("🚨 Auditing Models are missing! Please run 'python src/train.py' first to compile and save your ML pipelines.")
                st.stop()
                
            fair_freight = float(pipelines['regressor'].predict(input_df)[0])
            risk_prediction = int(pipelines['classifier'].predict(input_df)[0])
            risk_probability = float(pipelines['classifier'].predict_proba(input_df)[0][1])
            
            # Calculations
            price_variance = dollars_input - po_dollars_input
            freight_overcharge = max(0.0, freight_input - fair_freight)
            total_invoiced = dollars_input + freight_input
            total_expected = po_dollars_input + fair_freight
            total_variance = total_invoiced - total_expected
            
            # Renders glowing custom vector alert badges (high visibility)
            if risk_prediction == 1 or price_variance > 5.0 or freight_input > fair_freight * 1.5:
                st.markdown(f"""
                <div class="ap-badge ap-badge-audit">
                    <div class="ap-badge-title">
                        <i class="bi bi-shield-fill-x"></i> Audit Recommended
                    </div>
                    <div style="font-size: 14px; font-weight: 550; opacity: 0.9;">Machine Learning has flagged extreme pricing/freight discrepancies. Manual review is recommended. (Anomaly Risk Score: {risk_probability*100:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ap-badge ap-badge-approved">
                    <div class="ap-badge-title">
                        <i class="bi bi-shield-fill-check"></i> Auto-Approved
                    </div>
                    <div style="font-size: 14px; font-weight: 550; opacity: 0.9;">Transaction billing matches contracting baselines. Auto-clearance recommended. (Anomaly Risk Score: {risk_probability*100:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
                
            # White Comparison Table - Packaged in single HTML div to prevent code block parsing
            price_var_str = f"${price_variance:,.2f}" if price_variance <= 0 else f"<span style='color:#dc2626; font-weight:700;'>+${price_variance:,.2f}</span>"
            f_over_str = "$0.00" if freight_overcharge <= 0 else f"<span style='color:#dc2626; font-weight:700;'>+${freight_overcharge:,.2f}</span>"
            t_var_str = f"${total_variance:,.2f}" if total_variance <= 0 else f"<span style='color:#dc2626; font-weight:750;'>+${total_variance:,.2f}</span>"
            
            variance_card_html = f"""<div class="flux-card">
<b style="color: #0f172a !important; display: block; font-size: 16px; margin-bottom: 12px;">B2B Cost Variance Ledger</b>
<table class="flux-table">
<thead>
<tr>
<th style="color: #475569 !important;">B2B Segment</th>
<th style="color: #475569 !important;">Billed Invoice ($)</th>
<th style="color: #475569 !important;">Agreed PO ($)</th>
<th style="color: #475569 !important;">Variance ($)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Core Item price</td>
<td>${dollars_input:,.2f}</td>
<td>${po_dollars_input:,.2f}</td>
<td>{price_var_str}</td>
</tr>
<tr>
<td>Freight & Shipping</td>
<td>${freight_input:,.2f}</td>
<td>${fair_freight:,.2f}</td>
<td>{f_over_str}</td>
</tr>
<tr style="border-top: 2px solid #cbd5e1; font-weight: 700; font-size: 15px;">
<td>Total Fee</td>
<td>${total_invoiced:,.2f}</td>
<td>${total_expected:,.2f}</td>
<td>{t_var_str}</td>
</tr>
</tbody>
</table>
</div>"""
            st.markdown(variance_card_html, unsafe_allow_html=True)
            
            # Obsidian contrast cards (Bottom XAI risk dashboard) - Unified in single unindented HTML block to prevent parsing issues
            risk_factors = []
            if price_variance > 0:
                risk_factors.append(("Billing Overcharge", f"+${price_variance:,.2f}", "✗ ⚠️ Increases Risk"))
            else:
                risk_factors.append(("Billing Congruency", "Matched", "✔ Safe"))
                
            if freight_input > fair_freight:
                diff = freight_input - fair_freight
                risk_factors.append(("Inflated Freight", f"+${diff:,.2f}", "✗ ⚠️ Increases Risk"))
            else:
                risk_factors.append(("Freight Rate within baseline", "Normal", "✔ Safe"))
                
            if days_input > 10:
                risk_factors.append(("Billing Delay", f"{days_input} days", "✗ ⚠️ Increases Risk"))
            else:
                risk_factors.append(("Billing Speed", "Fast", "✔ Safe"))
                
            xai_rows = ""
            for factor, val, impact in risk_factors:
                color_style = "color:#f43f5e;" if "Increases Risk" in impact else "color:#10b981;"
                xai_rows += f"""<tr>
<td style="color: #cbd5e1 !important;">{factor}</td>
<td style="color: #cbd5e1 !important;">{val}</td>
<td style="{color_style}">{impact}</td>
</tr>"""
                
            xai_card_html = f"""<div class="obsidian-card">
<b style="color: #ffffff !important; display: block; font-size: 16px; margin-bottom: 6px;"><i class="bi bi-activity"></i> Risk Factors Dashboard</b>
<p style="color: #cbd5e1 !important; font-size: 14px; margin-bottom: 14px;">How individual invoice signals contribute to the Audit decision:</p>
<table class="flux-table flux-dark-table">
<thead>
<tr>
<th style="color: #94a3b8 !important;">Invoice Signal</th>
<th style="color: #94a3b8 !important;">Metric Value</th>
<th style="color: #94a3b8 !important;">Audit Impact</th>
</tr>
</thead>
<tbody>
{xai_rows}
</tbody>
</table>
</div>"""
            st.markdown(xai_card_html, unsafe_allow_html=True)
        else:
            # High-visibility placeholder card
            st.markdown("""
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 50px; text-align: center; color: #475569;">
                <i class="bi bi-shield-lock" style="font-size: 54px; color: #b4b6f9; display: block; margin-bottom: 16px;"></i>
                <div style="font-size: 16px; font-weight: 600; color: #0f172a; margin-bottom: 4px;">Auditor Standby</div>
                Enter transaction parameters on the left and click "Run Compliance Audit" to run predictions.
            </div>
            """, unsafe_allow_html=True)

# ====================================================
# PAGE 3: Database Audit Ledger
# ====================================================
elif page == "Database Audit Ledger":
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">Audit Ledger</div>
        <div class="dashboard-subtitle">Cross-reference and verify every invoice inside the database. Filter high-risk variances dynamically.</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_filters1, col_filters2 = st.columns(2)
    with col_filters1:
        vendor_filter = st.multiselect("Filter by Vendor Name", historical_vendors)
    with col_filters2:
        risk_filter = st.selectbox("Filter by Audit Status", ["All Records", "Flagged for Audit ⚠️", "Auto-Approved ✅"])
        
    df_filtered = df_audited.copy()
    if vendor_filter:
        df_filtered = df_filtered[df_filtered["VendorName"].isin(vendor_filter)]
        
    if risk_filter == "Flagged for Audit ⚠️":
        df_filtered = df_filtered[df_filtered["AuditRequired"] == 1]
    elif risk_filter == "Auto-Approved ✅":
        df_filtered = df_filtered[df_filtered["AuditRequired"] == 0]
        
    st.write(f"Showing **{len(df_filtered):,}** matching transaction records:")
    
    # Table layout
    display_cols = [
        "PONumber", "VendorName", "InvoiceDate", "OrderedDollars", 
        "InvoicedDollars", "price_variance", "InvoicedFreight", "AuditRequired"
    ]
    df_display = df_filtered[display_cols].copy()
    df_display = df_display.rename(columns={
        "PONumber": "PO Number",
        "VendorName": "Vendor Name",
        "InvoiceDate": "Invoice Date",
        "OrderedDollars": "Agreed PO ($)",
        "InvoicedDollars": "Billed Invoice ($)",
        "price_variance": "Price Variance ($)",
        "InvoicedFreight": "Freight Billed ($)",
        "AuditRequired": "Audit Required?"
    })
    
    df_display["Invoice Date"] = df_display["Invoice Date"].dt.strftime('%Y-%m-%d')
    df_display["Audit Required?"] = df_display["Audit Required?"].map({1: "⚠️ Yes", 0: "✅ No"})
    
    st.dataframe(df_display, use_container_width=True)
    
    csv_data = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Flagged Audit Ledger as CSV",
        data=csv_data,
        file_name="AP_invoice_compliance_ledger.csv",
        mime="text/csv"
    )
