import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------
# Load model artifacts (cached so they load once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    with open("metrics.json") as f:
        metrics = json.load(f)
    with open("churn_distribution.json") as f:
        churn_dist = json.load(f)
    return model, encoders, feature_cols, metrics, churn_dist

model, encoders, feature_cols, metrics, churn_dist = load_artifacts()

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# ----------------------------------------------------------------------
# Minimal CSS polish (cards, badges) on top of the dark theme
# ----------------------------------------------------------------------
st.markdown("""
<style>
.result-card {
    padding: 1.4rem 1.6rem; border-radius: 10px; margin-bottom: 1rem;
}
.card-high   { background-color: #3a1414; border: 1px solid #ef4444; }
.card-low    { background-color: #12321c; border: 1px solid #22c55e; }
.metric-box  { background-color: #1a1d24; border-radius: 10px; padding: 1rem 1.2rem; text-align:center; }
.footer-box  { text-align:center; color:#9ca3af; padding-top: 2rem; padding-bottom: 1rem; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Prediction", "Model Info", "About"], label_visibility="visible")

# ========================================================================
# HOME PAGE
# ========================================================================
if page == "Home":
    st.markdown("## 📊 Customer Churn Prediction")
    st.write("This application predicts whether a telecom customer is likely to churn using a trained **Machine Learning model** (Random Forest).")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-box"><h3>{metrics["accuracy"]}%</h3>Accuracy</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><h3>{metrics["roc_auc"]}</h3>ROC-AUC</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><h3>{metrics["precision"]}%</h3>Precision</div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-box"><h3>{metrics["recall"]}%</h3>Recall</div>', unsafe_allow_html=True)

    st.markdown("### Dataset Churn Distribution")
    fig = px.pie(
        names=list(churn_dist.keys()), values=list(churn_dist.values()),
        color=list(churn_dist.keys()),
        color_discrete_map={"Yes": "#ef4444", "No": "#22c55e"},
        hole=0.5
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fafafa")
    st.plotly_chart(fig, use_container_width=True)

    st.info("👉 Go to **Prediction** in the sidebar to score a customer.")

# ========================================================================
# PREDICTION PAGE
# ========================================================================
elif page == "Prediction":
    st.markdown("## 🔮 Customer Churn Prediction")
    st.caption("Fill in the customer's profile below, then click Predict.")

    with st.form("prediction_form"):
        st.markdown("#### Demographics")
        col1, col2, col3, col4 = st.columns(4)
        gender = col1.selectbox("Gender", ["Female", "Male"])
        senior = col2.selectbox("Senior Citizen", ["No", "Yes"])
        partner = col3.selectbox("Partner", ["Yes", "No"])
        dependents = col4.selectbox("Dependents", ["Yes", "No"])

        st.markdown("#### Account Info")
        col1, col2, col3 = st.columns(3)
        tenure = col1.slider("Tenure (months)", 1, 72, 10)
        contract = col2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = col3.selectbox("Paperless Billing", ["Yes", "No"])

        col1, col2 = st.columns(2)
        payment_method = col1.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = col2.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=65.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=10000.0, value=float(round(monthly_charges * tenure, 2)), step=10.0)

        st.markdown("#### Services")
        col1, col2, col3 = st.columns(3)
        phone_service = col1.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = col2.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = col3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        col1, col2, col3 = st.columns(3)
        online_security = col1.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = col2.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = col3.selectbox("Device Protection", ["Yes", "No", "No internet service"])

        col1, col2, col3 = st.columns(3)
        tech_support = col1.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = col2.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = col3.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("Predict", use_container_width=False)

    if submitted:
        raw_input = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "tenure": tenure,
        }

        # ---- Build feature row in the exact order the model expects ----
        row = {}
        for col in feature_cols:
            if col in CATEGORICAL_COLS:
                le = encoders[col]
                val = raw_input[col]
                if val not in le.classes_:
                    val = le.classes_[0]  # safety fallback
                row[col] = le.transform([val])[0]
            else:
                row[col] = raw_input[col]
        X = pd.DataFrame([row])[feature_cols]

        proba = model.predict_proba(X)[0][1]
        pred = "Yes" if proba >= 0.5 else "No"
        pct = round(proba * 100, 1)

        if pct >= 75 or pct <= 25:
            confidence = "High"
        elif 60 <= pct < 75 or 25 < pct <= 40:
            confidence = "Medium"
        else:
            confidence = "Low"

        st.markdown("---")
        # ---------------- Result card ----------------
        if pred == "Yes":
            st.markdown(f"""
            <div class="result-card card-high">
            <h3>🔴 HIGH RISK — Likely to Churn</h3>
            <p><b>Churn Probability:</b> {pct}%</p>
            <p><b>Confidence:</b> {confidence}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card card-low">
            <h3>🟢 LOW RISK — Likely to Stay</h3>
            <p><b>Churn Probability:</b> {pct}%</p>
            <p><b>Confidence:</b> {confidence}</p>
            </div>""", unsafe_allow_html=True)

        # ---------------- Charts ----------------
        col1, col2 = st.columns(2)

        with col1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                title={"text": "Churn Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if pred == "Yes" else "#22c55e"},
                    "steps": [
                        {"range": [0, 40], "color": "#12321c"},
                        {"range": [40, 70], "color": "#3a2f14"},
                        {"range": [70, 100], "color": "#3a1414"},
                    ],
                }
            ))
            gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#fafafa", height=300, margin=dict(t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True)

        with col2:
            imp = metrics["feature_importances"]
            top8 = dict(list(imp.items())[:8])
            bar = px.bar(
                x=list(top8.values())[::-1], y=list(top8.keys())[::-1],
                orientation="h", labels={"x": "Importance", "y": ""},
                title="Top Contributing Features (Model-Wide)"
            )
            bar.update_traces(marker_color="#ef4444")
            bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fafafa", height=300, margin=dict(t=40, b=10))
            st.plotly_chart(bar, use_container_width=True)

        pie = px.pie(
            names=["Churn", "No Churn"], values=[pct, 100 - pct],
            color=["Churn", "No Churn"],
            color_discrete_map={"Churn": "#ef4444", "No Churn": "#22c55e"},
            hole=0.5, title="This Customer: Churn vs. Stay"
        )
        pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#fafafa", height=320)
        st.plotly_chart(pie, use_container_width=True)

        # ---------------- Business recommendations ----------------
        st.markdown("### 💼 Business Recommendation")
        recs = []
        if pred == "Yes":
            recs.append("Offer a loyalty discount or promotional pricing")
            if contract == "Month-to-month":
                recs.append("Incentivize a switch to a 1-year or 2-year contract")
            if tech_support == "No" and internet_service != "No":
                recs.append("Offer a free trial of Tech Support add-on")
            if online_security == "No" and internet_service != "No":
                recs.append("Bundle in Online Security at a discount")
            if payment_method == "Electronic check":
                recs.append("Encourage a switch to automatic payment (bank/credit card)")
            recs.append("Assign the account to the Retention Team for proactive outreach")
            recs.append("Schedule a customer satisfaction / feedback call")
        else:
            recs.append("Continue standard engagement — customer is stable")
            recs.append("Consider upsell opportunities (streaming, device protection)")
            recs.append("Include in loyalty/referral program outreach")

        for r in recs:
            st.markdown(f"✔ {r}")

        with st.expander("View raw input & encoded features sent to the model"):
            st.write("**Raw input:**")
            st.json(raw_input)
            st.write("**Encoded feature row:**")
            st.dataframe(X)

# ========================================================================
# MODEL INFO PAGE
# ========================================================================
elif page == "Model Info":
    st.markdown("## 🧠 Model Information")
    st.write("The prediction engine behind this app is a **Random Forest Classifier**, trained on a telecom customer dataset with the same structure as the industry-standard IBM Telco Customer Churn dataset.")

    info = {
        "Model": metrics["model_name"],
        "Accuracy": f'{metrics["accuracy"]}%',
        "ROC-AUC": metrics["roc_auc"],
        "Precision": f'{metrics["precision"]}%',
        "Recall": f'{metrics["recall"]}%',
        "Training samples": metrics["n_train"],
        "Test samples": metrics["n_test"],
    }
    st.table(pd.DataFrame(info.items(), columns=["Metric", "Value"]).set_index("Metric"))

    st.markdown("### Full Feature Importance")
    imp_df = pd.DataFrame(metrics["feature_importances"].items(), columns=["Feature", "Importance"])
    fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
    fig.update_traces(marker_color="#ef4444")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fafafa",
                       yaxis={"categoryorder": "total ascending"}, height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Why Random Forest?**
    - Handles a mix of categorical + numeric features without heavy preprocessing
    - Robust to outliers and non-linear relationships (e.g. tenure vs. charges)
    - Provides built-in feature importance for explainability
    - Strong baseline before moving to gradient boosting / SHAP-based explanations
    """)

# ========================================================================
# ABOUT PAGE
# ========================================================================
elif page == "About":
    st.markdown("## ℹ️ About This Project")
    st.markdown("""
### Project Overview
This is an end-to-end **Customer Churn Prediction System** built for a telecom business use case.
It predicts whether a customer is likely to churn, explains *why*, and translates that prediction
into a concrete business recommendation — bridging data science and business/analyst decision-making.

### Workflow
```
User Input → Preprocessing/Encoding → ML Model → Probability → Business Recommendation
```

### Dataset
A telecom customer dataset structured like the widely-used **IBM Telco Customer Churn** dataset,
covering demographics, account information, subscribed services, billing, and churn status.

### Machine Learning
- **Model:** Random Forest Classifier (scikit-learn)
- **Preprocessing:** Label encoding of categorical fields
- **Evaluation:** Accuracy, ROC-AUC, Precision, Recall on a held-out test set
- **Explainability:** Feature importance ranking (SHAP planned as a future enhancement)

### Technologies
- **Python** — data processing & modeling
- **scikit-learn** — Random Forest model, train/test evaluation
- **Pandas / NumPy** — data wrangling
- **Streamlit** — interactive web app
- **Plotly** — gauge, bar, and pie chart visualizations

### Roadmap / Next Steps
- Integrate SHAP for per-customer explainability
- Add a Power BI dashboard for KPI monitoring (churn rate, revenue at risk, contract mix)
- Add CSV/PDF export of prediction reports
- Deploy on Streamlit Community Cloud with a public GitHub repo

### Links
- 🔗 GitHub: *add your repo link here*
- 🔗 LinkedIn: *add your profile link here*
""")

# ------------------------------------------------------------------------
# Footer (shown on every page)
# ------------------------------------------------------------------------
st.markdown("""
<div class="footer-box">
Developed by <b>Ayush Yadav</b><br>
M.Sc. Data Science<br>
Python | SQL | Power BI | Machine Learning
</div>
""", unsafe_allow_html=True)
