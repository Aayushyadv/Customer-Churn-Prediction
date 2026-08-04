
# Customer Churn Prediction App (v2.0)

An end-to-end telecom customer churn prediction system: real trained ML model,
19 customer features, gauge/bar/pie visualizations, business recommendations,
model info page, and an About page — built for a Data Analyst / Business Analyst portfolio.

## Project structure
```
churn_app/
├── app.py                  # Streamlit app (Home, Prediction, Model Info, About)
├── generate_data.py         # Builds the synthetic-but-realistic Telco-style dataset
├── train_model.py           # Trains & evaluates the Random Forest model
├── telco_churn.csv          # Generated training dataset
├── model.pkl                 # Trained Random Forest model
├── encoders.pkl               # Saved LabelEncoders for categorical fields
├── feature_columns.pkl        # Ordered feature list the model expects
├── metrics.json                # Accuracy / ROC-AUC / precision / recall / feature importances
├── churn_distribution.json     # Churn vs. no-churn split (for the Home page pie chart)
├── requirements.txt
└── .streamlit/config.toml      # Dark theme config
```

## How to run locally

```bash
pip install -r requirements.txt

# (Artifacts are already generated & committed, but to regenerate from scratch:)
python generate_data.py
python train_model.py

streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## About the dataset
The dataset is synthetically generated to mirror the structure and category names of the
well-known **IBM Telco Customer Churn** dataset (19 features: demographics, account info,
subscribed services, billing, and churn label), with churn probability driven by a logical
formula (contract type, tenure, monthly charges, support add-ons, payment method, etc.) so the
model learns realistic, explainable relationships. Swap in the real IBM dataset (or your own
company data) by replacing `telco_churn.csv` with the same column names and re-running
`train_model.py`.

## Deploying
This app is ready for **Streamlit Community Cloud**:
1. Push this folder to a public GitHub repo.
2. On https://share.streamlit.io, point to `app.py` in that repo.
3. Add the repo/live-app links to the About page and your resume/LinkedIn.

## Suggested next steps (see About page in-app)
- SHAP-based per-prediction explainability
- Power BI dashboard for churn KPIs
- CSV/PDF export of prediction reports

---
Developed by **Ayush Yadav** — M.Sc. Data Science — Python | SQL | Power BI | Machine Learning
