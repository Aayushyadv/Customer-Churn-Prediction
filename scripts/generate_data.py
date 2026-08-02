"""
Generates a synthetic but realistic Telco Customer Churn dataset.
Mirrors the structure of the well-known IBM Telco Customer Churn dataset
(same column names/categories), with churn probability driven by a logical
formula so the trained model learns sensible, explainable relationships.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 4000

gender = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])
tenure = np.random.randint(1, 73, N)
phone_service = np.random.choice(["Yes", "No"], N, p=[0.90, 0.10])

multiple_lines = np.array([
    "No phone service" if ps == "No" else np.random.choice(["Yes", "No"], p=[0.42, 0.58])
    for ps in phone_service
])

internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])

def dep_internet_choice(net, p_yes=0.5):
    if net == "No":
        return "No internet service"
    return np.random.choice(["Yes", "No"], p=[p_yes, 1 - p_yes])

online_security = np.array([dep_internet_choice(n, 0.35) for n in internet_service])
online_backup = np.array([dep_internet_choice(n, 0.40) for n in internet_service])
device_protection = np.array([dep_internet_choice(n, 0.40) for n in internet_service])
tech_support = np.array([dep_internet_choice(n, 0.35) for n in internet_service])
streaming_tv = np.array([dep_internet_choice(n, 0.45) for n in internet_service])
streaming_movies = np.array([dep_internet_choice(n, 0.45) for n in internet_service])

contract = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.24, 0.21])
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21]
)

base_charge = np.where(internet_service == "Fiber optic", 70,
              np.where(internet_service == "DSL", 45, 20))
addon_cols = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
addon_cost = sum((col == "Yes").astype(int) for col in addon_cols) * 5
phone_cost = np.where(phone_service == "Yes", 5, 0)
noise = np.random.normal(0, 5, N)
monthly_charges = np.clip(base_charge + addon_cost + phone_cost + noise, 18, 120).round(2)

total_charges = np.clip(monthly_charges * tenure + np.random.normal(0, 50, N), 18, None).round(2)

# --- Logical churn probability formula ---
logit = (
    -1.2
    + 1.6 * (contract == "Month-to-month")
    - 0.9 * (contract == "Two year")
    - 0.03 * tenure
    + 0.015 * (monthly_charges - 60)
    + 0.5 * (internet_service == "Fiber optic")
    - 0.5 * (online_security == "Yes")
    - 0.45 * (tech_support == "Yes")
    + 0.35 * (paperless_billing == "Yes")
    + 0.4 * (payment_method == "Electronic check")
    - 0.3 * (partner == "Yes")
    - 0.2 * (dependents == "Yes")
    + 0.15 * senior_citizen
    + np.random.normal(0, 0.35, N)
)
prob_churn = 1 / (1 + np.exp(-logit))
churn = np.where(np.random.rand(N) < prob_churn, "Yes", "No")

df = pd.DataFrame({
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
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
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn,
})

df.to_csv("telco_churn.csv", index=False)
print(df.shape)
print(df["Churn"].value_counts(normalize=True))
