
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

data = pd.DataFrame({
    "customer_id": range(1, n+1),
    "tenure_months": np.random.randint(1, 60, n),
    "monthly_charges": np.random.normal(75, 20, n).clip(20, 150),
    "usage_minutes_last_30d": np.random.normal(400, 120, n).clip(50, 800),
    "avg_sessions_per_week": np.random.normal(5, 2, n).clip(1, 15),
    "support_tickets_last_90d": np.random.poisson(1.5, n),
})

data["usage_trend_ratio"] = np.random.normal(1.0, 0.2, n).clip(0.5, 1.5)

data["churn"] = (
    (data["tenure_months"] < 6).astype(int) +
    (data["support_tickets_last_90d"] > 3).astype(int) +
    (data["usage_trend_ratio"] < 0.8).astype(int)
)

data["churn"] = (data["churn"] > 1).astype(int)

data.to_csv("data/subscription_usage_data.csv", index=False)
print("Dataset generated.")
