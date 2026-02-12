
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from feature_engineering import engineer_features

df = pd.read_csv("data/subscription_usage_data.csv")
df = engineer_features(df)

features = [
    "tenure_months",
    "monthly_charges",
    "usage_minutes_last_30d",
    "avg_sessions_per_week",
    "support_tickets_last_90d",
    "usage_trend_ratio",
    "revenue_lifetime",
    "engagement_score"
]

X = df[features]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict_proba(X_test)[:,1]
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, lr_preds))

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict_proba(X_test)[:,1]
print("XGBoost ROC-AUC:", roc_auc_score(y_test, xgb_preds))

joblib.dump(lr, "models/logistic_regression.pkl")
joblib.dump(xgb, "models/xgboost_model.pkl")

explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)

plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.savefig("reports/shap_summary.png")
plt.close()

predictions = X_test.copy()
predictions["actual_churn"] = y_test.values
predictions["lr_probability"] = lr_preds
predictions["xgb_probability"] = xgb_preds
predictions.to_csv("powerbi/churn_predictions_for_powerbi.csv", index=False)
