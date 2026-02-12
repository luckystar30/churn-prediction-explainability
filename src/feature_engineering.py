
def engineer_features(df):
    df["revenue_lifetime"] = df["tenure_months"] * df["monthly_charges"]
    df["engagement_score"] = (
        df["usage_minutes_last_30d"] * 0.6 +
        df["avg_sessions_per_week"] * 0.4
    )
    return df
