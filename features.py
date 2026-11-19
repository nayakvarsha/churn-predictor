import pandas as pd

SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

def add_features(df):
    df = df.copy()                                                    # never modify the caller's data
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)  # blanks -> 0 (tenure-0 customers)
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)      # count of add-ons per customer
    df["charge_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)      # +1 avoids divide-by-zero
    df["tenure_bucket"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                                 labels=["0-1y", "1-2y", "2-4y", "4y+"]).astype(str)
    return df