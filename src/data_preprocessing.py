import pandas as pd
from sklearn.preprocessing import StandardScaler


RAW_DATA_PATH = "data/raw/data.csv"
PROCESSED_DATA_PATH = "data/processed/features.csv"


def load_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)


def create_features(df):
    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year

    customer_features = df.groupby("CustomerId").agg(
        total_transaction_amount=("Amount", "sum"),
        average_transaction_amount=("Amount", "mean"),
        transaction_count=("TransactionId", "count"),
        std_transaction_amount=("Amount", "std"),
        total_value=("Value", "sum"),
        average_value=("Value", "mean"),
        average_transaction_hour=("transaction_hour", "mean"),
        average_transaction_day=("transaction_day", "mean"),
        average_transaction_month=("transaction_month", "mean"),
        fraud_count=("FraudResult", "sum"),
        fraud_rate=("FraudResult", "mean"),
    ).reset_index()

    customer_features["std_transaction_amount"] = (
        customer_features["std_transaction_amount"].fillna(0)
    )

    return customer_features


def scale_features(df):
    df = df.copy()

    id_col = df[["CustomerId"]]

    numeric_cols = df.drop(columns=["CustomerId"]).columns

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(df[numeric_cols])

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=numeric_cols
    )

    final_df = pd.concat(
        [id_col.reset_index(drop=True), scaled_df],
        axis=1
    )

    return final_df


def main():
    df = load_data()
    features = create_features(df)
    scaled_features = scale_features(features)

    scaled_features.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Task 3 completed successfully.")
    print(f"Processed features saved to: {PROCESSED_DATA_PATH}")
    print("Shape:", scaled_features.shape)


if __name__ == "__main__":
    main()