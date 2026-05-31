import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


RAW_DATA_PATH = "data/raw/data.csv"
FEATURES_PATH = "data/processed/features.csv"
MODEL_DATA_PATH = "data/processed/model_data.csv"


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


def create_rfm_target(raw_df, features_df):
    raw_df = raw_df.copy()

    raw_df["TransactionStartTime"] = pd.to_datetime(
        raw_df["TransactionStartTime"]
    )

    snapshot_date = raw_df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = raw_df.groupby("CustomerId").agg(
        Recency=(
            "TransactionStartTime",
            lambda x: (snapshot_date - x.max()).days
        ),
        Frequency=(
            "TransactionId",
            "count"
        ),
        Monetary=(
            "Value",
            "sum"
        )
    ).reset_index()

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby("cluster").agg(
        avg_recency=("Recency", "mean"),
        avg_frequency=("Frequency", "mean"),
        avg_monetary=("Monetary", "mean"),
        customer_count=("CustomerId", "count")
    )

    print("\nCluster Summary:")
    print(cluster_summary)

    high_risk_cluster = cluster_summary["avg_frequency"].idxmin()

    print("\nHigh risk cluster selected:", high_risk_cluster)

    rfm["is_high_risk"] = (
        rfm["cluster"] == high_risk_cluster
    ).astype(int)

    final_df = features_df.merge(
        rfm[
            [
                "CustomerId",
                "Recency",
                "Frequency",
                "Monetary",
                "is_high_risk"
            ]
        ],
        on="CustomerId",
        how="left"
    )

    return final_df


def main():
    raw_df = load_data()

    features = create_features(raw_df)
    features.to_csv(FEATURES_PATH, index=False)

    scaled_features = scale_features(features)

    final_df = create_rfm_target(
        raw_df,
        scaled_features
    )

    final_df.to_csv(MODEL_DATA_PATH, index=False)

    print("\nTask 3 and Task 4 completed successfully.")
    print(f"Features saved to: {FEATURES_PATH}")
    print(f"Model data saved to: {MODEL_DATA_PATH}")
    print("\nTarget distribution:")
    print(final_df["is_high_risk"].value_counts())


if __name__ == "__main__":
    main()
    