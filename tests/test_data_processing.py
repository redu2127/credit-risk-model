import pandas as pd

from src.data_preprocessing import create_features


def test_create_features_returns_expected_columns():
    sample_data = pd.DataFrame({
        "CustomerId": ["C1", "C1", "C2"],
        "TransactionId": ["T1", "T2", "T3"],
        "Amount": [100, 200, 50],
        "Value": [100, 200, 50],
        "TransactionStartTime": [
            "2024-01-01 10:00:00",
            "2024-01-02 11:00:00",
            "2024-01-03 12:00:00"
        ],
        "FraudResult": [0, 1, 0],
    })

    features = create_features(sample_data)

    expected_columns = {
        "CustomerId",
        "total_transaction_amount",
        "average_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "total_value",
        "average_value",
        "fraud_count",
        "fraud_rate",
    }

    assert expected_columns.issubset(set(features.columns))


def test_create_features_transaction_count():
    sample_data = pd.DataFrame({
        "CustomerId": ["C1", "C1", "C2"],
        "TransactionId": ["T1", "T2", "T3"],
        "Amount": [100, 200, 50],
        "Value": [100, 200, 50],
        "TransactionStartTime": [
            "2024-01-01 10:00:00",
            "2024-01-02 11:00:00",
            "2024-01-03 12:00:00"
        ],
        "FraudResult": [0, 1, 0],
    })

    features = create_features(sample_data)

    c1_count = features.loc[
        features["CustomerId"] == "C1",
        "transaction_count"
    ].iloc[0]

    assert c1_count == 2
