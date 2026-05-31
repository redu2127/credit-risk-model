from pydantic import BaseModel


class CustomerData(BaseModel):
    total_transaction_amount: float
    average_transaction_amount: float
    transaction_count: float
    std_transaction_amount: float
    total_value: float
    average_value: float
    average_transaction_hour: float
    average_transaction_day: float
    average_transaction_month: float
    fraud_count: float
    fraud_rate: float
    Recency: float
    Frequency: float
    Monetary: float
    