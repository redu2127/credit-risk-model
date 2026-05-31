\# Interim Report: Credit Risk Probability Model for Alternative Data



\## 1. Introduction



Bati Bank is partnering with an eCommerce company to support a buy-now-pay-later service. The goal of this project is to build a credit risk probability model that can help identify customers as high risk or low risk.



The dataset contains transaction-level records from the Xente eCommerce platform, including customer identifiers, transaction amounts, timestamps, product information, channels, pricing strategy, and fraud result.



The main challenge is that the dataset does not contain a direct loan default label. Therefore, a proxy target variable must be created using customer transaction behavior.



\## 2. Business Understanding



Credit scoring estimates the likelihood that a borrower may default. For Bati Bank, this model can support credit approval, risk probability estimation, and customer segmentation for a buy-now-pay-later product.



Because credit decisions affect customers directly, the model must be explainable, well documented, and carefully evaluated.



\## 3. Basel II and Interpretability



The Basel II Accord emphasizes risk measurement, documentation, and transparency. This means the credit risk model should be interpretable and reproducible.



The bank should be able to explain what data was used, how the proxy target was created, what features were engineered, which model was selected, and what limitations exist.



For this reason, Logistic Regression with Weight of Evidence is suitable as an interpretable baseline, while complex models such as Gradient Boosting can be used for comparison.



\## 4. Proxy Target Variable



The dataset does not include an actual default label. Since supervised machine learning requires a target variable, a proxy target must be created.



The planned proxy target will be based on RFM metrics:



\- Recency

\- Frequency

\- Monetary value



Customers with low engagement, low transaction frequency, and low monetary value may be labeled as high-risk proxy customers.



However, this is not actual loan default. It is only a behavioral assumption. This creates business risks such as misclassification, unfair credit decisions, and possible credit losses. Therefore, the proxy variable must be clearly documented and later validated with real repayment data.



\## 5. Exploratory Data Analysis



The EDA was completed in notebooks/eda.ipynb.



The analysis covered:



\- Dataset overview

\- Summary statistics

\- Missing values

\- Duplicate records

\- Numerical feature distributions

\- Categorical feature distributions

\- Correlation analysis

\- Outlier detection

\- Time-based transaction analysis

\- Customer-level analysis

\- FraudResult distribution



\## 6. Key EDA Findings



\### Insight 1: Transaction values are skewed



Most transactions are small, while a few transactions have very large values. This suggests that scaling and outlier handling will be important.



\### Insight 2: Customer behavior varies



Some customers transact frequently, while others make very few transactions. This supports using RFM analysis for customer segmentation.



\### Insight 3: Categorical variables are imbalanced



Features such as product category, provider, channel, and pricing strategy have uneven category distributions. Encoding should be handled carefully.



\### Insight 4: FraudResult is imbalanced



Fraud cases are rare compared with non-fraud cases. FraudResult is not the same as default, but it may provide useful risk-related context.



\### Insight 5: Time features may be useful



Transaction hour, day, month, and year may capture useful customer behavior patterns.



\## 7. Next Steps



The final phase will include:



1\. Feature engineering pipeline

2\. RFM metric calculation

3\. K-Means clustering

4\. Creation of the is\_high\_risk proxy target

5\. Model training and comparison

6\. MLflow experiment tracking

7\. FastAPI deployment

8\. Docker containerization

9\. GitHub Actions CI/CD



\## 8. Conclusion

The interim phase completed the business understanding and exploratory analysis. The main modeling challenge is the absence of a true default label. The next phase will address this by creating a proxy target using RFM-based customer segmentation.

