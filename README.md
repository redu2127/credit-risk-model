cat > README.md <<'EOF'
# Credit Risk Probability Model for Alternative Data

## Project Overview

This project is part of the 10 Academy Artificial Intelligence Mastery Week 4 Challenge. The goal is to build a credit risk probability model for Bati Bank using transaction data from the Xente eCommerce platform.

Bati Bank wants to support a buy-now-pay-later service. Since the dataset does not contain a direct loan default label, this project will use customer transaction behavior to create a proxy risk label.

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord influence the need for an interpretable and well-documented model?

The Basel II Accord emphasizes accurate risk measurement, documentation, transparency, and sound risk management. For a credit risk model, this means the bank must be able to explain how risk is measured and how credit decisions are made.

An interpretable model is important because credit decisions affect customers directly. If a customer is classified as high risk, the bank should be able to explain the reason. A well-documented model also helps auditors, regulators, and internal risk teams review the model.

In this project, Basel II influences the modeling approach by requiring clear documentation of the data, assumptions, proxy target variable, feature engineering process, model choice, evaluation metrics, and limitations.

---

### 2. Why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The dataset does not include a direct default label. There is no column showing whether a customer failed to repay a loan. Because supervised machine learning requires a target variable, a proxy variable is necessary.

The proxy target will be created using customer behavior, especially RFM metrics:

- Recency: how recently the customer made a transaction
- Frequency: how often the customer transacts
- Monetary: how much value the customer generates

Customers with low engagement may be treated as high-risk proxy customers.

However, this introduces business risks. The proxy is not actual loan default. Some customers may be incorrectly classified as risky even if they would repay a loan. Other customers may appear active but still default. This can lead to unfair credit decisions, lost revenue, or increased credit losses.

Therefore, the proxy variable must be clearly documented as an assumption and should eventually be validated using real repayment data.

---

### 3. What are the trade-offs between Logistic Regression with WoE and Gradient Boosting?

Logistic Regression with Weight of Evidence is simple, interpretable, and commonly used in credit scoring. It is easier to explain to regulators and business teams. It also supports scorecard-style modeling.

Gradient Boosting can capture complex patterns and may produce better predictive performance. However, it is harder to explain and may require additional interpretability tools.

In a regulated financial context, the best model is not always the most complex one. The model must balance predictive performance, interpretability, fairness, and regulatory acceptability.

For this project, Logistic Regression will be used as an interpretable baseline, while more complex models can be compared later.
EOF