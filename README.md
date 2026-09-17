# AI Customer Churn & Retention Dashboard

An interactive decision-support application that demonstrates how machine learning can help a digital business identify customers at risk of leaving, prioritise retention activity and estimate the potential business value of an intervention.

**[Open the live application](https://ai-customer-retention-dashboard-neryqhtoam5kzpaz4s7ncy.streamlit.app/)**

## Business problem

Customer churn can reduce recurring revenue and increase acquisition pressure. This project translates customer-behaviour data into a practical workflow: understand the customer base, estimate individual churn risk, review suggested actions and explore a retention scenario.

## Key features

- Business overview with customer and churn KPIs
- Interactive charts for customer patterns
- Customer-level churn probability
- Low-, medium- and high-risk classification
- Transparent, human-reviewed retention recommendations
- Scenario-based business-value estimation
- Model evaluation using accuracy and ROC-AUC
- Responsible-AI limitations and controls

## How it works

The application generates a reproducible synthetic dataset containing monthly spend, tenure, satisfaction, support-ticket volume, digital engagement and contract type. It then:

1. Splits the data into training and test sets.
2. Scales numerical variables and one-hot encodes contract type.
3. Trains a logistic-regression classifier with scikit-learn.
4. Converts predicted probabilities into clear risk categories.
5. Uses explicit business rules to suggest reviewable retention actions.

The model prediction and the operational recommendation are intentionally separated so that a human remains responsible for the final decision.

## Technology

- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- Git and GitHub

## Run locally

```bash
git clone https://github.com/adinaik177/ai-customer-retention-dashboard.git
cd ai-customer-retention-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Responsible use

All customer records are synthetic. This is an educational portfolio project, not a production system or a tool for real customer decisions. A real implementation would require lawful data processing, privacy safeguards, representative data, fairness testing, stakeholder approval, human oversight and continuous monitoring.

## Skills demonstrated

Machine Learning · Python · Predictive Analytics · Data Visualisation · Digital Business · Business Analysis · Streamlit · Responsible AI · Decision Support
