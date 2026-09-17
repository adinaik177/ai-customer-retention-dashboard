# AI Customer Churn & Retention Dashboard

An independent personal project exploring how a machine-learning model can support customer-retention decisions in a digital business. The application estimates churn risk, visualises customer patterns and suggests human-reviewed retention actions.

## Business problem

Customer churn reduces recurring revenue and increases customer-acquisition pressure. This project demonstrates how a business could use customer behaviour data to identify higher-risk accounts, prioritise interventions and estimate the possible value of a retention campaign.

## What the application demonstrates

- Synthetic customer-data generation
- Data preparation and feature processing
- Logistic-regression classification with scikit-learn
- Churn-probability predictions
- Business KPIs and data visualisation
- Transparent retention recommendations
- Scenario-based business-value estimation
- Model-performance reporting
- Responsible-AI controls and limitations

## Technology

- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- GitHub

## Live application

Add Streamlit link here after deployment.

## Run locally

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Install the packages:

```bash
pip install -r requirements.txt
```

4. Start the application:

```bash
streamlit run app.py
```

## Deploy free with Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt` and this `README.md` to a public GitHub repository.
2. Go to `https://share.streamlit.io` and sign in with GitHub.
3. Select **Create app**.
4. Choose the repository and the `main` branch.
5. Set the main file path to `app.py`.
6. Select **Deploy**. No API key or paid service is required.

## Method

The application creates a reproducible fictional dataset containing monthly spend, tenure, support-ticket volume, satisfaction, digital engagement and contract type. It splits the data into training and test sets, scales numerical variables, one-hot encodes contract type and trains a logistic-regression classifier.

The predicted probability supports a low-, medium- or high-risk category. Retention actions are produced through explicit business rules so that the distinction between model prediction and operational recommendation remains clear.

## Responsible use

All customer records are synthetic. The application is an educational portfolio demonstration and must not be used for real customer decisions. A production implementation would require lawful data processing, privacy safeguards, representative data, fairness assessment, stakeholder approval, human oversight and continuous monitoring.

## Skills demonstrated

Machine Learning, Artificial Intelligence, Digital Business, Predictive Analytics, Data Visualisation, Business Analysis, Python, Streamlit, Responsible AI, Decision Support and GitHub.
