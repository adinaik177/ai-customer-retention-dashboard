import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="AI Customer Retention Dashboard",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data
def create_customer_data(number_of_customers: int = 600) -> pd.DataFrame:
    """Create reproducible fictional customer data for this portfolio project."""
    rng = np.random.default_rng(42)

    monthly_spend = rng.normal(62, 22, number_of_customers).clip(10, 160)
    tenure_months = rng.integers(1, 61, number_of_customers)
    support_tickets = rng.poisson(2.2, number_of_customers).clip(0, 10)
    satisfaction_score = rng.integers(1, 6, number_of_customers)
    digital_engagement = rng.integers(5, 101, number_of_customers)
    contract_type = rng.choice(
        ["Monthly", "Annual", "Two-year"],
        size=number_of_customers,
        p=[0.52, 0.32, 0.16],
    )

    churn_logit = (
        -1.1
        + 0.36 * support_tickets
        - 0.58 * satisfaction_score
        - 0.018 * tenure_months
        - 0.012 * digital_engagement
        + 1.05 * (contract_type == "Monthly")
        + 0.22 * (monthly_spend > 90)
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))
    churned = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "monthly_spend": monthly_spend.round(2),
            "tenure_months": tenure_months,
            "support_tickets": support_tickets,
            "satisfaction_score": satisfaction_score,
            "digital_engagement": digital_engagement,
            "contract_type": contract_type,
            "churned": churned,
        }
    )


@st.cache_resource
def train_model(data: pd.DataFrame):
    features = [
        "monthly_spend",
        "tenure_months",
        "support_tickets",
        "satisfaction_score",
        "digital_engagement",
        "contract_type",
    ]
    target = "churned"

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric_features = [
        "monthly_spend",
        "tenure_months",
        "support_tickets",
        "satisfaction_score",
        "digital_engagement",
    ]
    categorical_features = ["contract_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "test_size": len(y_test),
    }
    return model, metrics


def retention_recommendation(customer: dict, risk: float) -> list[str]:
    actions = []
    if customer["satisfaction_score"] <= 2:
        actions.append("Arrange a service-recovery call and record the main concern.")
    if customer["support_tickets"] >= 4:
        actions.append("Escalate unresolved support issues to a senior support owner.")
    if customer["digital_engagement"] < 35:
        actions.append("Send a personalised onboarding or product-education campaign.")
    if customer["contract_type"] == "Monthly":
        actions.append("Offer an annual-plan incentive only if it is suitable for the customer.")
    if customer["monthly_spend"] > 90:
        actions.append("Review whether the current plan delivers clear value for its price.")
    if not actions and risk < 0.4:
        actions.append("Maintain normal engagement and monitor future satisfaction signals.")
    return actions[:3]


data = create_customer_data()
model, model_metrics = train_model(data)

st.title("🧠 AI Customer Churn & Retention Dashboard")
st.write(
    "A digital-business decision-support prototype that uses machine learning "
    "to identify customers at risk of leaving and recommend human-reviewed retention actions."
)
st.info(
    "Portfolio demonstration: all customer records are synthetic. Predictions are educational, "
    "not operational business advice."
)

overview_tab, prediction_tab, impact_tab, responsibility_tab = st.tabs(
    [
        "Business Overview",
        "Customer Prediction",
        "Business Impact",
        "Model & Responsible AI",
    ]
)

with overview_tab:
    st.subheader("Customer retention overview")
    churn_rate = data["churned"].mean()
    average_spend = data["monthly_spend"].mean()
    high_risk_count = int(data["churned"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers analysed", f"{len(data):,}")
    col2.metric("Observed churn rate", f"{churn_rate:.1%}")
    col3.metric("Average monthly spend", f"£{average_spend:,.2f}")
    col4.metric("Churned customers", f"{high_risk_count:,}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Churn rate by contract")
        contract_churn = (
            data.groupby("contract_type")["churned"].mean().mul(100).round(1)
        )
        st.bar_chart(contract_churn, color="#ff4b4b")
    with right:
        st.markdown("#### Churn rate by satisfaction score")
        satisfaction_churn = (
            data.groupby("satisfaction_score")["churned"].mean().mul(100).round(1)
        )
        st.line_chart(satisfaction_churn, color="#0068c9")

    st.markdown("#### Business interpretation")
    st.write(
        "The dashboard turns customer behaviour into an operational decision: who may need "
        "attention, which factors require investigation, and which retention action could be reviewed."
    )

with prediction_tab:
    st.subheader("Predict an individual customer's churn risk")
    st.caption("Adjust the inputs, then select Predict churn risk.")

    input_left, input_right = st.columns(2)
    with input_left:
        monthly_spend = st.slider("Monthly spend (£)", 10, 160, 65)
        tenure_months = st.slider("Customer tenure (months)", 1, 60, 18)
        support_tickets = st.slider("Support tickets in recent period", 0, 10, 2)
    with input_right:
        satisfaction_score = st.slider("Satisfaction score", 1, 5, 3)
        digital_engagement = st.slider("Digital engagement score", 5, 100, 55)
        contract_type = st.selectbox(
            "Contract type", ["Monthly", "Annual", "Two-year"]
        )

    if st.button("Predict churn risk", type="primary"):
        customer = {
            "monthly_spend": monthly_spend,
            "tenure_months": tenure_months,
            "support_tickets": support_tickets,
            "satisfaction_score": satisfaction_score,
            "digital_engagement": digital_engagement,
            "contract_type": contract_type,
        }
        customer_frame = pd.DataFrame([customer])
        risk = float(model.predict_proba(customer_frame)[0, 1])

        if risk >= 0.65:
            label, status = "High risk", "error"
        elif risk >= 0.40:
            label, status = "Medium risk", "warning"
        else:
            label, status = "Low risk", "success"

        st.metric("Predicted churn probability", f"{risk:.1%}")
        getattr(st, status)(f"Decision-support category: {label}")

        st.markdown("#### Suggested human-reviewed actions")
        for action in retention_recommendation(customer, risk):
            st.write(f"• {action}")

        st.caption(
            "The probability comes from the trained model. The suggested actions are transparent "
            "business rules and must be reviewed by a person before use."
        )

with impact_tab:
    st.subheader("Estimate potential retention value")
    st.write(
        "This scenario calculator connects the prediction workflow to a measurable business outcome."
    )

    customers_contacted = st.slider("High-risk customers contacted", 10, 500, 100)
    retention_success = st.slider("Assumed retention success rate", 1, 50, 15)
    monthly_value = st.slider("Average monthly customer value (£)", 10, 200, 60)
    value_months = st.slider("Value period (months)", 1, 24, 12)

    retained_customers = customers_contacted * (retention_success / 100)
    estimated_value = retained_customers * monthly_value * value_months

    value_col1, value_col2 = st.columns(2)
    value_col1.metric("Estimated customers retained", f"{retained_customers:.0f}")
    value_col2.metric("Illustrative retained revenue", f"£{estimated_value:,.0f}")

    st.warning(
        "This is a scenario estimate, not a guaranteed financial return. A real business would "
        "validate the assumptions through a controlled pilot."
    )

with responsibility_tab:
    st.subheader("Model performance")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Test accuracy", f"{model_metrics['accuracy']:.1%}")
    metric_col2.metric("ROC-AUC", f"{model_metrics['roc_auc']:.2f}")
    metric_col3.metric("Test records", f"{model_metrics['test_size']}")

    matrix = model_metrics["confusion_matrix"]
    matrix_frame = pd.DataFrame(
        matrix,
        index=["Actual: stayed", "Actual: churned"],
        columns=["Predicted: stayed", "Predicted: churned"],
    )
    st.markdown("#### Confusion matrix")
    st.dataframe(matrix_frame, use_container_width=True)

    st.markdown("#### Responsible-AI controls")
    controls = pd.DataFrame(
        {
            "Control": [
                "Human oversight",
                "Data transparency",
                "Purpose limitation",
                "Model monitoring",
                "Fairness review",
            ],
            "How it is addressed": [
                "Recommendations require human review before customer contact.",
                "The app clearly identifies the data as synthetic.",
                "The prediction is limited to retention decision support.",
                "Accuracy, ROC-AUC and classification errors are displayed.",
                "A real deployment must test outcomes across relevant customer groups.",
            ],
        }
    )
    st.dataframe(controls, hide_index=True, use_container_width=True)

    st.markdown("#### Known limitations")
    st.write(
        "The dataset is synthetic, the relationships are simplified, and the model has not been "
        "validated with real customers. Sensitive personal characteristics are deliberately excluded. "
        "A production system would require privacy controls, bias testing, monitoring, stakeholder "
        "approval and an evaluation of whether automated prediction is appropriate."
    )

st.divider()
st.caption(
    "Created by Aditya Naik as an independent portfolio project demonstrating AI application "
    "to a digital-business problem."
)
