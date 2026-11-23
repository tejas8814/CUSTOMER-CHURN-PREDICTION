import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

@st.cache_resource
def train_models():
    # 1. Load dataset
    data = pd.read_csv(DATA_FILE)

    # 2. Clean TotalCharges: some rows contain blank strings
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # 3. Remove ID because it does not help predict customer behaviour
    data = data.drop(columns=["customerID"])

    # 4. Separate features (X) and target (y)
    X = data.drop(columns=["Churn"])
    y = data["Churn"].map({"No": 0, "Yes": 1})

    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessing for numerical data
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Apply correct preprocessing to each column
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # Keep preprocessing inside the pipeline to avoid data leakage
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    }

    # Hold out test data. It is not used for model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Compare models using cross-validation on training data only
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []
    fitted_models = {}

    for name, classifier in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("model", classifier)
        ])
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        results.append({
            "Model": name,
            "Mean CV ROC-AUC": scores.mean(),
            "CV Std": scores.std()
        })

        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values(
        "Mean CV ROC-AUC", ascending=False
    ).reset_index(drop=True)

    # Select model based on training cross-validation, then test only once
    best_name = results_df.loc[0, "Model"]
    best_model = fitted_models[best_name]
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test, predictions, target_names=["No Churn", "Churn"], zero_division=0
    )

    return data, X.columns.tolist(), numeric_features, categorical_features, results_df, best_name, best_model, metrics, cm, report


st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to leave the service.")

try:
    data, feature_names, numeric_features, categorical_features, results_df, best_name, best_model, metrics, cm, report = train_models()
except FileNotFoundError:
    st.error(f"Dataset not found. Keep {DATA_FILE} in the same folder as app.py.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "🔮 Predict Churn", "📁 Dataset"])

with tab1:
    st.subheader("Model performance")
    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
    b.metric("Precision", f"{metrics['Precision']:.1%}")
    c.metric("Recall", f"{metrics['Recall']:.1%}")
    d.metric("F1-score", f"{metrics['F1-score']:.1%}")
    e.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

    st.success(f"Selected model: {best_name}")
    st.subheader("Model comparison (3-fold cross-validation)")
    st.dataframe(results_df, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1], ["No Churn", "Churn"])
        ax.set_yticks([0, 1], ["No Churn", "Churn"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Classification Report")
        st.text(report)

    st.caption("The test set is kept separate from cross-validation and is used only for final evaluation.")

with tab2:
    st.subheader("Enter customer details")
    st.info("Choose the customer's details below and click Predict.")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        with col2:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
            total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)

        submitted = st.form_submit_button("Predict Churn", type="primary")

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }], columns=feature_names)

        prediction = int(best_model.predict(customer)[0])
        probability = float(best_model.predict_proba(customer)[0][1])

        if prediction == 1:
            st.error("Prediction: Customer is likely to CHURN.")
        else:
            st.success("Prediction: Customer is likely to STAY.")

        st.metric("Estimated churn probability", f"{probability:.1%}")
        st.progress(probability)

with tab3:
    st.subheader("Dataset overview")
    x, y, z = st.columns(3)
    x.metric("Total customers", f"{len(data):,}")
    y.metric("Total features", len(feature_names))
    z.metric("Churned customers", int((data["Churn"] == "Yes").sum()))
    st.write("Preview:")
    st.dataframe(data.head(20), use_container_width=True)
    st.write("Churn distribution:")
    st.bar_chart(data["Churn"].value_counts())
