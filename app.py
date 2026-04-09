import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE

# ===============================
# Page Config
# ===============================

st.set_page_config(page_title="AI Fraud Detection Dashboard", layout="wide")

st.title("💳 AI Fraud Detection Dashboard")
st.markdown("Random Forest + SMOTE | Credit Card Fraud Dataset")

# ===============================
# Load Dataset
# ===============================

@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

data = load_data()

# ===============================
# Sidebar
# ===============================

st.sidebar.header("Controls")
train_button = st.sidebar.button("🚀 Train Model")

# ===============================
# Train Model
# ===============================

if train_button:

    with st.spinner("Training model... Please wait..."):

        # 🔥 Smaller sample for faster demo
        data_sample = data.sample(n=50000, random_state=42)

        X = data_sample.drop("Class", axis=1)
        y = data_sample["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        model = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_resampled, y_train_resampled)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)

        # Save in session state
        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_prob = y_prob
        st.session_state.auc = auc

    st.success("✅ Model Trained Successfully!")

# ===============================
# If Model Exists
# ===============================

if "model" in st.session_state:

    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("ROC-AUC Score", f"{st.session_state.auc:.4f}")

    # -------------------------------
    # Confusion Matrix
    # -------------------------------

    st.subheader("Confusion Matrix")

    y_pred = st.session_state.model.predict(st.session_state.X_test)
    cm = confusion_matrix(st.session_state.y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, use_container_width=False)

    # -------------------------------
    # ROC Curve
    # -------------------------------

    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(
        st.session_state.y_test,
        st.session_state.y_prob
    )

    fig2, ax2 = plt.subplots(figsize=(5,4))
    ax2.plot(fpr, tpr, label=f"AUC = {st.session_state.auc:.4f}")
    ax2.plot([0,1], [0,1], linestyle='--')
    ax2.legend()
    st.pyplot(fig2, use_container_width=False)

    # -------------------------------
    # Feature Importance
    # -------------------------------

    st.subheader("Top 10 Important Features")

    importances = st.session_state.model.feature_importances_
    feature_names = st.session_state.X_test.columns

    feat_imp = pd.Series(importances, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=False).head(10)

    fig3, ax3 = plt.subplots(figsize=(6,4))
    feat_imp.plot(kind="bar", ax=ax3)
    st.pyplot(fig3, use_container_width=False)

    # -------------------------------
    # Random Prediction Section
    # -------------------------------

    st.subheader("🔍 Test Random Transaction")

    colA, colB = st.columns(2)

    with colA:
        normal_button = st.button("Predict Random Transaction")

    with colB:
        fraud_button = st.button("Predict Fraud Case (Demo)")

    if normal_button:

        random_index = np.random.randint(
            0, len(st.session_state.X_test)
        )
        sample = st.session_state.X_test.iloc[random_index:random_index+1]

        prediction = st.session_state.model.predict(sample)[0]
        probability = st.session_state.model.predict_proba(sample)[0][1]

        if prediction == 1:
            st.error(f"⚠ FRAUD DETECTED | Probability: {probability:.2%}")
        else:
            st.success(f"✅ Legitimate Transaction | Fraud Probability: {probability:.2%}")

    if fraud_button:

        fraud_cases = st.session_state.X_test[
            st.session_state.y_test == 1
        ]

        if len(fraud_cases) > 0:
            sample = fraud_cases.sample(1)

            prediction = st.session_state.model.predict(sample)[0]
            probability = st.session_state.model.predict_proba(sample)[0][1]

            st.error(f"⚠ FRAUD DETECTED | Probability: {probability:.2%}")
        else:
            st.warning("No fraud samples found in test split.")
