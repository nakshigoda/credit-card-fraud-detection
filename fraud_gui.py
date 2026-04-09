import tkinter as tk
from tkinter import ttk
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE

# ===============================
# Global Variables
# ===============================

model = None
X_test = None
y_test = None

# ===============================
# Colors
# ===============================

BG_COLOR = "#f4f6f9"
PRIMARY = "#2563eb"
SUCCESS = "#16a34a"
DANGER = "#dc2626"
CARD = "#ffffff"

# ===============================
# Training Thread
# ===============================

def train_model():
    threading.Thread(target=train_background).start()

def train_background():
    global model, X_test, y_test

    status_label.config(text="Training model...", fg=PRIMARY)
    progress.start()

    data = pd.read_csv("creditcard.csv")

    X = data.drop("Class", axis=1)
    y = data["Class"]

    X_train, X_test_local, y_train, y_test_local = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)

    X_test = X_test_local
    y_test = y_test_local

    progress.stop()
    status_label.config(text="Model trained successfully ✅", fg=SUCCESS)

# ===============================
# Evaluation
# ===============================

def evaluate_model():
    if model is None:
        status_label.config(text="Train model first!", fg=DANGER)
        return

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    auc_label.config(text=f"ROC-AUC Score: {auc:.4f}")

def predict_random():
    if model is None:
        status_label.config(text="Train model first!", fg=DANGER)
        return

    random_index = np.random.randint(0, len(X_test))
    sample = X_test.iloc[random_index:random_index+1]
    prediction = model.predict(sample)[0]

    if prediction == 1:
        prediction_label.config(text="⚠ FRAUD DETECTED", fg=DANGER)
    else:
        prediction_label.config(text="✔ Legitimate Transaction", fg=SUCCESS)

def convert_time_to_seconds(time_str):
    try:
        hours, minutes = map(int, time_str.split(":"))
        seconds = hours * 3600 + minutes * 60
        return seconds
    except:
        return None
    
def predict_manual():
    global model

    if model is None:
        status_label.config(text="Train model first!", fg=DANGER)
        return

    try:
        time_str = time_entry.get().strip()
        amount_val = float(amount_entry.get())

        time_seconds = convert_time_to_seconds(time_str)

        if time_seconds is None:
            prediction_label.config(text="Invalid time format! Use HH:MM", fg=DANGER)
            return

        # Create feature dictionary
        feature_values = {col: 0 for col in X_test.columns}

        feature_values["Time"] = time_seconds
        feature_values["Amount"] = amount_val

        # small randomness for demo
        feature_values["V5"] = np.random.normal(0,1)
        feature_values["V12"] = np.random.normal(0,1)

        input_df = pd.DataFrame([feature_values])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # CLEAR OLD RESULT
        prediction_label.config(text="")

        if prediction == 1:
            prediction_label.config(
                text=f"⚠ FRAUD DETECTED\nFraud Probability: {probability:.2f}",
                fg=DANGER
            )
        else:
            prediction_label.config(
                text=f"✔ Legitimate Transaction\nFraud Probability: {probability:.2f}",
                fg=SUCCESS
            )

    except Exception as e:
        prediction_label.config(text="Invalid input!", fg=DANGER)
# ===============================
# Visualizations
# ===============================

def show_confusion_matrix():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

def show_roc_curve():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def show_feature_importance():
    importances = model.feature_importances_
    feature_names = X_test.columns
    feat_imp = pd.Series(importances, index=feature_names)
    feat_imp.sort_values(ascending=False).head(10).plot(kind="bar", figsize=(6,4))
    plt.title("Top 10 Features")
    plt.show()

# ===============================
# GUI Setup
# ===============================

app = tk.Tk()
app.title("AI Fraud Detection Dashboard")
app.geometry("900x600")
app.configure(bg=BG_COLOR)

title = tk.Label(app, text="AI Fraud Detection Dashboard",
                 font=("Segoe UI", 22, "bold"),
                 bg=BG_COLOR)
title.pack(pady=20)

card = tk.Frame(app, bg=CARD, bd=0)
card.pack(pady=10, ipadx=20, ipady=20)

# ===============================
# Manual Transaction Input
# ===============================

input_frame = tk.Frame(app, bg=CARD)
input_frame.pack(pady=10)

# Time Label
tk.Label(
    input_frame,
    text="Enter Transaction Time (HH:MM)",
    font=("Segoe UI", 11),
    bg=CARD
).grid(row=0, column=0, padx=10, pady=5)

time_entry = tk.Entry(input_frame, width=15)
time_entry.grid(row=0, column=1, padx=10)

# Amount Label
tk.Label(
    input_frame,
    text="Enter Transaction Amount",
    font=("Segoe UI", 11),
    bg=CARD
).grid(row=0, column=2, padx=10, pady=5)

amount_entry = tk.Entry(input_frame, width=15)
amount_entry.grid(row=0, column=3, padx=10)

# Predict Button
tk.Button(
    input_frame,
    text="Predict Transaction",
    command=predict_manual,
    font=("Segoe UI", 11),
    bg=PRIMARY,
    fg="white",
    bd=0,
    activebackground="#1e40af",
).grid(row=0, column=4, padx=15)

btn_style = {
    "font": ("Segoe UI", 11),
    "bg": PRIMARY,
    "fg": "white",
    "bd": 0,
    "activebackground": "#1e40af",
    "width": 18,
    "height": 1
}

tk.Button(card, text="Train Model", command=train_model, **btn_style).grid(row=0, column=0, padx=10, pady=10)
tk.Button(card, text="Evaluate Model", command=evaluate_model, **btn_style).grid(row=0, column=1, padx=10, pady=10)
tk.Button(card, text="Predict Transaction", command=predict_random, **btn_style).grid(row=0, column=2, padx=10, pady=10)

tk.Button(card, text="Confusion Matrix", command=show_confusion_matrix, **btn_style).grid(row=1, column=0, padx=10, pady=10)
tk.Button(card, text="ROC Curve", command=show_roc_curve, **btn_style).grid(row=1, column=1, padx=10, pady=10)
tk.Button(card, text="Feature Importance", command=show_feature_importance, **btn_style).grid(row=1, column=2, padx=10, pady=10)

progress = ttk.Progressbar(app, mode="indeterminate", length=400)
progress.pack(pady=15)

status_label = tk.Label(app, text="Model not trained",
                        font=("Segoe UI", 12),
                        bg=BG_COLOR)
status_label.pack()

auc_label = tk.Label(app, text="ROC-AUC Score: --",
                     font=("Segoe UI", 14, "bold"),
                     bg=BG_COLOR)
auc_label.pack(pady=10)

prediction_label = tk.Label(app, text="",
                            font=("Segoe UI", 18, "bold"),
                            bg=BG_COLOR)
prediction_label.pack(pady=20)

def clear_inputs():
    time_entry.delete(0, tk.END)
    amount_entry.delete(0, tk.END)
    prediction_label.config(text="")

tk.Button(
    input_frame,
    text="Clear",
    command=clear_inputs,
    font=("Segoe UI", 11),
    bg="#6b7280",
    fg="white"
).grid(row=0, column=5, padx=10)

app.mainloop()