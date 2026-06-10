# ==========================================
# CREDIT CARD FRAUD DETECTION
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

# ==========================================
# LOAD DATASET
# ==========================================

print("=" * 60)
print("Loading Dataset...")

df = pd.read_csv('C:\\Users\\Dell\\Downloads\\creditcard.csv')

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# ==========================================
# BASIC INFORMATION
# ==========================================

print("\nDataset Information:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# CLASS DISTRIBUTION
# ==========================================

print("\nTransaction Distribution:")
print(df["Class"].value_counts())

plt.figure(figsize=(6, 4))
df["Class"].value_counts().plot(
    kind="bar",
    color=["skyblue", "red"]
)
plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ==========================================
# DATA PREPROCESSING
# ==========================================

print("\nPreprocessing Data...")

# Scale Amount Column
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

# Drop Time Column
if "Time" in df.columns:
    df.drop("Time", axis=1, inplace=True)

print("Data Preprocessing Completed!")

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# HANDLE CLASS IMBALANCE USING SMOTE
# ==========================================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nClass Distribution After SMOTE:")
print(y_resampled.value_counts())

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# ==========================================
# MODEL TRAINING
# ==========================================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(10)
)

plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "credit_card_fraud_model.pkl")

print("\nModel saved successfully!")
print("File Name: credit_card_fraud_model.pkl")

# ==========================================
# PROJECT COMPLETED
# ==========================================

print("\n" + "=" * 60)
print("CREDIT CARD FRAUD DETECTION PROJECT COMPLETED")
print("=" * 60)