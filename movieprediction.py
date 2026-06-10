# ==========================================
# Movie Rating Prediction - IMDb India Dataset
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(
    r"C:\Users\Dell\Downloads\IMDb Movies India.csv.csv",
    encoding='latin1'
)

print("Dataset Shape:", df.shape)

# ==========================================
# Select Required Columns
# ==========================================

columns_needed = [
    'Genre',
    'Director',
    'Actor 1',
    'Actor 2',
    'Actor 3',
    'Duration',
    'Year',
    'Votes',
    'Rating'
]

df = df[columns_needed]

# ==========================================
# Handle Missing Values
# ==========================================

df.dropna(subset=['Rating'], inplace=True)

for col in ['Genre','Director','Actor 1','Actor 2','Actor 3']:
    df[col] = df[col].fillna("Unknown")

# ==========================================
# Clean Duration
# Example: "120 min" -> 120
# ==========================================

df['Duration'] = (
    df['Duration']
    .astype(str)
    .str.replace(' min','', regex=False)
)

df['Duration'] = pd.to_numeric(
    df['Duration'],
    errors='coerce'
)

# ==========================================
# Clean Year
# Example: "(2019)" -> 2019
# ==========================================

df['Year'] = (
    df['Year']
    .astype(str)
    .str.extract(r'(\d{4})')[0]
)

df['Year'] = pd.to_numeric(
    df['Year'],
    errors='coerce'
)

# ==========================================
# Clean Votes
# Example: "1,234" -> 1234
# ==========================================

df['Votes'] = (
    df['Votes']
    .astype(str)
    .str.replace(',', '', regex=False)
)

df['Votes'] = pd.to_numeric(
    df['Votes'],
    errors='coerce'
)

# ==========================================
# Remove Remaining Missing Values
# ==========================================

df.dropna(inplace=True)

print("Dataset Shape After Cleaning:", df.shape)

# ==========================================
# Encode Categorical Columns
# ==========================================

categorical_cols = [
    'Genre',
    'Director',
    'Actor 1',
    'Actor 2',
    'Actor 3'
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ==========================================
# Features and Target
# ==========================================

X = df.drop('Rating', axis=1)
y = df['Rating']

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Train Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Complete")

# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== RESULTS ==========")
print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R² Score:", round(r2, 3))

# ==========================================
# Feature Importance
# ==========================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# ==========================================
# Plot Feature Importance
# ==========================================

plt.figure(figsize=(10,5))
plt.bar(
    importance['Feature'],
    importance['Importance']
)

plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# Actual vs Predicted
# ==========================================

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Ratings")

plt.tight_layout()
plt.show()

# ==========================================
# Save Predictions
# ==========================================

results = pd.DataFrame({
    'Actual Rating': y_test,
    'Predicted Rating': y_pred
})

results.to_csv(
    "movie_rating_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")

# ==========================================
# Sample Predictions
# ==========================================

print("\nSample Predictions:")
print(results.head(10))