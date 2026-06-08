import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load data from Titanic dataset
url = 'https://github.com/copilot/share/00235180-4b60-8845-b840-7407c07e48fd'

df = pd.read_csv(r"C:\Users\Dell\Downloads\Titanic-Dataset.csv")


# 2. Preprocess
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Select features and target
X = pd.get_dummies(
    df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]], 
    drop_first=True
)
y = df["Survived"]
 
# 3. Split & Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Score
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2%}")
print(f"\nFeature Importance:")
for feature, importance in sorted(
    zip(X.columns, model.feature_importances_), 
    key=lambda x: x[1], 
    reverse=True
):
    print(f"  {feature}: {importance:.4f}")
