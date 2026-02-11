import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("churn.csv")

# 2. Encode categorical columns
le = LabelEncoder()
df["ContractType"] = le.fit_transform(df["ContractType"])
df["Churn"] = le.fit_transform(df["Churn"])

# 3. Split features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
