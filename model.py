# using logisticregression model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Loading dataset
df = pd.read_csv("data/parkinsons.csv")

# Droping non-feature column
X = df.drop(columns=["name", "status"])
y = df["status"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training Logistic Regression with class balance
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Saving model and scaler
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Logistic Regression model training complete.")
