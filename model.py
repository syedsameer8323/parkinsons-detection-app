# using randomforest model
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# import joblib
# import os

# # Load dataset
# df = pd.read_csv("data\parkinsons.csv")

# # Drop non-feature column
# X = df.drop(columns=["name", "status"])
# y = df["status"]  # 1 = Parkinson's, 0 = Healthy

# # Split and scale
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Train model
# model = RandomForestClassifier(class_weight="balanced",random_state=42)
# model.fit(X_train_scaled, y_train)

# # Ensure model directory exists
# os.makedirs("model", exist_ok=True)

# # Save model and scaler
# joblib.dump(model, "model/model.pkl")
# joblib.dump(scaler, "model/scaler.pkl")

# print("✅ Model training complete. Files saved in /model")

# using logisticregression model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load dataset
df = pd.read_csv("data/parkinsons.csv")

# Drop non-feature column
X = df.drop(columns=["name", "status"])
y = df["status"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression with class balance
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Logistic Regression model training complete.")
