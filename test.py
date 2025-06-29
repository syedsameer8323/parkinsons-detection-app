import joblib
import numpy as np

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
# Expected: 🧠 Parkinson’s Detected
# yes_pd_sample = [[119.992, 157.302, 74.997, 0.00784, 0.00007,
#                   0.00370, 0.00554, 0.01110, 0.04374, 0.426,
#                   0.02182, 0.03130, 0.02971, 0.06545, 0.02211, 21.033,
#                   0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654]]

# Expected: ✔️ No Parkinson’s Detected
no_pd_sample = [[181.0, 201.0, 171.0, 0.0025, 0.00003,
                 0.0012, 0.0018, 0.0036, 0.017, 0.15,
                 0.008, 0.009, 0.010, 0.024, 0.004, 26.5,
                 0.45, 0.7, -4.1, 0.25, 1.9, 0.12]]


# sample = np.array(yes_pd_sample)

sample = np.array(no_pd_sample)

scaled = scaler.transform(sample)
pred = model.predict(scaled)
prob = model.predict_proba(scaled)

print("Predicted Class:", pred)
print("Probability of Parkinson’s (class 1):", prob[0][1])
