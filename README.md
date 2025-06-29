## 🧠 NeuroScan – Parkinson’s Disease Detection System

NeuroScan is a machine learning-powered web application designed to predict the likelihood of Parkinson’s disease based on various biomedical voice measurements. Built with Flask and scikit-learn, this tool allows users to input specific features and receive instant predictions.

---

### 🚀 Features

- 🧪 Predict Parkinson’s disease based on 22 input voice metrics.
- 📊 Instant results with a confidence score.
- 🧠 Option to try sample inputs (Healthy / Parkinson’s) for demo/testing.
- 🎨 Stylish and responsive user interface with three pages:
  - **Home page** with introduction
  - **Form page** to collect patient features
  - **Result page** displaying diagnosis

---

### 📁 Project Structure

.
├── app.py                  # Main Flask application
├── model/
│   ├── model.pkl           # Trained ML model
│   └── scaler.pkl          # Preprocessing scaler
├── templates/
│   ├── home.html           # Intro and overview
│   ├── form.html           # Input form
│   └── result.html         # Result output
├── requirements.txt
└── README.md



---

### 📦 Installation & Setup

1. **Clone the repository**
bash
git clone https://github.com/syedsameer8323/parkinsons-detection-app.git
cd parkinsons-detection-app

2.install dependencies
bash
pip install -r requirements.txt
Run the Flask app

3.Run the Flask app
bash
python app.py
Visit in browser

4.Visit in browser
🔗 **Live Demo**:(https://parkinsons-detector-qsr2.onrender.com)



🧠 Model Info

The model is trained using voice measurement data from a Parkinson’s dataset and includes preprocessing steps using StandardScaler. The app uses 22 relevant features for prediction.


🌐 Deployment

To deploy on platforms like Render, follow these steps:

1.Push to GitHub
2.Add requirements.txt and app.py
3.Configure render.yaml or deployment settings with:

Start Command: python app.py


🙏 Acknowledgements

Dataset: UCI ML Parkinson’s Data Set
Flask, scikit-learn, and the open-source community 