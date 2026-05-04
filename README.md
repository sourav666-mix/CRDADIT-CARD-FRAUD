
readme_content = """# 🛡️ Credit Card Fraud Detection

A real-time machine learning web application built with **Streamlit** and **XGBoost** to detect fraudulent credit card transactions using PCA-transformed features.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🎯 Overview

Credit card fraud is a major concern for financial institutions. This project provides an interactive web application that uses a trained XGBoost classifier to predict whether a transaction is **legitimate** or **fraudulent** based on 28 PCA-transformed features, transaction time, and amount.

The app supports:
- 🔍 **Single transaction analysis** (manual input)
- 📂 **Batch prediction** (CSV upload)

---

## ✨ Features

- 🎨 **Dark-themed cyberpunk UI** with custom CSS
- ⚡ **Real-time predictions** with probability scores
- 📊 **Interactive probability bar charts**
- 📁 **Batch CSV processing** with downloadable results
- 📈 **Summary metrics** (total, fraud count, legitimate count)
- 💾 **Export predictions** as CSV

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | Core language |
| **Streamlit** | Web app framework |
| **XGBoost** | Machine learning model |
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation |
| **Scikit-learn** | Model training utilities |
| **Pickle** | Model serialization |

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\\Scripts\\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Place the Model File

Ensure your trained model file `fraud_xgb_model.pkl` is in the project root directory.

> **Note:** The model must be an XGBoost classifier trained on 29 features: `Time`, `V1`–`V28`.

---

## ▶️ Usage

### Run the Application

```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Manual Entry Mode

1. Navigate to the **🔍 Manual Entry** tab
2. Enter transaction details:
   - **Time**: Seconds elapsed since first transaction
   - **Amount**: Transaction amount in USD
   - **V1–V16, V18–V28**: PCA-transformed features
3. Click **🔎 Analyze Transaction**
4. View prediction result and probability breakdown

### Batch CSV Upload Mode

1. Navigate to the **📂 Batch CSV Upload** tab
2. Upload a CSV file with columns: `Time, V1, V2, ..., V28, Amount`
3. Click **🚀 Run Batch Prediction**
4. View summary metrics and download results

---

## 📊 Dataset

This project is designed for the classic **Credit Card Fraud Detection Dataset** from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- **Source**: European cardholders, September 2013
- **Transactions**: 284,807
- **Fraud cases**: 492 (0.172%)
- **Features**: 30 numerical input variables
  - `Time`: Seconds elapsed between transactions
  - `V1`–`V28`: PCA-transformed features (privacy protected)
  - `Amount`: Transaction amount
  - `Class`: Target variable (1 = fraud, 0 = legitimate)

---

## 🤖 Model

- **Algorithm**: XGBoost Classifier
- **Input features**: 29 (`Time` + `V1`–`V28`)
- **Excluded feature**: `Amount` (based on model training configuration)
- **Output**: Binary classification (0 = Legitimate, 1 = Fraud)
- **Serialization**: Pickle (`.pkl` format)

### Training Tips

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Save model
import pickle
with open('fraud_xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── app.py                     # Streamlit web application
├── fraud_xgb_model.pkl        # Trained XGBoost model (not included)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── data/                      # (Optional) Dataset folder
    └── creditcard.csv
```

---

## 📸 Screenshots

> *Add screenshots of your app here*

### Manual Entry Tab
![Manual Entry](screenshots/manual_entry.png)

### Batch Prediction Tab
![Batch Prediction](screenshots/batch_prediction.png)

---

## 🔮 Future Improvements

- [ ] Add data visualization (feature distributions, correlation heatmaps)
- [ ] Implement model explainability (SHAP values)
- [ ] Add user authentication
- [ ] Deploy to cloud (Streamlit Cloud / Heroku / AWS)
- [ ] Add model retraining pipeline
- [ ] Support for real-time API predictions
- [ ] Email/SMS alerts for high-risk transactions

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by Machine Learning Group - ULB
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [XGBoost](https://xgboost.readthedocs.io/) for the powerful gradient boosting library

---

## 📧 Contact

For questions or feedback, please reach out:

- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

<p align=\"center\">Made with ❤️ for safer transactions</p>
"""

with open('/mnt/agents/output/README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("README.md created successfully!")
