# 🛒 SmartCart Customer Segmentation System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://smartcart-segmentation-system-unsupervised-clusering-problem.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end unsupervised machine learning pipeline that transforms raw e-commerce telemetry into 4 actionable customer personas. Built to solve generic, high-cost marketing spend, this repository includes data preprocessing, cluster validation, a real-time KNN prediction pipeline, and an interactive Streamlit web dashboard.

---

## 🚀 Live Demo

👉 **Try the Interactive Web App:** [SmartCart Streamlit Dashboard](https://smartcart-segmentation-system-unsupervised-clusering-problem.streamlit.app/)

---

## 📌 Problem & Solution

* **The Problem:** Broad, one-size-fits-all marketing campaigns treat all 2,240 e-commerce customers identically. This results in inefficient budget spend, low campaign conversions, and high customer churn risks.
* **The Solution:** An unsupervised machine learning workflow that categorizes customers into 4 distinct, data-driven clusters—each paired with specific marketing strategies and business actions.

---

## 📊 Customer Segments Summary

Using continuous behavioral features, spending patterns, and demographic indicators, the model segments customers into 4 primary clusters:

1. **Family Shoppers:** Budget-conscious households prioritized around high deal/discount redemptions and practical purchases.
2. **Premium Couples:** High-income segment with substantial spending across luxury product lines (e.g., wines and premium goods).
3. **Digital Bargain Shoppers:** Active online visitors who respond primarily to promotional campaigns and web-exclusive discounts.
4. **High-Value VIP Singles:** High spending power, low household size, and strong responsiveness to direct premium loyalty programs.

---

## 🛠️ Architecture & Workflow

```
┌───────────────────────┐    ┌─────────────────────────┐    ┌────────────────────────┐
│  Raw Customer Data    │ ──>│  Data Preprocessing &   │ ──>│  Agglomerative         │
│  (2,240 Records)      │    │  Feature Engineering    │    │  Clustering (K=4)      │
└───────────────────────┘    └─────────────────────────┘    └───────────┬────────────┘
                                                                        │
┌───────────────────────┐    ┌─────────────────────────┐                ▼
│  Streamlit App &      │ <──│  KNN Classifier         │ <── Assign Cluster Labels
│  Interactive Visuals  │    │  (Real-Time Inference)  │     for Real-Time Prediction
└───────────────────────┘    └─────────────────────────┘
```

### 1. Preprocessing & Feature Engineering
* **Missing Value Imputation:** Handled missing `Income` values using median imputation to preserve severe right-tail skewness without introducing mean distortion.
* **Outlier Removal:** Filtered extreme anomalies ($\text{Income} > \$200\text{k}$ and $\text{Age} > 90$) identified via pairplot analysis.
* **Category Consolidation:** Grouped sparse education levels into 3 tiers; engineered `Living_with` as a binary indicator (`Couple` vs. `Single`).
* **Composite Features:** Created high-signal features including `Total_spending` (sum of 6 product spending categories), `Total_childrens`, and `customer_tenure_days`.
* **Feature Scaling:** Applied `StandardScaler` to equalize feature magnitudes before distance computation.

### 2. Unsupervised Clustering
* Evaluated optimal cluster counts ($K=4$) using the **Elbow Method (WCSS)** and **Silhouette Score** ($\approx 0.36$).
* Selected **Agglomerative Hierarchical Clustering** (Ward linkage) over K-Means due to its ability to capture non-spherical spatial distributions and produce tight, well-defined boundaries in 3D PCA space.

### 3. Real-Time Inference Pipeline
* Since Agglomerative Clustering cannot natively predict cluster labels for new, unseen data points, a **K-Nearest Neighbors (KNN)** model was trained on the generated labels.
* Hyperparameters were tuned via `GridSearchCV`.
* Pipeline artifacts (`scaler.joblib`, `classifier.joblib`, `model_columns.joblib`) were serialized for instant inference in the production application.

---

## 🧰 Tech Stack

* **Language:** Python 3.9+
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (`StandardScaler`, `AgglomerativeClustering`, `KNeighborsClassifier`, `GridSearchCV`)
* **Visualization:** `plotly`, `matplotlib`, `seaborn`
* **Model Serialization:** `joblib`
* **Web App Framework:** `streamlit`

---

## 📂 Repository Structure

```text
.
├── app.py                  # Main Streamlit web application
├── artifacts/
│   ├── scaler.joblib       # Fitted StandardScaler instance
│   ├── classifier.joblib   # Trained KNN classifier for real-time predictions
│   └── model_columns.joblib# Feature column alignment mapping
├── data/
│   └── customer_data.csv   # Raw dataset
├── notebooks/
│   └── segmentation.ipynb  # EDA, feature engineering, & model analysis
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## ⚡ Quickstart & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app locally:**
   ```bash
   streamlit run app.py
   ```

---

## 🤝 Contributing

Contributions, feedback, and issue reports are welcome! Feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
