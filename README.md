# Credit-Card-Fraud-Detection
# 💳 Credit Card Fraud Detection

# Kaggle Project Link:
https://www.kaggle.com/code/joshinkthomas/credit-card-fraud-detection/notebook

A machine learning project to detect fraudulent credit card transactions using various classification algorithms and class-imbalance handling techniques.

---

## 📌 Problem Statement

Credit card fraud poses a significant threat to banks and customers, leading to substantial financial losses. This project builds predictive models to identify fraudulent transactions from a highly imbalanced real-world dataset — a collaboration between **Worldline** and the **Machine Learning Group**.

> According to the Nilson Report, banking fraud was estimated to reach **$30 billion worldwide by 2020**, making automated fraud detection a critical necessity.

---

## 📂 Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492 (~0.172%)
- **Class Imbalance:** Severe — requires special handling before model building

**Features:**
- `V1` to `V28` — PCA-transformed anonymized features
- `Time` — Seconds elapsed since first transaction
- `Amount` — Transaction amount in USD
- `Class` — Target variable (0 = Legitimate, 1 = Fraud)

---

## 🔍 Project Workflow

### Step 1 — Data Loading & Understanding
- Loaded dataset from CSV
- Inspected shape, datatypes, and descriptive statistics

### Step 2 — Data Cleansing
- Checked for null/missing values → **None found**
- Verified consistent datatypes across all columns

### Step 3 — Exploratory Data Analysis (EDA)
- Bar plot and pie chart of class distribution
- Box plots and KDE plots of transaction amounts by class
- Time distribution of transactions (fraud peaks at night hours)
- Scatter plots of Amount vs. Time for fraud vs. non-fraud

### Step 4 — Data Preprocessing
- Feature-target split
- Stratified train-test split (70:30) using `StratifiedShuffleSplit`
- Power transformation (`PowerTransformer`) to reduce skewness

### Step 5 — Handling Class Imbalance
Three oversampling techniques were applied and compared:
- **ROS** — Random Over Sampling
- **SMOTE** — Synthetic Minority Oversampling Technique
- **ADASYN** — Adaptive Synthetic Sampling

### Step 6 — Model Building
16 models were trained and evaluated across imbalanced and balanced datasets:

| Model | Balancing Technique |
|---|---|
| Logistic Regression | Imbalanced, ROS, SMOTE, ADASYN |
| Decision Tree | ROS, SMOTE, ADASYN |
| Random Forest | ROS, SMOTE, ADASYN |
| XGBoost | ROS, SMOTE, ADASYN |

### Step 7 — Hyperparameter Tuning
- `GridSearchCV` and `RandomizedSearchCV` with `StratifiedKFold` cross-validation
- Tuned Random Forest and XGBoost models for optimal performance

---

## 📊 Evaluation Metrics

Since accuracy is misleading on imbalanced data, the following metrics were prioritized:

- **Recall** — Captures the most fraud cases (minimizes false negatives)
- **ROC-AUC Score** — Measures discriminatory power
- **Precision-Recall Curve**
- **Confusion Matrix**
- **Classification Report**

---

## 🏆 Final Model

**XGBoost on SMOTE-balanced data** was selected as the best model based on ROC-AUC and Recall performance.

Key features identified as most important for fraud detection include PCA components `V14`, `V17`, `V12`, `V10`, and `V4`.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Boosting | `xgboost` |
| Imbalance Handling | `imbalanced-learn` (SMOTE, ADASYN, ROS) |
| Statistical Modeling | `statsmodels` |
| Serialization | `pickle` |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn statsmodels
```

### Run the Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project directory (or update the path in the notebook).

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook credit-card-fraud-detection.ipynb
   ```

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── credit-card-fraud-detection.ipynb   # Main notebook
├── README.md                           # Project documentation
└── creditcard.csv                      # Dataset (download separately from Kaggle)
```

---

## 📈 Key Insights

- Plain **accuracy is not a valid metric** for highly imbalanced fraud data — a model predicting all transactions as legitimate would score ~99.8% accuracy.
- **Fraud transactions tend to occur more at night** (hours 1–8 and 24–32).
- **Fraudulent amounts** are distributed differently from legitimate ones, but amount alone is insufficient for reliable detection.
- Balancing techniques significantly improved **Recall**, enabling the model to catch more actual fraud cases.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

