# Machine Learning Final Projects

A comprehensive machine learning portfolio containing multiple classification and prediction projects with exploratory data analysis, feature engineering, model training, and deployment components.

## 📁 Project Structure

```
.
├── README.md                          # This file
├── bankmarketing.py                   # Streamlit web application for bank marketing predictions
│
├── Notebooks:
│   ├── Final_project_real_estate.ipynb    # Machine learning pipeline for bank marketing prediction
│   ├── Final_project_banking.ipynb        # Bank marketing analysis & model training
│   └── bank-additional.ipynb              # Initial bank marketing data exploration
│
├── Data:
│   ├── bank-additional-full.csv          # Complete bank marketing dataset (original)
│   ├── bank-additional.ipynb              # Additional bank marketing dataset
│   ├── cleaned_df.csv                     # Processed & cleaned dataset ready for modeling
│   └── cleaned_df/                        # Directory containing cleaned data
│
├── Models:
│   ├── xgb_model.pkl                      # Trained XGBoost classifier model
│   └── lr_model.pkl                       # Trained Logistic Regression model
│
├── Training Logs:
│   └── catboost_info/                     # CatBoost training logs & metrics
│       ├── catboost_training.json
│       ├── learn_error.tsv
│       ├── time_left.tsv
│       └── learn/
│
├── Raw Data:
│   └── DATA_TO_BE_TRAINED/
│       ├── Bank_Marketing_Dataset.csv
│       ├── Lead_Scoring.csv
│       ├── Real_Estate_Sales_2001-2020_GL.csv
│       ├── test_Y3wMUE5_7gLdaTN.csv
│       └── train_u6lujuX_CVtuZ9i.csv
│
└── Final Data:
    ├── Final_project_banking.ipynb
    ├── bank-additional-full.csv
    └── bank-additional-names.txt
```

## 🎯 Projects Overview

### 1. Bank Marketing Campaign Prediction

**Objective:** Predict whether a bank customer will subscribe to a term deposit based on their characteristics and campaign interactions.

**Dataset:** Bank Additional Marketing Dataset
- **Features:** 20+ input variables including demographic, financial, and campaign-related attributes
- **Target Variable:** `y` (yes/no - whether customer subscribed to term deposit)
- **Total Records:** 45,211 samples
- **Class Distribution:** Imbalanced (88.7% negative, 11.3% positive)

**Key Features:**
- **Demographic:** age, job, marital status, education, default status
- **Financial:** housing loan, personal loan, balance
- **Campaign:** campaign number, days since last contact, previous outcomes
- **Economic:** employment variation rate, consumer confidence index, Euribor 3-month rate

**Data Processing Pipeline:**
1. **Data Cleaning**
   - Removed duplicates
   - Handled missing values (None/unknown)
   - Identified never-contacted clients (pdays = -1)

2. **Feature Engineering**
   - Created `contacted_before` binary indicator
   - Binned age into groups: young, middle_aged, senior, retirees
   - Created seasonal features from month data
   - Combined macro-economic indicators into `economic_stability` score
   - Encoded binary categorical variables

3. **Data Preprocessing**
   - Removed low-variance features (default: >99% same class)
   - Dropped duration (unavailable at prediction time)
   - Scaled numerical features using StandardScaler
   - Encoded categorical variables with LabelEncoder

4. **Class Imbalance Handling**
   - Applied SMOTE (Synthetic Minority Over-sampling Technique)
   - Balanced training data for fair model learning

**Machine Learning Models:**

| Model | Algorithm | Performance |
|-------|-----------|-------------|
| Logistic Regression | Linear classifier with regularization | ROC-AUC score computed |
| XGBoost | Gradient boosting ensemble method | ROC-AUC score computed |

**Model Evaluation Metrics:**
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC-AUC Score
- Feature Importance Analysis

### 2. Exploratory Data Analysis (EDA)

The notebooks contain comprehensive EDA with visualizations answering:
- Which months have the highest conversion rates? (Seasonal patterns)
- Does age affect subscription likelihood? (Demographics impact)
- Does increasing contact frequency improve conversion? (Campaign effectiveness)
- How does economic stability affect client decisions? (Macro-factors)

**Visualization Tools:** Plotly Express for interactive charts

## 💻 Streamlit Application

**File:** `bankmarketing.py`

A web-based interactive dashboard for bank marketing predictions and analysis.

**Features:**
- **🏠 Home Page:** Dataset overview and statistics
- **📊 Analysis Page:** Interactive exploratory data analysis with visualizations
- **🤖 ML Page:** Model predictions and performance metrics

**Usage:**
```bash
streamlit run bankmarketing.py
```

**Functionality:**
- Load and display data in interactive tables
- Visualize target distribution
- Analyze conversion rates by:
  - Age group
  - Season/Month
  - Campaign frequency
- Make predictions using trained models
- Display model performance metrics

## 🛠️ Technologies & Libraries

**Data Processing:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing

**Machine Learning:**
- `scikit-learn` - ML algorithms, preprocessing, and metrics
- `xgboost` - XGBoost classifier
- `imblearn` - SMOTE for class imbalance handling
- `catboost` - Alternative boosting framework

**Visualization:**
- `plotly` - Interactive visualizations
- `matplotlib` - Static visualizations
- `seaborn` - Statistical data visualization

**Deployment:**
- `streamlit` - Web application framework
- `joblib` - Model serialization and loading

## 📊 Data Exploration Results

### Target Distribution
- **Subscribe (Yes):** 11.3% of customers
- **No Subscribe (No):** 88.7% of customers
- **Imbalance Ratio:** Highly imbalanced dataset

### Key Insights

1. **Seasonal Patterns:** Conversion rates vary by season
2. **Age Demographics:** Different age groups show varying subscription rates
3. **Campaign Frequency:** More contacts don't always lead to better outcomes
4. **Economic Factors:** Economic indicators influence customer decisions
5. **Previous Contact:** Customers previously contacted show different patterns

## 🚀 How to Use

### 1. Data Preparation
```python
# Data cleaning and preprocessing is handled in the notebooks
# Load cleaned data
df = pd.read_csv('cleaned_df.csv')
```

### 2. Model Training
The full model training pipeline including:
- Feature engineering
- Scaling
- SMOTE balancing
- Model fitting
- Validation

is implemented in the notebooks: `Final_project_real_estate.ipynb` and `Final_project_banking.ipynb`

### 3. Making Predictions
```python
import joblib

# Load trained models
xgb_model = joblib.load('xgb_model.pkl')
lr_model = joblib.load('lr_model.pkl')

# Make predictions
predictions = xgb_model.predict(X_test)
probabilities = xgb_model.predict_proba(X_test)
```

### 4. Web Interface
```bash
streamlit run bankmarketing.py
```
Then open browser to `http://localhost:8501`

## 📈 Model Performance

The models' performance is evaluated on test data using:
- **Confusion Matrix:** Shows true/false positives and negatives
- **Classification Report:** Provides precision, recall, and F1-scores
- **ROC-AUC Score:** Measures discrimination ability across all thresholds
- **Feature Importance:** Identifies most predictive features

## 📝 Files Description

| File | Purpose |
|------|---------|
| `bankmarketing.py` | Streamlit application for interactive dashboard |
| `Final_project_real_estate.ipynb` | Complete ML pipeline with analysis and modeling |
| `Final_project_banking.ipynb` | Banking dataset analysis and model training |
| `bank-additional.ipynb` | Initial exploratory data analysis |
| `cleaned_df.csv` | Pre-processed dataset for modeling |
| `xgb_model.pkl` | Serialized XGBoost model |
| `lr_model.pkl` | Serialized Logistic Regression model |

## 🔍 Data Dictionary

**Key Columns:**
- `age_group`: Binned age categories
- `job`: Type of employment
- `marital`: Marital status
- `education`: Education level
- `default`: Credit default status
- `housing`: Housing loan status
- `loan`: Personal loan status
- `contact`: Contact communication type
- `campaign`: Number of contacts during campaign
- `pdays`: Days since previous contact (-1 = never contacted)
- `previous`: Number of previous contacts
- `poutcome`: Previous campaign outcome
- `season`: Season derived from month
- `economic_stability`: Combined economic indicator score
- `y`: Target (1 = subscribed, 0 = did not subscribe)

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.7+
- pip or conda

### Installation
```bash
# Install required packages
pip install pandas numpy scikit-learn xgboost imblearn plotly streamlit matplotlib seaborn joblib catboost

# Or using conda
conda install -c conda-forge pandas numpy scikit-learn xgboost pandas-imblearn plotly streamlit matplotlib seaborn joblib catboost
```

### Running the Application
```bash
cd "path/to/FINAL PROJECT"
streamlit run bankmarketing.py
```

## 📌 Key Findings

1. **Class Imbalance:** Addressed using SMOTE for balanced training
2. **Feature Importance:** Economic factors and campaign history are key predictors
3. **Model Selection:** Both Logistic Regression and XGBoost provide complementary insights
4. **Seasonal Impact:** Campaign timing significantly affects conversion rates
5. **Age Factor:** Different age groups respond differently to campaigns

## 🎓 Learning Outcomes

This project demonstrates:
- Complete machine learning pipeline implementation
- Feature engineering and domain knowledge application
- Handling imbalanced datasets
- Model comparison and evaluation
- Data visualization and storytelling
- Deployment with Streamlit
- Class imbalance handling with SMOTE

## 📞 Contact & Support

For questions about these projects, refer to the individual notebook comments and code documentation.

---

**Last Updated:** April 2026  
**Status:** Completed with trained models and interactive dashboard
