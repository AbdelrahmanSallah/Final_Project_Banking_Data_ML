
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score 
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


st.set_page_config(layout='wide', page_title="Bank marketing campaign project")

# Title
html_title = """<h1 style="color:white;text-align:center;"> Banking Credit card marketing Campaign Project </h1>"""
st.markdown(html_title, unsafe_allow_html=True)

# Image
st.image('https://static.investindia.gov.in/s3fs-public/2020-02/shutterstock_400246663.jpg')


@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_df.csv")
    return df

df = load_data()

def load_models():
    xgb_model = joblib.load("xgb_model.pkl")
    lr_model = joblib.load("lr_model.pkl")
    return xgb_model, lr_model

xgb_model, lr_model = load_models()

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Analysis", "🤖 ML"])

# =========================
# 🏠 HOME PAGE
# =========================
if page == "🏠 Home":
    st.title("🏠 Bank Marketing Dataset Overview")
    
    st.markdown("""
    ### 📌 About the Dataset
    This dataset contains information about clients contacted during a marketing campaign.
    
    🎯 **Goal:** Predict whether a client will subscribe to a product (y = yes/no).
    Models used: 
    - Logistic Regression 
    - XGBoost 
    It includes:
    - Demographic features (age, job, education)
    - Financial data (loans, defaults)
    - Campaign interactions
    - Economic indicators
    """)
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    st.subheader("📘 Column Descriptions")
    column_descriptions = {
        "age": "Client age in years.",
        "campaign": "Number of contacts performed during this campaign.",
        "previous": "Number of contacts performed before this campaign.",
        "emp.var.rate": "Employment variation rate (quarterly indicator).",
        "cons.price.idx": "Consumer price index (monthly indicator).",
        "cons.conf.idx": "Consumer confidence index (monthly indicator).",
        "euribor3m": "Euribor 3-month rate (daily indicator).",
        "nr.employed": "Number of employees (quarterly indicator).",

        
        # Engineered Features
        "total_contacts": "Total number of contacts (campaign + previous).",
        "contact_efficiency": "Effectiveness of contacts (previous / campaign).",
        "economic_stability": "Combined economic indicator (emp.var.rate + cons.conf.idx - euribor3m).",

        # Binary Features
        "has_loan": "Indicates whether the client has a personal loan (1 = yes, 0 = no).",
        "prev_success": "Indicates whether previous marketing campaign was successful (1 = yes, 0 = no).",

        # Encoded / Derived Features
        "contacted_before": "Whether the client was contacted before (1 = yes, 0 = no).",

        # Categorical Encoded Features (examples)
        "job_*": "Type of job (one-hot encoded).",
        "marital_*": "Marital status (one-hot encoded).",
        "education_*": "Education level (one-hot encoded).",
        "housing_*": "Housing loan status (one-hot encoded).",
        "loan_*": "Personal loan status (one-hot encoded).",
        "day_of_week_*": "Day of the week of last contact (encoded).",
        "poutcome_*": "Outcome of previous marketing campaign (encoded).",

        # Time / Group Features
        "age_group_*": "Age category (young, middle-aged, senior, retirees).",
        "season_*": "Season when the contact happened (spring, summer, winter).",

        # Target
        "y": "Target variable: whether the client subscribed to the term deposit (1 = yes, 0 = no)."
      
        }
    desc_df = pd.DataFrame(list(column_descriptions.items()),
    columns=["Column Name", "Description"])
    st.table(desc_df)

    st.subheader("📌 Unique Values per Column")
    unique_counts = pd.DataFrame({
        'Column': df.columns,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(unique_counts)

# =========================
# 📊 ANALYSIS PAGE
# =========================
elif page == "📊 Analysis":
    st.title("📊 Exploratory Data Analysis")
    
    # Target Distribution
    st.subheader("Target Distribution")
    fig = px.histogram(df, x='y', title="Target Distribution")
    st.plotly_chart(fig)
    
    # Age vs Conversion
    if 'age_group' in df.columns:
        st.subheader("Conversion Rate by Age Group")
        age_conv = df.groupby('age_group')['y'].mean().reset_index()
        fig = px.bar(age_conv, x='age_group', y='y', title="Conversion Rate by Age Group")
        st.plotly_chart(fig)
    
    # Season
    if 'season' in df.columns:
        st.subheader("Conversion by Season")
        season_conv = df.groupby('season')['y'].mean().reset_index()
        fig = px.bar(season_conv, x='season', y='y', title="Conversion Rate by Season")
        st.plotly_chart(fig)
    
    # Campaign
    st.subheader("Campaign vs Conversion")
    camp = df.groupby('campaign')['y'].mean().reset_index()
    fig = px.line(camp, x='campaign', y='y', title="Conversion Rate by Number of Campaign Contacts")
    st.plotly_chart(fig)
    
    # Economic Stability
    if 'economic_stability' in df.columns:
        st.subheader("Economic Stability vs Conversion")
        fig = px.scatter(df, x='economic_stability', y='y', title="Economic Stability vs Conversion")
        st.plotly_chart(fig)
        fig2 = px.histogram(df, x='economic_stability', color='y', title="Economic Stability Distribution by Conversion")
        st.plotly_chart(fig2)

# =========================
# 🤖 ML PAGE
# =========================# =========================

# 🤖 ML PAGE

# =========================# 🤖 ML PAGE
# =========================

elif page == "🤖 ML":
    st.title("🤖 Prediction")
    st.markdown("Enter client details to predict subscription:")
    model_choice = st.selectbox("Choose Model", ["XGBoost", "Logistic Regression"])

    # ✅ IMPORTANT: get correct features from model (NOT df)
    feature_columns = xgb_model.get_booster().feature_names

    st.info(f"Model expects {len(feature_columns)} features")

    # Create input dictionary with zeros
    input_dict = {col: 0 for col in feature_columns}

    # =========================
    # INPUTS
    # =========================

    age = st.slider("Age", 18, 100, 30)
    campaign = st.number_input("Number of Campaign Contacts", 0, 50, 1)
    previous = st.number_input("Number of Previous Contacts", 0, 20, 0)
    pdays = st.number_input("Days Since Last Contact (-1 = not contacted)", -1, 500, -1)

    emp_var_rate = st.slider("Employment Variation Rate", -3.0, 2.0, 0.0, 0.1)
    cons_price = st.slider("Consumer Price Index", 90.0, 100.0, 93.0, 0.1)
    cons_conf = st.slider("Consumer Confidence Index", -60.0, 0.0, -40.0, 1.0)
    euribor = st.slider("Euribor 3 Month Rate", 0.0, 6.0, 2.0, 0.1)
    nr_employed = st.slider("Number of Employees", 4000.0, 5500.0, 5000.0, 50.0)

    # Fill numerical
    input_dict['campaign'] = campaign
    input_dict['previous'] = previous
    input_dict['pdays'] = pdays
    input_dict['emp.var.rate'] = emp_var_rate
    input_dict['cons.price.idx'] = cons_price
    input_dict['cons.conf.idx'] = cons_conf
    input_dict['euribor3m'] = euribor
    input_dict['nr.employed'] = nr_employed

    # =========================
    # CATEGORICAL (ENCODED)
    # =========================

    job_options = [col.replace('job_', '') for col in feature_columns if col.startswith('job_')]
    if job_options:
        job = st.selectbox("Job Type", job_options)
        input_dict[f'job_{job}'] = 1

    marital_options = [col.replace('marital_', '') for col in feature_columns if col.startswith('marital_')]
    if marital_options:
        marital = st.selectbox("Marital Status", marital_options)
        input_dict[f'marital_{marital}'] = 1

    edu_options = [col.replace('education_', '') for col in feature_columns if col.startswith('education_')]
    if edu_options:
        education = st.selectbox("Education Level", edu_options)
        input_dict[f'education_{education}'] = 1

    # =========================
    # FEATURE ENGINEERING
    # =========================

    input_dict['economic_stability'] = emp_var_rate + cons_conf - euribor
    input_dict['contacted_before'] = 1 if previous > 0 else 0
    input_dict['prev_success'] = 1  # simple default (you can improve later)

    # Age group
    if age < 30:
        group = 'young'
    elif age < 50:
        group = 'middle_aged'
    elif age < 65:
        group = 'senior'
    else:
        group = 'retirees'

    if f'age_group_{group}' in input_dict:
        input_dict[f'age_group_{group}'] = 1

    # =========================
    # PREDICTION
    # =========================
    if st.button("Predict Subscription"):
        try:
            input_data = pd.DataFrame([input_dict])

            model = xgb_model if model_choice == "XGBoost" else lr_model

            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]

            if prediction == 1:
                st.success("✅ Client WILL Subscribe!")
            
            else:
                st.error("❌ Client WILL NOT Subscribe!")
            

        except Exception as e:
            st.error(f"Error: {str(e)}")
