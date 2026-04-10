
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
# =========================
elif page == "🤖 ML":
    st.title("🤖 Prediction")
    st.markdown("Enter client details to predict subscription:")
    model_choice = st.selectbox("Choose Model", ["XGBoost", "Logistic Regression"])
    
    # Get the feature columns from cleaned_df (excluding target 'y')
    feature_columns = [col for col in df.columns if col != 'y']
    
    st.info(f"Model expects {len(feature_columns)} features")
    
    # Create input dictionary with zeros for all columns
    input_dict = {col: 0 for col in feature_columns}
    
    # =========================
    # INPUTS
    # =========================
    
    # Numerical inputs
    if 'age' in input_dict:
        age = st.slider("Age", 18, 100, 30)
        input_dict['age'] = age
    
    if 'campaign' in input_dict:
        campaign = st.number_input("Number of Campaign Contacts", 0, 50, 1)
        input_dict['campaign'] = campaign
    
    if 'previous' in input_dict:
        previous = st.number_input("Number of Previous Contacts", 0, 20, 0)
        input_dict['previous'] = previous
    
    if 'pdays' in input_dict:
        pdays = st.number_input("Days Since Last Contact (-1 = not contacted)", -1, 500, -1)
        input_dict['pdays'] = pdays
    
    st.subheader("Economic Indicators")
    
    if 'emp.var.rate' in input_dict:
        emp_var_rate = st.slider("Employment Variation Rate", -3.0, 2.0, 0.0, 0.1)
        input_dict['emp.var.rate'] = emp_var_rate
    
    if 'cons.price.idx' in input_dict:
        cons_price = st.slider("Consumer Price Index", 90.0, 100.0, 93.0, 0.1)
        input_dict['cons.price.idx'] = cons_price
    
    if 'cons.conf.idx' in input_dict:
        cons_conf = st.slider("Consumer Confidence Index", -60.0, 0.0, -40.0, 1.0)
        input_dict['cons.conf.idx'] = cons_conf
    
    if 'euribor3m' in input_dict:
        euribor = st.slider("Euribor 3 Month Rate", 0.0, 6.0, 2.0, 0.1)
        input_dict['euribor3m'] = euribor
    
    if 'nr.employed' in input_dict:
        nr_employed = st.slider("Number of Employees", 4000.0, 5500.0, 5000.0, 50.0)
        input_dict['nr.employed'] = nr_employed
    
    st.subheader("Demographic Information")
    
    # Job
    job_cols = [col for col in feature_columns if col.startswith('job_')]
    if job_cols:
        job_options = [col.replace('job_', '') for col in job_cols]
        job = st.selectbox("Job Type", job_options)
        input_dict[f'job_{job}'] = 1
    
    # Marital Status
    marital_cols = [col for col in feature_columns if col.startswith('marital_')]
    if marital_cols:
        marital_options = [col.replace('marital_', '') for col in marital_cols]
        marital = st.selectbox("Marital Status", marital_options)
        input_dict[f'marital_{marital}'] = 1
    
    # Education
    edu_cols = [col for col in feature_columns if col.startswith('education_')]
    if edu_cols:
        edu_options = [col.replace('education_', '') for col in edu_cols]
        education = st.selectbox("Education Level", edu_options)
        input_dict[f'education_{education}'] = 1
    
    # Housing Loan
    housing_cols = [col for col in feature_columns if col.startswith('housing_')]
    if housing_cols:
        housing_options = [col.replace('housing_', '') for col in housing_cols]
        housing = st.selectbox("Housing Loan", housing_options)
        input_dict[f'housing_{housing}'] = 1
    
    # Personal Loan
    loan_cols = [col for col in feature_columns if col.startswith('loan_')]
    if loan_cols:
        loan_options = [col.replace('loan_', '') for col in loan_cols]
        loan = st.selectbox("Personal Loan", loan_options)
        input_dict[f'loan_{loan}'] = 1
    
    # Contact Type
    contact_cols = [col for col in feature_columns if col.startswith('contact_')]
    if contact_cols:
        contact_options = [col.replace('contact_', '') for col in contact_cols]
        contact = st.selectbox("Contact Type", contact_options)
        input_dict[f'contact_{contact}'] = 1
    
    # Day of Week
    day_cols = [col for col in feature_columns if col.startswith('day_of_week_')]
    if day_cols:
        day_options = [col.replace('day_of_week_', '') for col in day_cols]
        day = st.selectbox("Day of Week", day_options)
        input_dict[f'day_of_week_{day}'] = 1
    
    # Previous Outcome
    poutcome_cols = [col for col in feature_columns if col.startswith('poutcome_')]
    if poutcome_cols:
        poutcome_options = [col.replace('poutcome_', '') for col in poutcome_cols]
        poutcome = st.selectbox("Previous Campaign Outcome", poutcome_options)
        input_dict[f'poutcome_{poutcome}'] = 1
        
        # Set prev_success based on poutcome
        if 'prev_success' in input_dict:
            prev_success = 1 if poutcome == 'success' else 0
            input_dict['prev_success'] = prev_success
    
    # Month and Season
    season_cols = [col for col in feature_columns if col.startswith('season_')]
    if season_cols:
        month = st.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        if month in ['mar', 'apr', 'may']:
            season = 'spring'
        elif month in ['jun', 'jul', 'aug']:
            season = 'summer'
        elif month in ['sep', 'oct', 'nov']:
            season = 'autumn'
        else:
            season = 'winter'
        
        input_dict[f'season_{season}'] = 1
    
    # Age Group
    age_group_cols = [col for col in feature_columns if col.startswith('age_group_')]
    if age_group_cols:
        if age < 30:
            age_group = 'young'
        elif age < 50:
            age_group = 'middle_aged'
        elif age < 65:
            age_group = 'senior'
        else:
            age_group = 'retirees'
        
        if f'age_group_{age_group}' in input_dict:
            input_dict[f'age_group_{age_group}'] = 1
    
    # =========================
    # FEATURE ENGINEERING
    # =========================
    
    # Total contacts
    if 'total_contacts' in input_dict:
        total_contacts = campaign + previous
        input_dict['total_contacts'] = total_contacts
    
    # Contact efficiency
    if 'contact_efficiency' in input_dict:
        contact_efficiency = previous / (campaign + 1) if (campaign + 1) > 0 else 0
        input_dict['contact_efficiency'] = contact_efficiency
    
    # Economic stability
    if 'economic_stability' in input_dict:
        economic_stability = emp_var_rate + cons_conf - euribor
        input_dict['economic_stability'] = economic_stability
    
    # Contacted before
    if 'contacted_before' in input_dict:
        contacted_before = 1 if previous > 0 else 0
        input_dict['contacted_before'] = contacted_before
    
    # =========================
    # PREDICTION
    # =========================
    if st.button("Predict Subscription"):
        try:
            # Create DataFrame with the input data
            input_data = pd.DataFrame([input_dict])
            
            # Select the model
            if model_choice == "XGBoost":
                model = xgb_model
            else:
                model = lr_model
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(input_data)[0][1]
            else:
                prob = 0.0
            
            # Display result
            if prediction == 1:
                st.success(f"✅ Client WILL Subscribe to the product!")
                st.metric("Probability of Subscription", f"{prob:.2%}")
            else:
                st.error(f"❌ Client WILL NOT Subscribe to the product!")
                st.metric("Probability of Subscription", f"{prob:.2%}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please make sure all inputs are filled correctly.")
