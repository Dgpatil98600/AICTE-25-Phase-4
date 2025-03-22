import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# Load the pipeline
with open("diabetes_pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .big-font {
        font-size: 20px !important;
        color: #2E86C1;
    }
    .header-style {
        font-size: 30px !important;
        font-weight: bold;
        color: #E74C3C;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #F4F6F6;
    }
    .stButton button {
        background-color: #28B463 !important;
        color: white !important;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #239B56 !important;
    }
    .stProgress > div > div > div {
        background-color: #2E86C1;
    }
    .stMarkdown {
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def user_input_features():
    st.sidebar.header("📋 Enter Patient Details")
    st.sidebar.markdown("**Please provide the following details:**")
    
    pregnancies = st.sidebar.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("🍬 Glucose Level", min_value=0, max_value=300, value=100)
    blood_pressure = st.sidebar.number_input("🩸 Blood Pressure", min_value=0, max_value=200, value=80)
    skin_thickness = st.sidebar.number_input("📏 Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("💉 Insulin Level", min_value=0, max_value=1000, value=80)
    bmi = st.sidebar.number_input("⚖️ BMI", min_value=0.0, max_value=100.0, value=25.0)
    diabetes_pedigree = st.sidebar.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.sidebar.number_input("👵 Age", min_value=1, max_value=120, value=30)
    
    # Create a DataFrame with the input data
    data = {
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    }
    df = pd.DataFrame(data)
    
    # Perform feature engineering (same as in p3.ipynb)
    def set_insulin(row):
        if 16 <= row["Insulin"] <= 166:
            return "Normal"
        else:
            return "Abnormal"
    
    df['NewInsulinScore'] = df.apply(set_insulin, axis=1)
    
    # Categorize BMI
    NewBMI_categories = ["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"]
    df['NewBMI'] = "Normal"  # Default value
    df.loc[df["BMI"] < 18.5, "NewBMI"] = "Underweight"
    df.loc[(df["BMI"] >= 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = "Normal"
    df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = "Overweight"
    df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = "Obesity 1"
    df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = "Obesity 2"
    df.loc[df["BMI"] > 39.9, "NewBMI"] = "Obesity 3"
    
    # Convert 'NewBMI' to categorical
    df['NewBMI'] = pd.Categorical(df['NewBMI'], categories=NewBMI_categories)
    
    # Categorize Glucose
    NewGlucose_categories = ["Low", "Normal", "Overweight", "Secret", "High"]
    df["NewGlucose"] = "Normal"  # Default value
    df.loc[df["Glucose"] <= 70, "NewGlucose"] = "Low"
    df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = "Normal"
    df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = "Overweight"
    df.loc[df["Glucose"] > 126, "NewGlucose"] = "High"
    
    # Convert 'NewGlucose' to categorical
    df['NewGlucose'] = pd.Categorical(df['NewGlucose'], categories=NewGlucose_categories)
    
    # One-hot encoding with all possible categories
    df = pd.get_dummies(df, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)
    
    # Add missing columns (if any)
    expected_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
        'NewBMI_Normal', 'NewBMI_Overweight', 'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 'NewBMI_Obesity 3',
        'NewInsulinScore_Normal', 'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret', 'NewGlucose_High'
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value 0
    
    # Ensure the columns are in the correct order
    df = df[expected_columns]
    
    return df

# Streamlit app title
st.markdown('<p class="header-style">🩺 Diabetes Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">Enter the required details in the sidebar and get the prediction result.</p>', unsafe_allow_html=True)

# Get user input and perform feature engineering
input_df = user_input_features()

# Debug: show input data as built
st.subheader("📊 Input Data")
st.dataframe(input_df.style.applymap(lambda x: 'background-color: #D6EAF8'), height=30)

if hasattr(pipeline, 'named_steps'):
    scalers = [step for step in pipeline.named_steps.items() if 'scaler' in step[0]]
    if scalers:
        features = input_df.copy()
        for name, scaler in scalers:
            features = scaler.transform(features)
else:
    st.error("Pipeline structure is different from p3.ipynb")

if st.button("🔮 Predict"):
    try:
        # Get both prediction and probability
        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]
        
        # Show detailed prediction information
        st.subheader("📈 Prediction Details")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Prediction Probabilities:**")
            st.write(f"✅ Non-Diabetic: {probabilities[0]:.3f}")
            st.write(f"⚠️ Diabetic: {probabilities[1]:.3f}")
        
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        with col2:
            st.markdown(f"**Final Prediction:**")
            st.subheader(f"🎯 {result}")
        
        # Add confidence information
        confidence = max(probabilities)
        st.progress(confidence)
        st.write(f"Confidence: {confidence:.2%}")
        
        if confidence < 0.6:
            st.warning("⚠️ Low confidence prediction. Please consult a doctor for further evaluation.")
        
        if prediction == 1:
            st.error("🚨 The model predicts that the patient may have diabetes. Please consult a doctor.")
        else:
            st.success("🎉 The model predicts that the patient is not diabetic.")
            
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.write("Pipeline Steps:", list(pipeline.named_steps.keys()))