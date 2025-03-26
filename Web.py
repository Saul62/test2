import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Set page title and layout
st.set_page_config(page_title="Patient Medication Adherence Calculator", layout="wide")

# System instructions in sidebar
st.sidebar.markdown("""
# System Instructions

## About This System
This is a Patient Medication Adherence Calculator based on Random Forest algorithm that predicts treatment outcomes by analyzing patient health indicators.

## Prediction Results
The system predicts treatment outcome categories (0-2):
- Category 0: Poor Treatment Effect
- Category 1: Moderate Treatment Effect
- Category 2: Good Treatment Effect

## How to Use
1. Fill in personal health information on the main screen
2. Click the Predict button to generate predictions
3. View prediction results and feature importance analysis

## Important Notes
- Please ensure accurate personal information
- All fields are required
- Enter numbers for numerical fields
- Select from options for other fields
""")

# Create main title
st.title("Patient Medication Adherence Calculator")

# Load model and preprocessors
@st.cache_resource
def load_model_package():
    try:
        model_package = joblib.load('rfc.pkl')
        return model_package
    except Exception as e:
        st.error(f"Error loading model package: {str(e)}")
        return None

model_package = load_model_package()
if model_package is None:
    st.stop()

model = model_package['model']
scaler = model_package['scaler']
encoder = model_package['encoder']  # 确保加载编码器
continuous_cols = model_package['continuous_cols']
categorical_cols = model_package['categorical_cols']

# Input form
st.header("Please input patient's clinical indicators:")

# All inputs in one form
age = st.number_input('Age', min_value=18, max_value=85, value=50)
height = st.number_input('Height (cm)', min_value=145, max_value=185, value=165)
weight = st.number_input('Weight (kg)', min_value=40, max_value=100, value=65)
diseases = st.number_input('Number of Comorbid Diseases', min_value=0, max_value=7, value=1)
medications = st.number_input('Number of Concomitant Medications', min_value=0, max_value=5, value=1)
gender = st.selectbox('Gender', ['Female', 'Male'])
location = st.selectbox('Treatment Type', ['Outpatient', 'Inpatient'])
education = st.selectbox('Education Level', ['Primary School or Below', 'High School', 'College or Above'])
stroke = st.selectbox('History of Stroke', ['No', 'Yes'])

# Add predict button
predict_button = st.button('Predict')

if predict_button:
    # Convert data
    gender_map = {'Female': 0, 'Male': 1}
    location_map = {'Outpatient': 0, 'Inpatient': 1}
    education_map = {'Primary School or Below': 0, 'High School': 1, 'College or Above': 2}
    stroke_map = {'No': 0, 'Yes': 1}

    data = {
        'age': age,
        'height': height,
        'weight': weight,
        'number_of_comorbid_diseases': diseases,
        'number_of_concomitant_medications': medications,
        'gender': gender_map[gender],
        'location': location_map[location],
        'educational_attainment': education_map[education],
        'history_of_stroke': stroke_map[stroke]
    }
    df = pd.DataFrame(data, index=[0])

    # 分别处理连续特征和分类特征
    df_continuous = df[continuous_cols]
    df_categorical = df[categorical_cols]

    # 先进行特征转换
    df_categorical_encoded = encoder.transform(df_categorical)

    # 合并特征
    df_combined = np.concatenate([df_continuous, df_categorical_encoded], axis=1)

    # 标准化所有特征
    df_preprocessed = scaler.transform(df_combined)

    # Make prediction
    prediction = model.predict(df_preprocessed)
    prediction_proba = model.predict_proba(df_preprocessed)
    
    st.subheader("Prediction Result")
    result_map = {0: "Poor Treatment Effect", 1: "Moderate Treatment Effect", 2: "Good Treatment Effect"}
    st.write(f"Predicted Category: {result_map[prediction[0]]}")
    
    st.subheader("Prediction Probability")
    prob_df = pd.DataFrame(prediction_proba, 
                          columns=['Category 0 (Poor)', 'Category 1 (Moderate)', 'Category 2 (Good)'])
    st.write(prob_df)

    # SHAP value explanation
    st.header("Feature Importance Analysis")
    st.write("The charts below show how each factor influences the prediction:")

    @st.cache_resource
    def get_shap_explainer():
        explainer = shap.TreeExplainer(model)
        return explainer

    explainer = get_shap_explainer()
    shap_values = explainer.shap_values(df_preprocessed)

    # Plot SHAP values for the predicted class
    st.subheader("Feature Importance Ranking")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 获取特征名称
    feature_names = (continuous_cols.tolist() + 
                    [f"{col}_{val}" for col, vals in 
                     zip(categorical_cols, encoder.categories_) 
                     for val in vals])
    
    # 直接使用 summary_plot，不需要指定类别
    shap.summary_plot(shap_values, 
                     df_preprocessed,
                     feature_names=feature_names,
                     plot_type="bar",
                     show=False)
    st.pyplot(fig)
    plt.clf()