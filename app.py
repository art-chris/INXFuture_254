#  importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import joblib


## ml libaries
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


import streamlit as st
import joblib
import numpy as np

# Create a title for our app
st.title('Employee Performance Prediction')
st.divider()

# Load the model
model = joblib.load("model.pkl")

# User input fields
st.write("Please enter the candidate's information as required for the performance score prediction:")

# Input fields corresponding to the columns in X_data
gender = st.selectbox("Gender", options=["Male", "Female"])
education_background = st.selectbox("Education Background", options=["Life Sciences", "Medical", "Technical Degree", "Human Resources", "Marketing", "Other"])
marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
emp_department = st.selectbox("Employee Department", options=["Research & Development", "Sales", "Human Resources"])
business_travel_frequency = st.selectbox("Business Travel Frequency", options=["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
distance_from_home = st.number_input("Distance From Home (miles)", min_value=0, max_value=50, value=10)
emp_education_level = st.number_input("Employee Education Level (1-5)", min_value=1, max_value=5, value=3)
emp_environment_satisfaction = st.number_input("Employee Environment Satisfaction (1-4)", min_value=1, max_value=4, value=3)
emp_hourly_rate = st.number_input("Employee Hourly Rate", min_value=0, max_value=100, value=30)
emp_job_involvement = st.number_input("Employee Job Involvement (1-4)", min_value=1, max_value=4, value=3)
emp_job_level = st.number_input("Employee Job Level (1-5)", min_value=1, max_value=5, value=3)
emp_job_satisfaction = st.number_input("Employee Job Satisfaction (1-4)", min_value=1, max_value=4, value=3)
num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)
overtime = st.selectbox("Overtime", options=["No", "Yes"])
emp_last_salary_hike_percent = st.number_input("Employee Last Salary Hike Percent", min_value=0, max_value=100, value=10)
emp_relationship_satisfaction = st.number_input("Employee Relationship Satisfaction (1-4)", min_value=1, max_value=4, value=3)
total_work_experience_in_years = st.number_input("Total Work Experience (Years)", min_value=0, max_value=40, value=5)
training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
emp_work_life_balance = st.number_input("Employee Work-Life Balance (1-4)", min_value=1, max_value=4, value=3)
experience_years_at_this_company = st.number_input("Experience Years at This Company", min_value=0, max_value=40, value=5)
experience_years_in_current_role = st.number_input("Experience Years in Current Role", min_value=0, max_value=40, value=3)
years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=2)
years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=15, value=2)
attrition = st.selectbox("Attrition", options=["No", "Yes"])  # Add the missing feature

# Map categorical variables to numerical values
gender_mapping = {"Male": 1, "Female": 0}
education_background_mapping = {
    "Life Sciences": 0,
    "Medical": 1,
    "Technical Degree": 2,
    "Human Resources": 3,
    "Marketing": 4,
    "Other": 5
}
marital_status_mapping = {"Single": 0, "Married": 1, "Divorced": 2}
emp_department_mapping = {
    "Research & Development": 0,
    "Sales": 1,
    "Human Resources": 2
}
business_travel_frequency_mapping = {
    "Non-Travel": 0,
    "Travel_Rarely": 1,
    "Travel_Frequently": 2
}
overtime_mapping = {"No": 0, "Yes": 1}
attrition_mapping = {"No": 0, "Yes": 1}  # Map attrition to numerical values

# Prepare input data
X = np.array([
    gender_mapping[gender],
    education_background_mapping[education_background],
    marital_status_mapping[marital_status],
    emp_department_mapping[emp_department],
    business_travel_frequency_mapping[business_travel_frequency],
    distance_from_home,
    emp_education_level,
    emp_environment_satisfaction,
    emp_hourly_rate,
    emp_job_involvement,
    emp_job_level,
    emp_job_satisfaction,
    num_companies_worked,
    overtime_mapping[overtime],
    emp_last_salary_hike_percent,
    emp_relationship_satisfaction,
    total_work_experience_in_years,
    training_times_last_year,
    emp_work_life_balance,
    experience_years_at_this_company,
    experience_years_in_current_role,
    years_since_last_promotion,
    years_with_curr_manager,
    attrition_mapping[attrition]  # Include the missing feature
]).reshape(1, -1)  # Reshape to match model input format

st.divider()

# Prediction button
predictionbutton = st.button('Predict')

if predictionbutton:
    # Make prediction
    prediction = model.predict(X)
    st.success(f"Predicted Performance Score: {prediction[0]}")  # Correct usage of st.success()
else:
    st.write("Please use the button for prediction")