import streamlit as st
import pandas as pd
import pickle

# Function to load the trained model
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

# Function to get user input
def get_user_input():
    # Modify these inputs to match your model's feature set
    male = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.slider('Age', 18, 100, 50)
    education = st.sidebar.selectbox('Education Level', ['Some high school', 'High school/GED', 'Some college/vocational school', 'College'])
    current_smoker = st.sidebar.selectbox('Current Smoker', ['Yes', 'No'])
    cigs_per_day = st.sidebar.slider('Cigarettes Per Day', 0, 60, 10)
    bp_meds = st.sidebar.selectbox('On Blood Pressure Medication', ['Yes', 'No'])
    prevalent_stroke = st.sidebar.selectbox('Prevalent Stroke', ['Yes', 'No'])
    prevalent_hyp = st.sidebar.selectbox('Prevalent Hypertensive', ['Yes', 'No'])
    diabetes = st.sidebar.selectbox('Diabetic', ['Yes', 'No'])
    tot_chol = st.sidebar.slider('Total Cholesterol (mg/dL)', 100, 700, 250)
    sys_bp = st.sidebar.slider('Systolic Blood Pressure (mmHg)', 80, 250, 120)
    dia_bp = st.sidebar.slider('Diastolic Blood Pressure (mmHg)', 50, 150, 80)
    bmi = st.sidebar.slider('Body Mass Index', 10.0, 50.0, 25.0)
    heart_rate = st.sidebar.slider('Heart Rate (beats/minute)', 40, 150, 75)
    glucose = st.sidebar.slider('Blood Glucose Level (mg/dL)', 50, 200, 100)

    # Convert categorical inputs to binary
    user_data = {
        'male': 1 if male == 'Male' else 0,
        'age': age,
        'education': ['Some high school', 'High school/GED', 'Some college/vocational school', 'College'].index(education) + 1,
        'currentSmoker': 1 if current_smoker == 'Yes' else 0,
        'cigsPerDay': cigs_per_day,
        'BPMeds': 1 if bp_meds == 'Yes' else 0,
        'prevalentStroke': 1 if prevalent_stroke == 'Yes' else 0,
        'prevalentHyp': 1 if prevalent_hyp == 'Yes' else 0,
        'diabetes': 1 if diabetes == 'Yes' else 0,
        'totChol': tot_chol,
        'sysBP': sys_bp,
        'diaBP': dia_bp,
        'BMI': bmi,
        'heartRate': heart_rate,
        'glucose': glucose
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

# Load the model
model = load_model()

# Main function to run the app
def main():
    st.title("10-Year Risk of Coronary Heart Disease Prediction")

    # Get user input
    user_input = get_user_input()

    # Display the user input
    st.subheader('User Input:')
    st.write(user_input)

    # Run prediction if the model and user input are available
    if model and user_input is not None:
        prediction = model.predict(user_input)
        probability = model.predict_proba(user_input)

        # Display the prediction
        st.subheader('Prediction (TenYearCHD Risk):')
        if prediction[0] == 0:
            st.write("Low Risk")
        else:
            st.write("High Risk")
        
        st.subheader('Prediction Probability:')
        st.write(f"Low Risk: {probability[0][0]*100:.2f}%")
        st.write(f"High Risk: {probability[0][1]*100:.2f}%")

if __name__ == '__main__':
    main()
