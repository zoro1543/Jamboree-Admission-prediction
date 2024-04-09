import streamlit as st
import pickle
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Load the pickled model
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


# Define a function to preprocess input variables and make predictions
def predict(variables):
    input_ = np.array([variables])
    input_scaled = classifier.steps[0][1].transform(input_)
    pred = classifier.steps[1][1].predict(input_scaled)[0] 
    return pred

def main():
    st.title('Admission Probability')
    gre = st.number_input('GRE Score', min_value=0, max_value=340, step=1)
    toefl = st.number_input('TOEFL Score', min_value=0, max_value=120, step=1)
    university_rating = st.number_input('University Rating', min_value=0.0, max_value=5.0, step=0.01)
    sop = st.number_input('SOP Rating', min_value=0.0, max_value=5.0, step=0.01)
    lor = st.number_input('LOR Rating', min_value=0.0, max_value=5.0, step=0.01)
    cgpa = st.number_input('CGPA', min_value=0.0, max_value=10.0, step=0.001)
    research = st.selectbox("Have done Research?", ['No','Yes'])

    # Convert research to numerical type
    research = 1 if research == 'Yes' else 0

    # Make prediction when the button is clicked
    if st.button('Predict'):
        result = predict([gre, toefl, university_rating, sop, lor, cgpa, research])
        result = round(100*result,2)
        st.success(f'You have {result}% chance of admission.')

if __name__ == '__main__':
    main()
