import numpy as np
import pandas as pd
import streamlit as st
import pickle

st.title('Social Add Purchased Model')

age = st.number_input('Enter The Age', 1, 100)

EstimatedSalary = st.number_input('Enter The Estimated Salary', 1, 1000000)

X = pd.DataFrame(np.array([[age, EstimatedSalary]]), columns=['Age', 'EstimatedSalary'])

# Preprocessing the data
scaler = pickle.load(open('Streamlit Apps/Social_Network_Adds/knn_scaler.pkl', 'rb'))

X = scaler.transform(X)

# Loading The Model

model = pickle.load(open('Streamlit Apps/Social_Network_Adds/knn_model.pkl', 'rb'))

button = st.button('Predict')

if button:
    prediction = model.predict(X)

    if prediction == [0]:
        st.write('Not Purchased')
    else:
        st.write('Purchased')

