import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from prediction import get_prediction

model = joblib.load(r'Model/Knn_model.pkl')

st.set_page_config(page_title="Iris Flower Prediction",page_icon="Flower", layout="wide")

features=['sepal.length','sepal.width','petal.length','petal.width']


st.markdown("<h1 style='text-align: center;'>Iris Flower Prediction App </h1>", unsafe_allow_html=True)

def main():
    with st.form('Prediction_form'):
        
        st.subheader("Enter the input for following features:")
        
        sepal.length = st.slider("Select sepal_length: ", min_value = 0.0, max_value = 10.0, step = 0.1, format = "%f")
        sepal.width = st.slider("Select sepal_width: ", min_value = 0.0, max_value = 10.0, step = 0.1, format = "%f")
        petal.length = st.slider("Select petal_length: ", min_value = 0.0, max_value = 10.0, step = 0.1, format = "%f")
        petal.width = st.slider("Select petal_width: ", min_value = 0.0, max_value = 10.0, step = 0.1, format = "%f")
        
        
        submit = st.form_submit_button("Predict")
    
    if submit:
        
        data = np.array([sepal.length,sepal.width,petal.length,petal.width])
        
        pred = get_prediction(data = data, model = pipe)
        
        st.write(f"The predicted Iris flower is:  {pred[0]}")
        
if __name__ == '__main__':
    main()
