import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from prediction import get_prediction
from sklearn.metrics import confusion_matrix,classification_report,precision_score,recall_score,f1_score
from sklearn.metrics import _dist_metrics

model = joblib.load(r'Model/Knn_model.pkl')

st.set_page_config(page_title="Iris Flower Prediction",page_icon="Flower", layout="wide")

features=['sepal_length','sepal_width','petal_length','petal_width']


st.markdown("<h1 style='text-align: center;'>Iris Flower Prediction App </h1>", unsafe_allow_html=True)

def main():
    with st.form('Prediction_form'):
        
        st.subheader("Enter the input for following features:")
        
        sepal_length = st.slider("Select sepal_length: ", min_value = 0.0, max_value = 10.0, step = 0.1, format = "%f")
        sepal_width = st.slider("Select sepal_width: ", min_value = 0.0, max_value = 10.0, step = 0.1, format = "%f")
        petal_length = st.slider("Select petal_length: ", min_value = 0.0, max_value = 10.0, step = 0.1, format = "%f")
        petal_width = st.slider("Select petal_width: ", min_value = 0.0, max_value = 10.0, step = 0.1, format = "%f")
        
        submit = st.form_submit_button("Predict")
    
    if submit:
        data = np.array([sepal_length,sepal_width,petal_length,petal_width])
        pred = get_prediction(data = data, model = model)
        st.write(f"The predicted Iris flower is:  {pred[0]}")
        
if __name__ == '__main__':
    main()
