import numpy as np
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd

model = load_model('Final_model')

# predict output for different inputs
def predict(model, input_df):
  predictions_df = predict_model(estimator=model, data=input_df)
  predictions = predictions_df['Label'][0]
  return predictions

# inputs can either be online or via csv
# when creating fields, it has to be in the same order as the training data

from PIL import Image
#image = Image.open('hopper.jpg')
#image_hospital = Image.open('office.jpg')
#st.image(image, use_column_width=False)
add_selectbox = st.sidebar.selectbox("How would you like to predict?",("Online", "Batch"))
st.sidebar.info("This app is created to predict if an employee will leave the company")
st.sidebar.success("https://www.pycaret.org")
#st.sidebar.image(image_hospital)
st.title("Predicting employee leaving")

if add_selectbox == "Online":
    satisfaction_level = st.number_input('satisfaction_level',min_value=0.1,max_value=1.0,value=0.1)
    last_evaluation = st.number_input('last_evaluation',min_value=0.1,max_value=1.0,value=0.1)
    number_project = st.number_input('number_project',min_value=0, max_value=50, value=5)
    time_spend_company = st.number_input('time_spend_company',min_value=1,max_value=10,value=3)
    Work_accident = st.number_input('Work_accident', min_value=0, max_value=50, value=0)
    promotion_last_5years = st.number_input('promotion_last_5years', min_value=0, max_value=50, value=0)
    salary = st.selectbox('salary', ['low', 'high','medium'])
    output=""
    input_dict={'satisfaction_level':satisfaction_level,
                'last_evaluation':last_evaluation,
                'number_project':number_project,
                'time_spend_company':time_spend_company,
                'Work_accident': Work_accident,
                'promotion_last_5years':promotion_last_5years,
                'salary' : salary,
                'average_montly_hours':0, 
                'department':0}
    input_df = pd.DataFrame([input_dict])
    if st. button("Predict"):
       output = predict(model=model, input_df=input_df)
       output = str(output)
       if output == 0:
          output_text = "leave"
       else:
          output_text = "stay"
       st.success("This employee is likely to {}".format(output_text))

if add_selectbox == "Batch":
    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
    if file_upload is not None:
       data = pd.read_csv(file_upload)
       predictions = predict_model(estimator=model, data=data)
       st.write(predictions)

