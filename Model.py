import pickle
import pandas as pd

import csv as csv

from pycaret.classification import load_model, predict_model
# Replace 'model_name.pkl' with your filename
loaded_model = load_model('excel_neuralnet_modell')
# Assuming your test data is in a pandas DataFrame named 'df_test'
test = pd.read_csv("test.csv")
# test_data = test.copy()  # Avoid modifying t    he original data


predictions = predict_model(loaded_model, data=test)
predictions=predictions.drop(columns=['date','state','store','product'])


if not predictions.empty:
    # # Create a DataFrame with the predictions
    # predictions_df = pd.DataFrame({predictions: [predictions]})  # Assuming 'Label' contains the predicted values
    
    # # Save predictions to 'sample.csv'
    predictions.to_csv('sample.csv',sep='\t',encoding='utf-8')
    
    print("Predictions saved to sample.csv.")
else:
    print("No predictions were generated.")



