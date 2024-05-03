import pickle
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

app = FastAPI()

@app.post("/predict")
async def handler(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file was provided")

    # Create a temporary file and write the uploaded file contents to it
    with NamedTemporaryFile(delete=False) as temp:
        temp.write(file.file.read())
        temp_file_path = temp.name

    try:
        # Read the uploaded Excel file from the temporary file path
        my_data = pd.read_excel(temp_file_path, engine='openpyxl')

        # Use the model to make predictions
        result = model.predict(my_data)

        col_list = list(my_data)
        ext = col_list[0:10]
        est = col_list[10:20]
        agr = col_list[20:30]
        csn = col_list[30:40]
        opn = col_list[40:50]

        my_sums = pd.DataFrame()
        my_sums['extroversion'] = my_data[ext].sum(axis=1) / 10
        my_sums['neurotic'] = my_data[est].sum(axis=1) / 10
        my_sums['agreeable'] = my_data[agr].sum(axis=1) / 10
        my_sums['conscientious'] = my_data[csn].sum(axis=1) / 10
        my_sums['open'] = my_data[opn].sum(axis=1) / 10
        my_sums['cluster'] = result

        my_sum = my_sums.drop('cluster', axis=1)

        # Convert the DataFrame to a list of dictionaries
        data = my_sum.to_dict('records')

        # Return the data as a JSON response
        return JSONResponse(content={'result': data})
    finally:
        # Remove the temporary file
        import os
        os.unlink(temp_file_path)

@app.get("/", response_class=RedirectResponse)
def redirect_to_docs():
    return "/docs"