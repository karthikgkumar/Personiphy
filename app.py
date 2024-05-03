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

        # Return the predictions as a JSON response
        return JSONResponse(content={'result': result.tolist()})
    finally:
        # Remove the temporary file
        import os
        os.unlink(temp_file_path)

@app.get("/", response_class=RedirectResponse)
def redirect_to_docs():
    return "/docs"