from fastapi import FastAPI, Body
from pydantic import BaseModel
from sklearn.cluster import KMeans
import numpy as np

app = FastAPI()

class Input(BaseModel):
    EXT1: float
    EXT2: float
    EXT3: float
    EXT4: float
    EXT5: float
    EXT6: float
    EXT7: float
    EXT8: float
    EXT9: float
    EXT10: float
    EST1: float
    EST2: float
    EST3: float
    EST4: float
    EST5: float
    EST6: float
    EST7: float
    EST8: float
    EST9: float
    EST10: float
    AGR1: float
    AGR2: float
    AGR3: float
    AGR4: float
    AGR5: float
    AGR6: float
    AGR7: float
    AGR8: float
    AGR9: float
    AGR10: float
    CSN1: float
    CSN2: float
    CSN3: float
    CSN4: float
    CSN5: float
    CSN6: float
    CSN7: float
    CSN8: float
    CSN9: float
    CSN10: float
    OPN1: float
    OPN2: float
    OPN3: float
    OPN4: float
    OPN5: float
    OPN6: float
    OPN7: float
    OPN8: float
    OPN9: float
    OPN10: float

# Load or train your K-Means model here
# This is just an example initialization
kmeans = KMeans(n_clusters=5, random_state=42)

@app.post("/predict")
def predict(input_data: Input = Body(...)):
    # Convert input data to a numpy array
    data = np.array([[input_data.EXT1, input_data.EXT2, input_data.EXT3, input_data.EXT4, input_data.EXT5,
                      input_data.EXT6, input_data.EXT7, input_data.EXT8, input_data.EXT9, input_data.EXT10,
                      input_data.EST1, input_data.EST2, input_data.EST3, input_data.EST4, input_data.EST5,
                      input_data.EST6, input_data.EST7, input_data.EST8, input_data.EST9, input_data.EST10,
                      input_data.AGR1, input_data.AGR2, input_data.AGR3, input_data.AGR4, input_data.AGR5,
                      input_data.AGR6, input_data.AGR7, input_data.AGR8, input_data.AGR9, input_data.AGR10,
                      input_data.CSN1, input_data.CSN2, input_data.CSN3, input_data.CSN4, input_data.CSN5,
                      input_data.CSN6, input_data.CSN7, input_data.CSN8, input_data.CSN9, input_data.CSN10,
                      input_data.OPN1, input_data.OPN2, input_data.OPN3, input_data.OPN4, input_data.OPN5,
                      input_data.OPN6, input_data.OPN7, input_data.OPN8, input_data.OPN9, input_data.OPN10]])

    # Make predictions using the K-Means model
    predictions = kmeans.predict(data)

    # Return the predicted cluster
    return {"cluster": int(predictions[0])}