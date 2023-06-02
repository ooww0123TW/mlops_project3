'''
main.py

Author: Wonseok Oh
Date: June 2023
'''
import logging

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

from pydantic import BaseModel
import pandas as pd
import pickle
from ml.data import process_data
from ml.model import inference

logging.basicConfig(
    filename="server.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()

class Input(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    salary: str

app = FastAPI()
model_name = "./model.pkl"
encoder_name = "./encoder.pkl"
label_binarizer_name = './label_binarizer.pkl'

with open(model_name, 'rb') as f_p:
    model = pickle.load(f_p)

with open(encoder_name, 'rb') as f_p:
    loaded_encoder = pickle.load(f_p)

with open(label_binarizer_name, 'rb') as f_p:
    label_binarizer = pickle.load(f_p)

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

@app.get("/")
async def get_welcome():
    return {"fetch": "Welcome!"}

@app.post("/data/")
async def inference_data(data: Input):
    if data.age < 0:
        raise HTTPException(status_code=400, detail="Age needs to be above 0.")

    try:
        test = pd.DataFrame(jsonable_encoder(data), index=[0])
        
        # Proces the test data with the process_data function.
        x_test, _, _, _ = process_data(
            test, categorical_features=cat_features, label='salary', training=False,
            encoder = loaded_encoder, lb=label_binarizer
        )

        preds = inference(model, x_test)
        data.salary = label_binarizer.inverse_transform(preds)[0]

        logger.info("Inference successful: {}".format(data.salary))

        return data
    
    except Exception as e:
        logger.error("An error occurred: {}".format(e))
        raise