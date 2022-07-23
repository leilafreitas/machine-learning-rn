"""
Creator: Arthur Diniz Flor Torquato Fernandes
Based on code by: Ivanovitch Silva
Date: 20 May. 2022
Create API
"""
# from typing import Union
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib
import os
import wandb
import sys
import keras
from pipeline import FeatureSelector, CategoricalTransformer, NumericalTransformer

# global variables
setattr(sys.modules["__main__"], "FeatureSelector", FeatureSelector)
setattr(sys.modules["__main__"], "CategoricalTransformer", CategoricalTransformer)
setattr(sys.modules["__main__"], "NumericalTransformer", NumericalTransformer)

# name of the model artifact
artifact_model_name = "Spotify_Neural_Network/model_export:latest"
label_encoder_name = 'Spotify_Neural_Network/encoder:latest'
pipeline_name = 'Spotify_Neural_Network/pipeline:latest'

# initiate the wandb project
run = wandb.init(project="Spotify_Neural_Network",job_type="api")

# create the api
app = FastAPI()

# declare request example data using pydantic
# a person in our dataset has the following attributes
class Person(BaseModel):
    danceability:float
    energy:float
    key:object
    loudness:float
    mode:object
    speechiness:float
    acousticness:float
    instrumentalness:float
    liveness:float
    valence:float
    tempo:float
    duration_ms:int
    time_signature:object

    class Config:
        '''
        danceability        float64
        energy              float64
        key                  object
        loudness            float64
        mode                 object
        speechiness         float64
        acousticness        float64
        instrumentalness    float64
        liveness            float64
        valence             float64
        tempo               float64
        duration_ms           int64
        time_signature       object
        '''
        schema_extra = {
            "example": {
                "danceability":0.5,
                "energy":0.5,
                "key":"3",
                "loudness":-5,
                "mode":"0",
                "speechiness":0.5,
                "acousticness":0.5,
                "instrumentalness":0.5,
                "liveness":0.5,
                "valence":0.5,
                "tempo":60,
                "duration_ms":300,
                "time_signature":"1"
            }
        }

# give a greeting using GET
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <p><span style="font-size:28px"><strong>Hello World</strong></span></p>"""\
    """<p><span style="font-size:20px">In this project, we will apply the skills """\
        """acquired in the Deploying a Scalable ML Pipeline in Production course to develop """\
        """a classification model on publicly available"""\
        """<a href="http://archive.ics.uci.edu/ml/datasets/Adult"> Census Bureau data</a>.</span></p>"""

# run the model inference and use a Person data structure via POST to the API.
@app.post("/predict")
async def get_inference(person: Person):
    
    # Download inference artifact
    model_export_path = run.use_artifact(artifact_model_name).file()
    label_encoder_path = run.use_artifact(label_encoder_name).file()
    pipeline_path = run.use_artifact(pipeline_name).file()
    
    model = keras.models.load_model(model_export_path)
    le = joblib.load(label_encoder_path)
    pipe = joblib.load(pipeline_path)
    
    # Create a dataframe from the input feature
    # note that we could use pd.DataFrame.from_dict
    # but due be only one instance, it would be necessary to
    # pass the Index.
    x_train = pd.DataFrame(person.dict(),index=[0])

    x_train_transform=pipe.transform(x_train)

    # Predict test data
    predict=model.predict(x_train_transform)
    predict=predict>0.5
    pred=le.inverse_transform(predict.flatten())[0]
    
    cat_list = ['Dark Trap', 'Underground Rap']
    
    return pred
    