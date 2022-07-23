
import numpy as np
import pandas as pd
import os
import wandb
import sys
import keras
from pipeline import FeatureSelector, CategoricalTransformer, NumericalTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import joblib 


artifact_model_name = "Spotify_Neural_Network/model_export:latest"
label_encoder_name = 'Spotify_Neural_Network/encoder:latest'
pipeline_name = 'Spotify_Neural_Network/pipeline:latest'

# initiate the wandb project
run = wandb.init(project="Spotify_Neural_Network",job_type="api")



schema_extra ={
    "danceability": 0.701 ,
    "energy": 0.585 ,
    "key": "5",
    "loudness": -7.612999999999999 ,
    "mode": "0",
    "speechiness": 0.132 ,
    "acousticness": 0.344 ,
    "instrumentalness": 0.0 ,
    "liveness": 0.114 ,
    "valence": 0.422 ,
    "tempo": 119.634 ,
    "duration_ms": 216294 ,
    "time_signature": "4",
}

x_train = pd.DataFrame(schema_extra,index=[0])




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
x_train_transform=pipe.transform(x_train)
x_train = pd.DataFrame(schema_extra,index=[0])

x_train_transform=pipe.transform(x_train)

# Predict test data
predict=model.predict(x_train_transform)
predict=predict>0.5
pred=le.inverse_transform(predict.flatten())[0]

print(pred)