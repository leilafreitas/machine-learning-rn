"""
Creator: Arthur Diniz Flor Torquato Fernandes
Based on code by: Ivanovitch Silva
Date: 20 May. 2022
API testing
"""
from fastapi.testclient import TestClient
import os
import sys
import pathlib
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# a unit test that tests the status code of the root path
def test_root():
    r = client.get("/")
    assert r.status_code == 200

# a unit test that tests the status code and response 
# for an instance with a low income

def test_get_inference_dark_trap():
    music={
        "danceability": 0.578 ,
        "energy": 0.61 ,
        "key": "7",
        "loudness": -10.375 ,
        "mode": "1",
        "speechiness": 0.0314 ,
        "acousticness": 0.00665 ,
        "instrumentalness": 0.0 ,
        "liveness": 0.177 ,
        "valence": 0.247 ,
        "tempo": 160.099 ,
        "duration_ms": 159702 ,
        "time_signature": "4"
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "Dark Trap"

def test_get_inference_underground_rap():
    music={
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
        "time_signature": "4"
    }

    r = client.post("/predict", json=music)
    print("file:")
    print(r.json())
    assert r.status_code == 200
    assert r.json() == "Underground Rap"




