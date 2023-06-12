'''
test_main.py

Author: Wonseok Oh
Date: May 2023
'''
import json

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get_welcome():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"fetch": "Welcome!"}

def test_post_malformed():
    data = {
        "age": -5,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Adm-clerical",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"}

    resp = client.post("/data/", content=json.dumps(data))

    assert resp.status_code == 400
    assert resp.json() == {"detail": "Age needs to be above 0."}

def test_post_success_1():
    data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Adm-clerical",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"}

    resp = client.post("/data/", content = json.dumps(data))
    assert resp.json() == "<=50K"
    assert resp.status_code == 200

def test_post_success_2():
    data = {
        "age": 30,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"}

    resp = client.post("/data/", content = json.dumps(data))
    assert resp.json() == ">50K"
    assert resp.status_code == 200
    