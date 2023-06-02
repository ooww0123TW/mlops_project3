import requests
import json

response = requests.get('https://income-predictor.onrender.com:10000')
print(response.status_code)
print(response.json())

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
    "native_country": "United-States",
    "salary": "<=50K"}

response = requests.post('https://income-predictor.onrender.com:10000/data/', json=data, auth=('user', 'pass'))
print(response.status_code)
