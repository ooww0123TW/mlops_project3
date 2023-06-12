import pandas as pd
import pytest
import numpy as np
import ml.model
from ml.data import process_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

import pickle

@pytest.fixture
def data():
    """ Simple function to generate some data from census.csv """
    data = pd.read_csv("./census.csv", sep=", ", engine='python')
    train, test = train_test_split(data, test_size=0.20)

    return train

def test_trained_model_type(data):
    '''
    Check whether the train_model function return LogisticRegression Type or not
    '''

    model = joblib.load('model.pkl')
    # encoder = joblib.load('encoder.pkl')
    # lb = joblib.load('label_binarizer.pkl')

    # x_train, y_train, _, _ = process_data(
    #     data, categorical_features=cat_features, label='salary', training=False,
    #     encoder=encoder, lb=lb
    # )

    # model = ml.model.train_model(x_train, y_train) 
    assert type(model) == LogisticRegression

def test_model_metrics_output_type(data):
    '''
    Check whether the model metric output type is ndarray
    '''

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    with open('model.pkl', 'rb') as f_p:
        model = pickle.load(f_p)

    with open('encoder.pkl', 'rb') as f_p:
        encoder = pickle.load(f_p)

    with open('label_binarizer.pkl', 'rb') as f_p:
        lb = pickle.load(f_p)


    # model = joblib.load('model.pkl')
    # encoder = joblib.load('encoder.pkl')
    # lb = joblib.load('label_binarizer.pkl')


    x_train, y_train, _, _ = process_data(
        data, categorical_features=cat_features, label='salary', training=False,
        encoder=encoder, lb=lb
    )   

    
    preds = model.predict(x_train)
    precision, recall, f1 = ml.model.compute_model_metrics(y_train, preds)
    assert type(precision) == np.float64
    assert type(recall) == np.float64
    assert type(f1) == np.float64

def test_inference_type(data):
    '''
    Check whetehr the inference output is desired type
    '''

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    with open('model.pkl', 'rb') as f_p:
        model = pickle.load(f_p)

    with open('encoder.pkl', 'rb') as f_p:
        encoder = pickle.load(f_p)

    with open('label_binarizer.pkl', 'rb') as f_p:
        lb = pickle.load(f_p)
    
    x_train, _, _, _ = process_data(
        data, categorical_features=cat_features, label='salary', training=False,
        encoder=encoder, lb=lb
    )
    
    preds = ml.model.inference(model, x_train)
    assert type(preds) == np.ndarray