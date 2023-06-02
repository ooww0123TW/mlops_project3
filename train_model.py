'''
train_model.py

Author: Wonseok Oh
Date: June 2023
'''

# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

import pickle
import numpy as np

# Add code to load in the data.
data = pd.read_csv("./census.csv", sep=", ", engine='python')
data.rename(columns = {'marital-status': 'marital_status', 
                       'education-num': 'education_num', 
                       'capital-gain': 'capital_gain', 
                       'capital-loss': 'capital_loss',
                       'hours-per-week': 'hours_per_week',
                       'native-country': 'native_country'}, inplace = True)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

x_train, y_train, encoder, lb = process_data(
     train, categorical_features=cat_features, label='salary', training=True
)
# Proces the test data with the process_data function.
x_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder = encoder, lb=lb
)


# Train and save a model.
model = train_model(x_train, y_train)
y_test = lb.transform(y_test).ravel()

print(classification_report(y_test, inference(model,x_test), zero_division=1))


with open('model.pkl', 'wb') as f_p:
    pickle.dump(model, f_p)

with open('encoder.pkl', 'wb') as f_p:
    pickle.dump(encoder, f_p)

with open('label_binarizer.pkl', 'wb') as f_p:
    pickle.dump(lb, f_p)



# Test the model, and compute the model metric
preds = inference(model, x_test)
precision, recall, f_1 = compute_model_metrics(y_test, preds)

print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f_1: {f_1}")

# Save the model metric to output file "slice_output.txt"
np.savetxt('slice_output.txt', (precision, recall, f_1))
