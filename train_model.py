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
import ydata_profiling

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

import pickle
import numpy as np

def calculate_model_metrics_slices(slice_list):

    with open('model.pkl', 'rb') as f_p:
        model = pickle.load(f_p)

    with open('encoder.pkl', 'rb') as f_p:
        encoder = pickle.load(f_p)

    with open('label_binarizer.pkl', 'rb') as f_p:
        lb = pickle.load(f_p)


    with open('slice_output.txt', 'w') as f_p:
        f_p.write("education, precision, recall, f_1\n")
        for group_df in slice_list:
            x_test, y_test, encoder, lb = process_data(
                group_df, categorical_features=cat_features, label="salary", training=False,
                encoder = encoder, lb=lb
            )

            preds = inference(model, x_test)
            precision, recall, f_1 = compute_model_metrics(y_test, preds)

            f_p.write(f"{group_df['education'].iloc[0]}, {precision}, {recall}, {f_1}\n")






if __name__ == "__main__":

    # Add code to load in the data.
    data = pd.read_csv("./census.csv", sep=", ", engine='python')

    # preprocess-data
    data.rename(columns = {'marital-status': 'marital_status', 
                        'education-num': 'education_num', 
                        'capital-gain': 'capital_gain', 
                        'capital-loss': 'capital_loss',
                        'hours-per-week': 'hours_per_week',
                        'native-country': 'native_country'}, inplace = True)

    cat_cols = data.select_dtypes(include='object').columns.tolist()
    num_cols = data.select_dtypes(exclude='object').columns.tolist()

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


    data = data.replace('?', np.nan)
    for col in cat_features:
        if col != 'salary':
            data[col] = data[col].fillna(data[col].mode()[0])

    for col in num_cols:
        if col != 'salary':
            data[col] = data[col].fillna(data[col].mean())

    report = data.profile_report()
    report.to_file('profile_report.html')



    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)



    x_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label='salary', training=True
    )
    # Process the test data with the process_data function.
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
    education_groups = data.groupby('education')
    education_dfs = []
    
    for education, group_df in education_groups:
        education_dfs.append(group_df)
    
    calculate_model_metrics_slices(education_dfs)
