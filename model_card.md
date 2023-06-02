# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Wonseok Oh created the model. It is logistic regression using the default hyperparameters in scikit-learn 1.2.2 

## Intended Use
This model should be used to predict the salary of a person based on attributes defined in census.csv
The users are prospective job hunter.

## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).
To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Metrics
The model was evaluated using F1 score. The value is 0.95

## Ethical Considerations
We risk expressing the viewpoint that the attributes in this dataset are the only ones that are predictive of someone's income, even though we know this is not the case.

