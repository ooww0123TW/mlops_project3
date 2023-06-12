# mlops_project3
Deploying a ML Model to Cloud Application Platform with FastAPI project in udacity mlops nanodegree course

## Model Building
By running train_model.py, you can create a machine learning model and save the model as "model.pkl", with an encoder "encoder.pkl" and a label binarizer "label_binarizer.pkl".

Also, you can get computed model metrics on slices of the data by running train_model.py. The computed model metrics precision, recall, f1 score are saved in "slice_output.txt".

## API Creation
The API is written in main.py, and its sample usage is written in "test_main.py" or "local_api.py".
Generated docs scereenshot is named as "example.png".


"test_main.py" includes one test case for each of the possible inference of the ML model (above & below 50K)

## API deployment
Screenshot that shows I enabled the continous development of my application is named as "continuous_development.png"

Screenshots at the homepage are saved as "live_get.png" and "live_post.png"