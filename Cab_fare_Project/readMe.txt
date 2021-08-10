1) code files folder
--contains notebook file name Cab_fare_prediction_final1.ipynb having all the visualization and model codes

--contains .py file name cab_fare_prediction_deployment.py having Random Forest model selection this file will execute indepently by configuring the input paths for train and test data.
in this file will not execute plots and other models 

This file can be used for further deployement and the .pkl file which we get using that we can test future test samples.

if you want to load that .pkl file u need to use following command
#rforest_from_joblib = joblib.load('cab_fare_r_forest_model.pkl')

--Contains requirement.txt file which needs to be executed before executing the project with the following command

pip install -r /path/to/requirements.txt

--contains R file which I tried to convert from python code some part of R file will not execute because of R Tools dependency and version mismatch some of the pre-exisiting libraries are also moved from CRAN repository. 
I would like to request to evaluate on basis of python as I confirmed from Edwisor support any 1 language implementation is sufficient for submission.
===============================================================================================

2) Output file folder 

-- contains final_prediction_test.csv which is the prediction file for test data where we have predicted the answers of each test samples

--contains imputed_data2.csv which is the proof for missing value substitution of using KNN method.

===============================================================================================
3) Reports

--Contains the project report and details which we worked on Cab Fare prediction regression problem having conclusions and deployement steps 

===============================================================================================

Note: .pkl file cannot be uploaded as its size greater than allowed for submission