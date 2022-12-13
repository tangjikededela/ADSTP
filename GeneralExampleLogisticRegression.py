import pandas as pd
import numpy as np
from pandas import read_csv
import controller as CT

# # # Example 1: A simple example. Just choose a model,
# input data, independent and dependent variables,
# the output will be a series of stories about fitting the data with this model.

# Step 1: Read the example dataset about diabetes, the dependent column (target variable) should use 0 or 1 to represent not having diabetes or having diabetes
col_names = ['pregnant', 'glucose level', 'blood pressure', 'skin', 'insulin level', 'BMI', 'pedigree', 'age', 'diabetes']
diabetes_dataset = read_csv("./data/diabetes.csv", header=None, names=col_names)
# Step 2: Choose the model (which is logistic regression here) and the independent and dependent variables, the stories will be generated.
pipeline=CT.general_datastory_pipeline
pipeline.LogisticFit(diabetes_dataset, [ 'glucose level', 'blood pressure', 'insulin level', 'BMI', 'age'],'diabetes')

# # # Example 2: A more complex example.
# # Users can create dummy variables for better fitting.
# # Select the question you want the system to answer.
#
# Step 1: Read the example dataset about direct marketing campaigns (phone calls) of a Portuguese banking institution.
# which aim at figure out what kinds of client subscribed a term deposit?
bank_dataset = read_csv("./data/banking.csv", header=0)
# Step 2: Convert some variables columns that contain only finite elements into multiple columns that contain only 0 or 1.
# (For example, Marital Status Column: Married, or Unmarried, or Divorced.
# Can be converted to. Married Column: 0 or 1; Unmarried Column: 0 or 1; Divorced Column: 0 or 1)
cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
bank_dataset = read_csv("./data/banking.csv", header=0)
data_final = CT.create_dummy_variables(bank_dataset, cat_vars)
# Step 3: Setting the more readable variable names
readable_names = dict((kv.split(': ') for kv in (l.strip(' \n') for l in open('./data/readableNamesforBank.txt'))))
# Step 4: Choose the model, the independent and dependent variables,
# replace the independent and dependent variables, set questions.
pipeline = CT.general_datastory_pipeline
pipeline.LogisticFit(data_final,
                     [ 'job_blue-collar', 'education_illiterate', "poutcome_success"], 'y',
                     [readable_names.get(key) for key in
                      [ 'job_blue-collar', 'education_illiterate', "poutcome_success"]],
                     readable_names.get('y'),
                     questionset=[1, 1, 0, 1])
