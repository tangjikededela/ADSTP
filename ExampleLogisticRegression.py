from pandas import read_csv
import controller as CT

# # # A simple example. Just choose a model,
# input data, independent and dependent variables,
# the output will be a series of stories about fitting the data with this model.
# Step 1: Read the example dataset about diabetes, the 'label' uses 0 and 1 to represent not having diabetes and having diabetes
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
diabetes_dataset = read_csv("./data/diabetes.csv", header=None, names=col_names)
# Step 2: Choose the model (which is logistic regression here) and the independent and dependent variables, the stories will be generated.
CT.LogisticModelStats(diabetes_dataset, ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'],'label')