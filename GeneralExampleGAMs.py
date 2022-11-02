from pandas import read_csv
import controller as CT

# # # Example 1: A simple example.
# # Just choose a model, input data, independent and dependent variables,
# # the output will be a series of stories about fitting the data with this model.

# Step 1: Read the example dataset about red wine quality
col_names = ["citric acid", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "sulphates", "alcohol",
             "quality"]
redwine_dataset = read_csv("./data/winequalityred.csv", header=None, names=col_names)
# Step 2: Choose the model (which is linear regression here) and the independent and dependent variables,
# the stories will be generated.
pipeline = CT.general_datastory_pipeline
pipeline.GAMsFit(redwine_dataset,
                   ["citric acid", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "sulphates", "alcohol"],
                   "quality")
