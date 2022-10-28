from pandas import read_csv
import controller as CT

# # # A simple example. Just choose a model,
# input data, independent and dependent variables,
# the output will be a series of stories about fitting the data with this model.
# Step 1: Read the example dataset about red wine quality
col_names = ["citric acid","chlorides","free sulfur dioxide","total sulfur dioxide","sulphates","alcohol","quality"]
redwine_dataset = read_csv("./data/winequalityred.csv", header=None, names=col_names)
# Step 2: Choose the model (which is linear regression here) and the independent and dependent variables, the stories will be generated.
CT.LinearModelStats(redwine_dataset, ["citric acid","chlorides","free sulfur dioxide","total sulfur dioxide","sulphates","alcohol"],"quality")

# # # A more complex example. Choose a model, do the dataset cleaning before fitting it to a model.
# Input data, independent and dependent variables.
# The following are optional:
# Set more readable names for variables.
# Select the question you want the system to answer.
# Choose your overall expectations for the fit.
# Step 1: Read the example dataset about crime rate and drop some columns
features = read_csv('./data/attributes.csv', delim_whitespace=True)
dataset = read_csv('./data/communities.data', names=features['attributes']).drop(columns=['state', 'county', 'community', 'communityname', 'fold','racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'], axis=1)
# dataset = dataset.drop(columns=['state', 'county', 'community', 'communityname', 'fold','racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'], axis=1)
# Step 2: Data cleaning
dataset = CT.cleanData(dataset, 0.8)  # Clear data with a threshold of 80%
# Setting the more readable variable names
readable_names = dict((kv.split(': ') for kv in (l.strip(' \n') for l in open('./data/readableNames.txt'))))
# Step 3: Choose the model, the independent and dependent variables, replace the independent and dependent variables, set questions, and the expectation.
CT.LinearModelStats(dataset, ['pctWPubAsst','PctHousLess3BR','PctPersOwnOccup'], 'ViolentCrimesPerPop', [readable_names.get(key) for key in ['pctWPubAsst','PctHousLess3BR','PctPersOwnOccup']],readable_names.get('ViolentCrimesPerPop'), questionset=[1, 1, 1, 1], trend=1)