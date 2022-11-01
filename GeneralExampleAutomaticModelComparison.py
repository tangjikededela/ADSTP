from pycaret.datasets import get_data
import controller as CT

# # An example for Automatic model comparison by PyCaret
dataset = get_data('diamond', profile=True)
data = dataset.sample(frac=0.9, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)
# 'types' parameter = 0 means find best Classifiction, = 1 means find best Regression.
# 'sort' parameter means Models will be compared primarily against this criterion
# 'exclude' parameter is used to block certain models
# 'n' parameter means find the best n models.
# Setting the pipelines.
pipeline=CT.general_datastory_for_pycaret_pipelines
# If no custom options are required, just use the input dataset, type of model and the target variable will be enough, as below.
# # pipeline.pycaret_find_best_model(dataset, 1, 'Price')
# Setting parameters allows users to decide which models do not need to compare, metrics and the number of models to output.
dataset, types, target_variable, sort, exclude, n, session_id = data, 1, 'Price', 'R2', ['xgboost','et','rf','lightgbm','gbr'], 1, 123
# Input the dataset and parameters to the pipelines
pipeline.pycaret_find_best_model(dataset, types, target_variable, sort, exclude, n, session_id,userinput="continue")

