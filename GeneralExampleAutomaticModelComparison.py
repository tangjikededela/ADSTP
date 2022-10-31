from pycaret.datasets import get_data
import controller as CT

## An example for Automatic model comparison by PyCaret
dataset = get_data('diamond', profile=True)
data = dataset.sample(frac=0.9, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)
# type parameter=0 means find best Classifiction, =1 means find best Regression.
# sort parameter means Models will be compared primarily against this criterion
# exclude parameter is used to block certain models
# n parameter means find the best n models.
pipeline=CT.general_datastory_for_pycaret_pipelines
dataset, type, target_variable, sort, exclude, n, session_id = data, 1, 'Price', 'R2', ['xgboost','et','rf','lightgbm','gbr'], 1, 123
pipeline.pycaret_find_best_model_con(dataset, type, target_variable, sort, exclude, n, session_id)