from pycaret.datasets import get_data
import IntegratedPipeline as IP

# # A classification example for Automatic model comparison by PyCaret
dataset = get_data('credit')
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

# Step 1: Setting the pipelines.
pipeline=IP.general_datastory_for_pycaret_pipelines
# 'types' parameter = 0 means find best Classifiction, = 1 means find best Regression.
# 'sort' parameter means Models will be compared primarily against this criterion
# 'exclude' parameter is used to block certain models
# 'n' parameter means find the best n models.
# If no custom options are required, just input dataset, type of model and the target variable will be enough, as below.
# # pipeline.pycaret_find_best_model(dataset, 0, 'default')
# Step 2: (optional) Setting parameters allows users to decide which models do not need to compare, metrics and the number of models to output.
dataset, types, target_variable, sort, exclude, n, session_id = data, 0, 'default', 'Accuracy', ['ridge','lda','gbc','ada','lightgbm','rf'], 1, 123
# For now, the pipelines could answer below questions:
# Q1: Which classification model is the best model for the dataset?
# Q2: How reliable is the best model? (Explain Accuracy and AOC of the best model.)
# Q3: Which independent variable is the most important independent variable?
# Q4: How does the most important independent variable affect the dependent variable? (Explain the SHAP figure.)
# Step 3: Run the pipelines to generate data stories. If set the 'userinput'="continue", it will shows the data stories with figures on dashboard.
pipeline.pycaret_find_best_model(dataset, types, target_variable, sort, exclude, n, session_id,userinput="continue")