from pycaret.datasets import get_data
from pycaret import classification
from pycaret import regression
import model as MD
import view as VW
import controller as CT
from jinja2 import Environment, FileSystemLoader
import numpy as np
import pandas as pd
# Loading the folder that contains the txt templates

file_loader = FileSystemLoader('templates')

# Creating a Jinja Environment

env = Environment(loader=file_loader)

automodelcompare = env.get_template('AMC1.txt')
# ## classification
#read data
# dataset = get_data('credit')
# # print(dataset.shape)
# data = dataset.sample(frac=0.95, random_state=786)
# data_unseen = dataset.drop(data.index)
# data.reset_index(inplace=True, drop=True)
# data_unseen.reset_index(inplace=True, drop=True)
# print('Data for Modeling: ' + str(data.shape))
# print('Unseen Data For Predictions: ' + str(data_unseen.shape))
#find best model
# exp_clf101 = classification.setup(data = data, target = 'default', session_id=123)
# dataset,type,target_variable,sort,exclude,n,session_id=data,0,'default','Accuracy',[],20,123
# CT.pycaret_find_best_model_con(dataset, type, target_variable, sort, exclude, n, session_id)
# best_model = classification.compare_models(n_select=20,sort='Accuracy')
# for i in range(len(best_model)):
#     print(type(best_model[i]))

# #or dt
# rf = create_model('rf')
# tuned_rf = tune_model(rf)
# print(tuned_rf)
# # plot_model(tuned_rf, plot = 'auc')

## regression
#read data
dataset = get_data('diamond', profile=True)
data = dataset.sample(frac=0.9, random_state=786)
data_unseen = dataset.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

# print('Data for Modeling: ' + str(data.shape))
# print('Unseen Data For Predictions: ' + str(data_unseen.shape))

dataset,type,target_variable,sort,exclude,n,session_id=data,1,'Price','R2',['xgboost'],3,123
exp_reg101 = regression.setup(data = data, target = 'Price', session_id=123)
model = regression.create_model('dt')
tuned_model = regression.tune_model(model)
regression.plot_model(tuned_model, plot='error',save=True)
regression.plot_model(tuned_model, plot='feature',save=True)
# regression.evaluate_model(tuned_model)
regression.interpret_model(tuned_model,save=True)
importance=pd.DataFrame({'Feature': regression.get_config('X_train').columns, 'Value' : abs(model.feature_importances_)}).sort_values(by='Value', ascending=False)
# print(importance['Feature'])
# print(importance['Value'])
# for ind in importance.index:
#     if importance['Value'][ind] == max(importance['Value']):
#         imp = importance['Feature'][ind]
# print(imp)
regression.predict_model(model)
r=regression.pull(model)
# MAE MSE RMSE R2
# print(r['MAE'])
# print(r['MSE'])
# print(r['RMSE'])
print(r['R2'][0])


# lightgbm = create_model('lightgbm')

# lgbm_params = {'num_leaves': np.arange(10,200,10),
#                         'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
#                         'learning_rate': np.arange(0.1,1,0.1)
#                         }
# tuned_lightgbm = tune_model(lightgbm, custom_grid = lgbm_params)
# print(tuned_lightgbm)
# plot_model(tuned_lightgbm)
# plot_model(tuned_lightgbm, plot = 'error')
# plot_model(tuned_lightgbm, plot='feature')
