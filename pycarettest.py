from pycaret.datasets import get_data
from pycaret.classification import *
from pycaret.regression import *
#
# ## classification
# #read data
# dataset = get_data('credit')
# # print(dataset.shape)
# data = dataset.sample(frac=0.95, random_state=786)
# data_unseen = dataset.drop(data.index)
# data.reset_index(inplace=True, drop=True)
# data_unseen.reset_index(inplace=True, drop=True)
# # print('Data for Modeling: ' + str(data.shape))
# # print('Unseen Data For Predictions: ' + str(data_unseen.shape))
# #find best model
# exp_clf101 = setup(data = data, target = 'default', session_id=123)
# # best_model = compare_models()
# # print(best_model)
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

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

exp_reg101 = setup(data = data, target = 'Price', session_id=123)

best = compare_models(exclude = ['ransac'])

lightgbm = create_model('lightgbm')
import numpy as np
lgbm_params = {'num_leaves': np.arange(10,200,10),
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'learning_rate': np.arange(0.1,1,0.1)
                        }
tuned_lightgbm = tune_model(lightgbm, custom_grid = lgbm_params)
print(tuned_lightgbm)
plot_model(tuned_lightgbm)
plot_model(tuned_lightgbm, plot = 'error')
plot_model(tuned_lightgbm, plot='feature')
