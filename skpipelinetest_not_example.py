from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas
from sklearn.linear_model import LinearRegression
from pandas import read_csv # For dataframes
from sklearn.metrics import mean_absolute_error,r2_score, mean_squared_error, make_scorer
from pandas import DataFrame # For dataframes
from numpy import ravel # For matrices
import matplotlib.pyplot as plt # For plotting data
import seaborn as sns # For plotting data
from sklearn.model_selection import train_test_split # For train/test splits
from sklearn.neighbors import KNeighborsClassifier # The k-nearest neighbor classifier
from sklearn.feature_selection import VarianceThreshold # Feature selector
from sklearn.pipeline import Pipeline # For setting up pipeline
# Various pre-processing steps
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV # For optimization
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

import controller as CT

# pipe = Pipeline([
# ('select', SelectKBest(k=2)),
# ('reduce_dim', PCA()),
# ('scaler', StandardScaler()),
# ('selector', VarianceThreshold()),
# ('linear', LinearRegression())
# ])
#
# CT.skpipeline_interpretation_con((pipe))

dataset=read_csv("./data/californiahousing.csv", header=0)
dataset.dropna(inplace=True)
X,y=dataset.to_numpy()[:,:-1],dataset.to_numpy()[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
# print(X_train.shape,X_test.shape)
p1=Pipeline([
('stdscaler', StandardScaler()),
('MinMaxScaler', MinMaxScaler()),
('linear', LinearRegression())
])

p1.fit(X_train,y_train)
train_pred=p1.predict(X_train)
test_pred=p1.predict(X_test)

# print("Train error is "+str(mean_absolute_error(train_pred,y_train)))
# print("Test error is "+str(mean_absolute_error(test_pred,y_test)))
# # for linear, the score is r2, need a new r2 template
# print(p1.score(X_test, y_test))
# # for linear, coef_ give the important score, need a new template
# print(p1.named_steps['linear'].coef_)

CT.skpipeline_interpretation_con((p1))
# datatransform=pandas.DataFrame(data=dataset)
#
# for i in range(np.size(p1)-1):
#     datatransform=p1[i].fit_transform(datatransform)
#
# col_names=dataset.columns.values.tolist()
# datatransform=pandas.DataFrame(data=datatransform)
# datatransform.columns = col_names
# # print(datatransform)
# pipeline = CT.general_datastory_pipeline
# pipeline.LinearFit(datatransform,
#                    ["housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"],
#                    "median_house_value")
Xcol=["housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
ycol="median_house_value"
CT.skpipeline_questions_answer(p1,dataset,Xcol,ycol)