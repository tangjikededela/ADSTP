import numpy as np
import seaborn as sns
import heapq
import pandas as pd
import graphviz
import statistics
from pandas import DataFrame
from sklearn.tree import export_graphviz
import pydot
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.signal import argrelextrema
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from yellowbrick.regressor import ResidualsPlot
from jinja2 import Environment, FileSystemLoader
from yellowbrick.classifier import ClassBalance, ROCAUC, ClassificationReport, ClassPredictionError
from sklearn.metrics import mean_squared_error, accuracy_score
from pygam import LinearGAM, s, f, te
from scipy import stats
import scipy.signal as signal
import math
import statsmodels.api as sm
import time
from jupyter_dash import JupyterDash
from dash import Dash, html, dcc
import plotly.express as px
from dash.dependencies import Input, Output
from dash import dash_table, callback
import dash_bootstrap_components as dbc
import base64
import language_tool_python
import tkinter as tk
from GPyOpt.methods import BayesianOptimization
import pwlf
from num2words import num2words
from numpy import inf

# Loading the folder that contains the txt templates

file_loader = FileSystemLoader('templates')

# Creating a Jinja Environment

env = Environment(loader=file_loader)

# Loading the Jinja templates from the folder

get_correlation = env.get_template('getcorrelation.txt')
model_comparison = env.get_template('modelcomparison.txt')
correlation_state = env.get_template('correlation.txt')
prediction_results = env.get_template('prediction.txt')
linearSummary = env.get_template('linearSummary.txt')
linearSummary2 = env.get_template('linearSummary2.txt')
linearSummary3 = env.get_template('linearSummary3.txt')
DecisionTree1 = env.get_template('decisiontree1.txt')
DecisionTree2 = env.get_template('decisiontree2.txt')
DecisionTree3 = env.get_template('decisiontree3.txt')
logisticSummary = env.get_template('logisticSummary.txt')
logisticSummary2 = env.get_template('logisticSummary2')
gamStory = env.get_template('gamStory.txt')
GB1 = env.get_template('GB1')
GB2 = env.get_template('GB2')
GB3 = env.get_template('GB3')

register_story = env.get_template('register.txt')
risk_factor_story = env.get_template('risk_factor.txt')
reregister_story = env.get_template('reregister.txt')
remain_story = env.get_template('remain_story.txt')
enquiries_story = env.get_template('enquiries_story.txt')

# creating the global variables
models_names = ['Gradient Boosting Regressor', 'Random Forest Regressor', 'Linear Regression',
                'Decision Tree Regressor', 'GAMs']
models_results = []
g_Xcol = []
g_ycol = []
X_train = []
X_test = []
y_train = []
y_test = []
metricsData = DataFrame()
tmp_metricsData = DataFrame()


# creating the methods for the library
def cleanData(data, threshold):
    """This function takes in as input a dataset, and returns a clean dataset.

    :param data: This is the dataset that will be cleaned.
    :param treshold: This is the treshold that decides whether columns are deleted or their missing values filled.
    :return: A dataset that does not have any missing values.
    """
    data = data.replace('?', np.nan)
    data = data.loc[:, data.isnull().mean() < threshold]  # filter data
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    for i in data.columns:
        imputer = imputer.fit(data[[i]])
        data[[i]] = imputer.transform(data[[i]])
    return data


def NormalizeData(data):
    """This function takes in as input a dataset, and returns a normalized dataset with values between 0 and 1.

    :param data: This is the dataset that will be normalized.
    :return: A dataset that has normalized values.
    """
    x = data.values  # returns a numpy array
    scaler = preprocessing.MinMaxScaler()
    scaled_x = scaler.fit_transform(x)
    data = pd.DataFrame(scaled_x, columns=data.columns)
    return data


def GetCorrelation(data, Xcol, ycol):
    """This function takes in as input a dataset,the independent variables and the dependent variable, returning
    a story about the correlation between each independent variable and the dependent variable.

    :param data: This is the dataset that will be used in the analysis.
    :param Xcol: A list of independent variables.
    :param Ycol: The dependent/target variable.
    :return: A story about the correlation between Xcol and Ycol.
    """
    p_values = []
    coeff_values = []
    correlation = []
    independent_variables_number = 0
    for i in list(Xcol):
        coeff, p_value = stats.pearsonr(data[i], data[ycol])
        p_values.append(p_value)
        coeff_values.append(coeff)
        independent_variables_number += 1
        correlation.append(((data[[i, ycol]].corr())))

    for i in range(independent_variables_number):
        print(get_correlation.render(ycol=ycol, Xcol=Xcol[i], p_value=p_values[i], coeff_value=coeff_values[i]))
        plt.figure()
        sns.heatmap(correlation[i], annot=True, fmt='.2g', cmap='flare')  # graph only one correlation
        plt.show()


def FeatureSelection(data, ycol, threshold):
    """This function takes in as input a dataset,the dependent variable and the correlation treshold, returning ?

    :param data: This is the dataset that will be used in the analysis.
    :param Ycol: The dependent/target variable.
    :param treshold: This is the treshold that decides a significant correlation.
    :return:?
    """
    num_columns = data.select_dtypes(exclude='object').columns
    keep2 = []
    keep = []
    negative = []
    positive = []
    for i in list(num_columns):
        coeff, p_value = stats.pearsonr(data[i], data[ycol])
        if p_value < 0.05 and i != ycol:
            keep.append(i)
            if -1 < coeff < threshold * (-1):
                negative.append(i)
                keep2.append(i)
            elif threshold < coeff < 1:
                positive.append(i)
                keep2.append(i)
        # else :
        # del data[i]
    print(correlation_state.render(treshold=threshold, keep2=keep2, positive=positive, negative=negative))


def ModelData(data, Xcol, ycol):
    global models_names, models_results, g_Xcol, g_ycol, X_train, X_test, y_train, y_test, metricsData
    X = data[Xcol].values
    y = data[ycol].values
    g_Xcol = Xcol
    g_ycol = ycol
    r2_metrics = []
    mae_metrics = []
    rmse_metrics = []
    # Dividing the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Creating a list with RenderdModels and Metrics
    for i in models_names:
        current_index = models_names.index(i)
        models_results.append(RenderModel(i, X_train, y_train))
        mae_metrics.append(MAE(models_results[current_index], X_test, y_test))
        rmse_metrics.append(RMSE(models_results[current_index], X_test, y_test))

    # Create DataFrame for the Model Metrics
    columns = {'MeanAbsoluteError': mae_metrics, 'RMSE': rmse_metrics}
    metricsData = DataFrame(data=columns, index=models_names)
    # Plot metrics data
    metricsData.plot(kind='barh', title='Model Comparison for Predictive Analysis', colormap='Pastel2')

    # Rank models and print comparison results
    metricsData['RankRMSE'] = metricsData['RMSE'].rank(method='min')
    metricsData['RankMAE'] = metricsData['MeanAbsoluteError'].rank(method='min')
    metricsData["rank_overall"] = metricsData[["RankRMSE", "RankMAE"]].apply(tuple, axis=1).rank(ascending=True)
    metricsData.sort_values(["rank_overall"], inplace=True, ascending=False)
    print(model_comparison.render(data=metricsData, yCol=ycol))


def RenderModel(model_type, X_train, y_train):  # Renders a model based on model name and X_train and Y_train values

    if model_type == 'Gradient Boosting Regressor':
        result = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2,
                                                    learning_rate=0.1)
        result.fit(X_train, y_train)
    elif model_type == 'Random Forest Regressor':
        result = RandomForestRegressor(n_estimators=500, random_state=0)
        result.fit(X_train, y_train)
    elif model_type == 'Linear Regression':
        result = LinearRegression()
        result.fit(X_train, y_train)
    elif model_type == 'Decision Tree Regressor':
        result = DecisionTreeRegressor(random_state=0)
        result.fit(X_train, y_train)
    elif model_type == 'GAMs':
        X = X_train
        y = y_train
        lams = np.random.rand(100, X.shape[1])  # Epochs
        lams = lams * X.shape[1] - 3
        lams = np.exp(lams)
        result = LinearGAM(n_splines=X.shape[1] + 5).gridsearch(X, y, lam=lams)
    return result


def R2(renderdModel, X_test, y_test):
    y_predict = renderdModel.predict(X_test)
    r_square = metrics.r2_score(y_test, y_predict)
    return r_square


def MAE(renderdModel, X_test, y_test):
    y_predict = renderdModel.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_predict)
    return mae


def RMSE(renderdModel, X_test, y_test):
    y_predict = renderdModel.predict(X_test)
    rmse = metrics.mean_squared_error(y_test, y_predict, squared=False)
    return rmse


def display_story_dashboard():  # display comparison between models
    global models_names, models_results, X_train, y_train, X_test, y_test, metricsData
    metricsData_plot = metricsData.drop(columns=['RankRMSE', 'RankMAE', 'rank_overall'])
    fig = px.bar(metricsData_plot)
    story = model_comparison.render(data=metricsData, yCol=g_ycol)
    story_app = JupyterDash(__name__)

    story_app.layout = html.Div([dcc.Tabs([
        dcc.Tab(label='Comparison Chart', children=[
            dcc.Graph(figure=fig)
        ]),
        dcc.Tab(label='Data Story', children=[
            html.P(story)
        ]),
    ])
    ])
    story_app.run_server(mode='inline', debug=True)


def display_residual_dashboard():  # Risidual plots for models
    global models_names, models_results, X_train, y_train, X_test, y_test, metricsData
    _base64 = []
    for ind in metricsData.index:
        _base64.append(base64.b64encode(open('pictures/{}.png'.format(ind), 'rb').read()).decode('ascii'))

    residual_app = JupyterDash(__name__)

    residual_app.layout = html.Div([
        dcc.Tabs([
            dcc.Tab(label=metricsData.index[0], children=[
                html.Img(src='data:image/png;base64,{}'.format(_base64[0]))
            ]),
            dcc.Tab(label=metricsData.index[1], children=[
                html.Img(src='data:image/png;base64,{}'.format(_base64[1]))
            ]),
            dcc.Tab(label=metricsData.index[2], children=[
                html.Img(src='data:image/png;base64,{}'.format(_base64[2]))
            ]),
            dcc.Tab(label=metricsData.index[3], children=[
                html.Img(src='data:image/png;base64,{}'.format(_base64[3]))
            ]),
        ])
    ])
    residual_app.run_server(mode='inline', debug=True)


def PrintGraphs(model_type='all'):
    global models_names, models_results, X_train, y_train, X_test, y_test, metricsData
    graphs = []
    if model_type == 'all':
        for ind in metricsData.index:
            current_index = models_names.index(ind)
            ysmodel = ResidualsPlot(models_results[current_index])
            ysmodel.fit(X_train, y_train)
            ysmodel.score(X_test, y_test)
            ysmodel.show(outpath='pictures/{}.png'.format(ind), clear_figure=True)

    else:
        current_index = models_names.index(model_type)
        ysmodel = ResidualsPlot(models_results[current_index])
        ysmodel.fit(X_train, y_train)
        ysmodel.score(X_test, y_test)
        ysmodel.show(outpath='pictures/{}.png'.format(current_index), clear_figure=True)


def Predict(model_type, values):
    global models_names, models_results, g_Xcol

    if len(values) != len(g_Xcol):
        print("The number of prediction values does not corespond with the number of predictive columns:")
        print("Required number of values is " + str(len(g_Xcol)) + "you put " + str(len(values)) + "values")
    else:
        prediction_value = models_results[models_names.index(model_type)].predict([values])
        print("Predicted Value is:" + str(prediction_value))
        print(prediction_results.render(xcol=g_Xcol, ycol=g_ycol, xcol_values=values, ycol_value=prediction_value,
                                        model_name=model_type, n=len(values)))


def LinearDefaultModel(X, y, Xcol):
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = sm.OLS(y_train, X_train).fit()
    # Create DataFrame with results from model
    columns = {'coeff': model.params.values[1:], 'pvalue': model.pvalues.round(4).values[1:]}
    linearData = DataFrame(data=columns, index=Xcol)
    predicted = model.predict(X_test)
    mse = mean_squared_error(y_test, predicted)
    rmse = mse ** (1 / 2.0)
    r2 = model.rsquared

    return (columns, linearData, predicted, mse, rmse, r2)


def LinearModelStats(data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1, 1], trend=1):
    data, Xcol, ycol = variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
    X = data[Xcol].values
    y = data[ycol]

    columns, linearData, predicted, mse, rmse, r2 = LinearDefaultModel(X, y, Xcol)
    # Store results for xcol
    for ind in linearData.index:
        ax = sns.regplot(x=ind, y=ycol, data=data)
        plt.savefig('pictures/{}.png'.format(ind))
        plt.clf()
    # Create Format index with file names
    _base64 = []
    for ind in linearData.index:
        _base64.append(base64.b64encode(open('pictures/{}.png'.format(ind), 'rb').read()).decode('ascii'))
    linear_app = JupyterDash(__name__)
    listTabs = []
    i = 0
    # Add to dashbord Linear Model Statistics
    fig = px.bar(linearData)
    intro = linearSummary2.render(r2=r2, indeNum=np.size(Xcol), modelName="Linear Model", Xcol=Xcol,
                                  ycol=ycol, qs=questionset, t=trend)
    # intro = MicroLexicalization(intro)
    aim = Xcol
    aim.insert(0, ycol)
    listTabs.append(dcc.Tab(label='LinearModelStats', children=[html.P(intro),
                                                                dash_table.DataTable(data[aim].to_dict('records'),
                                                                                     [{"name": i, "id": i} for i in
                                                                                      data[aim].columns],
                                                                                     style_table={'height': '400px',
                                                                                                  'overflowY': 'auto'})]), )
    aim.remove(ycol)
    pf = ""
    nf = ""
    nss = ""
    ss = ""
    imp = ""
    i = 0
    # Add to dashbord Xcol plots and data story

    for ind in linearData.index:
        conflict = linearSummary.render(xcol=ind, ycol=ycol, coeff=linearData['coeff'][ind],
                                        p=linearData['pvalue'][ind], qs=questionset, t=trend)
        # newstory = MicroLexicalization(story)
        if linearData['coeff'][ind] == max(linearData['coeff']):
            imp = ind
        if linearData['coeff'][ind] > 0:
            pf = pf + ind + ", "
        elif linearData['coeff'][ind] < 0:
            nf = nf + ind + ", "
        if linearData['pvalue'][ind] > 0.05:
            nss = nss + ind + ", "
        else:
            ss = ss + ind + ", "
        if questionset[1] == 1 or questionset[2] == 1:
            listTabs.append(dcc.Tab(label=ind, children=[
                html.Img(src='data:image/png;base64,{}'.format(_base64[i])), html.P(conflict)
            ]))
        i = i + 1

    summary = linearSummary3.render(imp=imp, ycol=ycol, nss=nss, ss=ss, pf=pf, nf=nf, t=trend, r2=r2,
                                    qs=questionset)
    listTabs.append(dcc.Tab(label='Summary', children=[dcc.Graph(figure=fig), html.P(summary)]), )
    linear_app.layout = html.Div([dcc.Tabs(listTabs)])
    linear_app.run_server(mode='inline', debug=True)


def LogisticrDefaultModel(X, y, Xcol):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = sm.Logit(y_train, X_train).fit()
    # predictions = model.predict(X_test)
    # accuracy = accuracy_score(y_test, predictions)
    # Create DataFrame with results from model
    columns1 = {'coeff': model.params.values, 'pvalue': model.pvalues.round(4).values}
    logisticData1 = DataFrame(data=columns1, index=Xcol)
    r2 = model.prsquared
    columns2 = {'importance score': abs(model.params.values)}
    logisticData2 = DataFrame(data=columns2, index=Xcol)
    return (columns1, logisticData1, columns2, logisticData2, r2)


def LogisticModelStats(data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1, 1]):
    data, Xcol, ycol = variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
    X = data[Xcol].values
    y = data[ycol]
    columns1, logisticData1, columns2, logisticData2, r2 = LogisticrDefaultModel(X, y, Xcol)
    # Store results for xcol
    for ind in logisticData1.index:
        ax = sns.regplot(x=ind, y=ycol, data=data, logistic=True)
        plt.savefig('pictures/{}.png'.format(ind))
        plt.clf()
    # Create Format index with file names
    _base64 = []
    for ind in logisticData1.index:
        _base64.append(base64.b64encode(open('pictures/{}.png'.format(ind), 'rb').read()).decode('ascii'))
    logistic_app = JupyterDash(__name__)
    listTabs = []
    i = 0

    # Add to dashbord Model Statistics
    intro = linearSummary2.render(r2=r2, indeNum=np.size(Xcol), modelName="Logistic Model", Xcol=Xcol,
                                  ycol=ycol, qs=questionset, t=9)
    aim = Xcol
    aim.insert(0, ycol)
    # micro planning
    intro = MicroLexicalization(intro)
    listTabs.append(dcc.Tab(label='LogisticModelStats', children=[html.P(intro),
                                                                  dash_table.DataTable(data[aim].to_dict('records'),
                                                                                       [{"name": i, "id": i} for i in
                                                                                        data[aim].columns],
                                                                                       style_table={'height': '400px',
                                                                                                    'overflowY': 'auto'})]), )
    aim.remove(ycol)
    pos_eff = ""
    neg_eff = ""
    nss = ""
    ss = ""
    imp = ""
    # Add to dashbord Xcol plots and data story

    for ind in logisticData1.index:
        # conflict
        conflict = logisticSummary.render(xcol=ind, ycol=ycol,
                                          odd=abs(100 * (math.exp(logisticData1['coeff'][ind]) - 1)),
                                          coeff=logisticData1['coeff'][ind], p=logisticData1['pvalue'][ind],
                                          qs=questionset)
        conflict = MicroLexicalization(conflict)
        if logisticData1['coeff'][ind] == max(logisticData1['coeff']):
            imp = ind
        if logisticData1['coeff'][ind] > 0:
            pos_eff = pos_eff + ind + ", "
        else:
            neg_eff = neg_eff + ind + ", "
        if logisticData1['pvalue'][ind] > 0.05:
            nss = nss + ind + ", "
        else:
            ss = ss + ind + ", "
        if questionset[1] == 1 or questionset[2] == 1:
            listTabs.append(dcc.Tab(label=ind, children=[
                html.Img(src='data:image/png;base64,{}'.format(_base64[i])), html.P(conflict)
            ]))
        i = i + 1
    fig = px.bar(logisticData2)
    summary = logisticSummary2.render(pos=pos_eff, neg=neg_eff, ycol=ycol, nss=nss, ss=ss, imp=imp,
                                      r2=r2, qs=questionset)
    summary = MicroLexicalization(summary)
    listTabs.append(dcc.Tab(label='Summary', children=[dcc.Graph(figure=fig), html.P(summary), ]), )

    logistic_app.layout = html.Div([dcc.Tabs(listTabs)])
    logistic_app.run_server(mode='inline', debug=True)


def TreeExplain(model, Xcol):
    n_nodes = model.estimators_[0].tree_.node_count
    children_left = model.estimators_[0].tree_.children_left
    children_right = model.estimators_[0].tree_.children_right
    feature = model.estimators_[0].tree_.feature
    threshold = model.estimators_[0].tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    explain = ""
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    explain = explain + (
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n ".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            explain = explain + (
                "{space}node={node} is a leaf node.\n".format(
                    space=node_depth[i] * "\t", node=i
                )
            )
        else:
            explain = explain + (
                "{space}node={node} is a split node: "
                "go to node {left} if {feature} <= {threshold} "
                "else to node {right}.\n".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=Xcol[feature[i]],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )
    return (explain)


def GradientBoostingDefaultModel(X, y, Xcol, gbr_params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = ensemble.GradientBoostingRegressor(**gbr_params)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    mse = mean_squared_error(y_test, model.predict(X_test))
    rmse = mse ** (1 / 2.0)
    r2 = model.score(X_test, y_test)
    return (model, mse, rmse, r2)


def GradientBoostingModelStats(data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1],
                               gbr_params={'n_estimators': 500,
                                           'max_depth': 3,
                                           'min_samples_split': 5,
                                           'learning_rate': 0.01,
                                           'loss': 'ls'}):
    data, Xcol, ycol, r2 = variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
    X = data[Xcol].values
    y = data[ycol]
    model, mse, rmse = GradientBoostingDefaultModel(X, y, Xcol, gbr_params)
    # Store importance figure
    plt.bar(Xcol, model.feature_importances_)

    plt.title("Importance Score")
    plt.savefig('pictures/{}.png'.format("GB1"))
    plt.clf()

    _base64 = []
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("GB1"), 'rb').read()).decode('ascii'))
    # Training & Test Deviance Figure
    test_score = np.zeros((gbr_params['n_estimators'],), dtype=np.float64)
    fig = plt.figure(figsize=(8, 8))
    plt.title('Deviance')
    plt.plot(np.arange(gbr_params['n_estimators']) + 1, model.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(gbr_params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    fig.tight_layout()
    plt.savefig('pictures/{}.png'.format("GB2"))
    plt.clf()
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("GB2"), 'rb').read()).decode('ascii'))

    GB_app = JupyterDash(__name__)
    listTabs = []
    i = 0
    # Add to dashbord Model Statistics
    intro = GB1.render(Xcol=Xcol, ycol=ycol, indeNum=np.size(Xcol), r2=r2)
    # newstory = MicroLexicalization(story)
    aim = Xcol
    aim.insert(0, ycol)
    listTabs.append(dcc.Tab(label='GB Stats', children=[html.P(intro),
                                                        dash_table.DataTable(data[aim].to_dict('records'),
                                                                             [{"name": i, "id": i} for i in
                                                                              data[aim].columns],
                                                                             style_table={'height': '400px',
                                                                                          'overflowY': 'auto'})]), )
    aim.remove(ycol)

    conflict = "conflict here"
    listTabs.append(dcc.Tab(label="Training & Test Deviance", children=[
        html.Img(src='data:image/png;base64,{}'.format(_base64[1])), html.P(conflict)
    ]))
    summary = GB3.render(Xcol=Xcol)
    listTabs.append(dcc.Tab(label='Summary', children=[html.Img(src='data:image/png;base64,{}'.format(_base64[0])),
                                                       html.P(summary), ]), )

    GB_app.layout = html.Div([dcc.Tabs(listTabs)])
    GB_app.run_server(mode='inline', debug=True)


def RandomForestDefaultModel(X, y, Xcol, n_estimators, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # Limit depth of tree to 3 levels
    rf_small = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    rf_small.fit(X_train, y_train)
    # Extract the small tree
    tree_small = rf_small.estimators_[5]
    # R2
    r2 = rf_small.score(X_train, y_train)
    # Use the forest's predict method on the test data
    predictions = rf_small.predict(X_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (abs(predictions - y_test) / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = mse ** (1 / 2.0)
    importance = rf_small.feature_importances_
    columns = {'important': importance}
    DTData = DataFrame(data=columns, index=Xcol)
    return (tree_small, rf_small, DTData, r2, mse, rmse)


def RandomForestModelStats(data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1], n_estimators=10,
                           max_depth=3):
    data, Xcol, ycol = variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
    X = data[Xcol].values
    y = data[ycol]
    tree_small, rf_small, DTData, r2, mse, rmse = RandomForestDefaultModel(X, y, Xcol, n_estimators, max_depth)
    # Save the tree as a png image
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    export_graphviz(tree_small, out_file='pictures/small_tree.dot', feature_names=Xcol, rounded=True, precision=1)
    (graph,) = pydot.graph_from_dot_file('pictures/small_tree.dot')
    graph.write_png('pictures/small_tree.png', prog=['dot'])
    encoded_image = base64.b64encode(open("pictures/small_tree.png", 'rb').read()).decode('ascii')
    # Text version of the tree node

    # Explain of the tree
    explain = TreeExplain(rf_small, Xcol)
    # Importance score Figure
    imp = ""
    fig = px.bar(DTData)
    for ind in DTData.index:
        if DTData['important'][ind] == max(DTData['important']):
            imp = ind
    RF_app = JupyterDash(__name__)
    listTabs = []
    # Add to dashbord Model Statistics
    intro = DecisionTree2.render(r2=r2, qs=questionset, indeNum=np.size(Xcol), modelName="Random Forest", Xcol=Xcol,
                                 ycol=ycol, )
    # intro = MicroLexicalization(intro)
    aim = Xcol
    aim.insert(0, ycol)
    listTabs.append(dcc.Tab(label='RandomForestModelStats', children=[html.P(intro),
                                                                      dash_table.DataTable(data[aim].to_dict('records'),
                                                                                           [{"name": i, "id": i} for i
                                                                                            in
                                                                                            data[aim].columns],
                                                                                           style_table={
                                                                                               'height': '400px',
                                                                                               'overflowY': 'auto'})]), )
    aim.remove(ycol)
    conflict = explain
    listTabs.append(
        dcc.Tab(label='Tree Explanation', children=[html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
                                                    html.Pre(conflict)]), )
    summary = DecisionTree3.render(imp=imp, ycol=ycol, r2=round(r2, 3), qs=questionset, mse=mse)
    listTabs.append(dcc.Tab(label='Summary', children=[dcc.Graph(figure=fig), html.P(summary)]), )
    RF_app.layout = html.Div([dcc.Tabs(listTabs)])
    RF_app.run_server(mode='inline', debug=True)


def DecisionTreeDefaultModel(X, y, Xcol, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = DecisionTreeRegressor(random_state=0, max_depth=max_depth)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = model.score(X_train, y_train)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** (1 / 2.0)
    importance = model.feature_importances_
    columns = {'important': importance}
    DTData = DataFrame(data=columns, index=Xcol)
    return (model, r2, mse, rmse, DTData)


def DecisionTreeModelStats(data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1], max_depth=3):
    data, Xcol, ycol = variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
    X = data[Xcol].values
    y = data[ycol]
    model, r2, mse, rmse, DTData = DecisionTreeDefaultModel(X, y, Xcol, max_depth)
    # Importance score Figure
    imp = ""
    fig = px.bar(DTData)
    for ind in DTData.index:
        if DTData['important'][ind] == max(DTData['important']):
            imp = ind
    DT_app = JupyterDash(__name__)
    listTabs = []
    # Add to dashbord Model Statistics
    intro = DecisionTree2.render(r2=r2, qs=questionset, indeNum=np.size(Xcol), modelName="Decision Tree", Xcol=Xcol,
                                 ycol=ycol, )
    # intro = MicroLexicalization(intro)
    aim = Xcol
    aim.insert(0, ycol)
    listTabs.append(dcc.Tab(label='DecisionTreeModelStats', children=[html.P(intro),
                                                                      dash_table.DataTable(data[aim].to_dict('records'),
                                                                                           [{"name": i, "id": i} for i
                                                                                            in
                                                                                            data[aim].columns],
                                                                                           style_table={
                                                                                               'height': '400px',
                                                                                               'overflowY': 'auto'})]), )
    aim.remove(ycol)

    # Figure of the tree
    fig2, axes = plt.subplots()
    tree.plot_tree(model,
                   feature_names=Xcol,
                   class_names=ycol,
                   filled=True);
    fig2.savefig('pictures/{}.png'.format("DT"))
    encoded_image = base64.b64encode(open("pictures/DT.png", 'rb').read()).decode('ascii')

    # Text version of the tree node
    # feature_names_for_text = [0] * np.size(Xcol)
    # for i in range(np.size(Xcol)):
    #     feature_names_for_text[i] = Xcol[i]
    # text_representation = export_text(model, feature_names=feature_names_for_text)
    # # print(text_representation)
    # Explain of the tree
    explain = TreeExplain(model, Xcol)
    # Text need to fix here
    conflict = explain
    listTabs.append(
        dcc.Tab(label='Tree Explanation', children=[html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
                                                    html.Pre(conflict)]), )
    summary = DecisionTree3.render(imp=imp, ycol=ycol, r2=round(r2, 3), qs=questionset, mse=mse)
    listTabs.append(dcc.Tab(label='Summary', children=[dcc.Graph(figure=fig), html.P(summary)]), )
    DT_app.layout = html.Div([dcc.Tabs(listTabs)])
    DT_app.run_server(mode='inline', debug=True)


def MicroLexicalization(text):
    tool = language_tool_python.LanguageTool('en-US')
    # get the matches
    matches = tool.check(text)
    my_mistakes = []
    my_corrections = []
    start_positions = []
    end_positions = []

    for rules in matches:
        if len(rules.replacements) > 0:
            start_positions.append(rules.offset)
            end_positions.append(rules.errorLength + rules.offset)
            my_mistakes.append(text[rules.offset:rules.errorLength + rules.offset])
            my_corrections.append(rules.replacements[0])

    my_new_text = list(text)

    for m in range(len(start_positions)):
        for i in range(len(text)):
            my_new_text[start_positions[m]] = my_corrections[m]
            if (i > start_positions[m] and i < end_positions[m]):
                my_new_text[i] = ""
    my_new_text = "".join(my_new_text)
    return (my_new_text)


def variablenamechange(dataset, Xcol, ycol, Xnewname, ynewname):
    if Xnewname != "":
        if np.size(Xnewname) != np.size(Xcol):
            raise Exception(
                "The column name of the replacement X is inconsistent with the size of the column name of the original data X.")
        for i in range(np.size(Xnewname)):
            if (Xnewname[i] != ''):
                dataset.rename(columns={Xcol[i]: Xnewname[i]}, inplace=True)
            else:
                Xnewname[i] = Xcol[i]
    elif type(Xcol) == str and Xnewname == "":
        Xnewname = Xcol
    if (ynewname != ''):
        dataset.rename(columns={ycol: ynewname}, inplace=True)
    else:
        ynewname = ycol
    return (dataset, Xnewname, ynewname)


def register_question1(register_dataset, per1000inCity_col, per1000nation_col,
                       table_col=['Period', 'Registrations In Aberdeen City',
                                  'Registrations per 1000 population in Aberdeen City',
                                  'Compared with last year for Aberdeen City']):
    question1_2_app = JupyterDash(__name__)
    listTabs = []
    registerstory = "The data from local comparators features in the Child Protection Register (CPR) report prepared quarterly. "
    diff=[0]*np.size(per1000inCity_col)
    i=0
    for ind in per1000inCity_col:
        diff[i] = (statistics.mean(register_dataset[ind]) - statistics.mean(register_dataset[per1000nation_col]))
        i=i+1
    i=0
    for ind in per1000inCity_col:
        reslut = register_story.render(Xcol=ind, minX=min(register_dataset[ind]), maxX=max(register_dataset[ind]),
                                       diff=diff[i])
        registerstory = registerstory + reslut
        i=i+1

    listTabs.append(dcc.Tab(label='What are the emerging trends or themes emerging from local and comparators data?',
                            children=[html.P(registerstory),
                                      dash_table.DataTable(register_dataset[table_col].to_dict('records'),
                                                           [{"name": i, "id": i} for i in
                                                            register_dataset[table_col].columns],
                                                           style_table={'height': '400px',
                                                                        'overflowY': 'auto'})]), )
    question1_2_app.layout = html.Div([dcc.Tabs(listTabs)])
    question1_2_app.run_server(mode='inline', debug=True)

    print(registerstory)


def riskfactor_question1(risk_factor_dataset, risk_factor_col, cityname="Aberdeen City", max_num=5):
    question1_3_app = JupyterDash(__name__)
    listTabs = []
    new_data = risk_factor_dataset[risk_factor_col].values[0:1][0]
    data_lastyear = risk_factor_dataset[risk_factor_col].values[0:2][1]
    max_data = (heapq.nlargest(max_num, new_data))
    max_data_lastyear = (heapq.nlargest(max_num, data_lastyear))
    max_factor = []
    max_factor_lastyear = []
    same_factor = 0
    for ind in risk_factor_col:
        if risk_factor_dataset[ind].values[0:1][0] in max_data:
            max_factor.append(ind)
    for ind in risk_factor_col:
        if risk_factor_dataset[ind].values[0:2][1] in max_data_lastyear:
            max_factor_lastyear.append(ind)
    for ind in max_factor:
        if ind in max_factor_lastyear:
            same_factor = same_factor + 1
    riskstory = risk_factor_story.render(indeNum=(np.size(max_factor)), max_factor=max_factor,
                                         same_factor=same_factor,
                                         cityname=cityname)
    listTabs.append(dcc.Tab(label='What are the emerging trends or themes emerging from local single agency data?',
                            children=[html.P(riskstory),
                                      dash_table.DataTable(risk_factor_dataset.to_dict('records'),
                                                           [{"name": i, "id": i} for i in
                                                            risk_factor_dataset.columns],
                                                           style_table={'height': '400px',
                                                                        'overflowY': 'auto'})]), )
    question1_3_app.layout = html.Div([dcc.Tabs(listTabs)])
    question1_3_app.run_server(mode='inline', debug=True)
    print(riskstory)


def re_register_question4(register_dataset, reregister_col, period_col='Period',
                          national_average_reregistration='13 - 16%',
                          table_col=['Period', 'Re-Registrations In Aberdeen City',
                                     'Re-registrations as a % of registrations in Aberdeen City',
                                     'Largest family for Aberdeen City',
                                     'Longest gap between registrations of Aberdeen City',
                                     'Shortest gap between registrations of Aberdeen City']):
    question4_app = JupyterDash(__name__)
    listTabs = []
    datasize = np.size(register_dataset[reregister_col])
    period = register_dataset[period_col].values[0:datasize][datasize - 1]
    reregister_lastyear = register_dataset[reregister_col].values[0:datasize][datasize - 1]
    # national_average_reregistration = '13 - 16%'  # I did not find where this data come from, it can be auto by given the data.
    reregisterstory = reregister_story.render(nar=national_average_reregistration, rrly=reregister_lastyear,
                                              time=period)
    listTabs.append(dcc.Tab(
        label='To what extent is Aberdeen City consistent with the national and comparator averages for re-registration?  Can the CPC be assured that deregistered children receive at least 3 months’ post registration multi-agency support?',
        children=[html.P(reregisterstory),
                  dash_table.DataTable(register_dataset[table_col].to_dict('records'),
                                       [{"name": i, "id": i} for i in
                                        register_dataset[table_col].columns],
                                       style_table={'height': '400px',
                                                    'overflowY': 'auto'})]), )
    question4_app.layout = html.Div([dcc.Tabs(listTabs)])
    question4_app.run_server(mode='inline', debug=True)
    print(reregisterstory)


def remain_time_question5(remain_data, check_col, period_col='Period'):
    question5_app = JupyterDash(__name__)
    listTabs = []
    period = remain_data[period_col]
    zero_lastdata = ""
    for ind in check_col:
        remain_num = remain_data[ind].values
        for i in range(np.size(remain_num)):
            if remain_num[i] == 0:
                zero_lastdata = period[i]
    remainstory = remain_story.render(zl=zero_lastdata)  # It can do more if I know the rule of answering this question
    listTabs.append(dcc.Tab(
        label='What is the number of children remaining on the CPR for more than 1 year and can the CPC be assured that it is necessary for any child to remain on the CPR for more than 1 year?',
        children=[html.P(remainstory),
                  dash_table.DataTable(remain_data.to_dict('records'),
                                       [{"name": i, "id": i} for i in
                                        remain_data.columns],
                                       style_table={'height': '400px',
                                                    'overflowY': 'auto'})]), )
    question5_app.layout = html.Div([dcc.Tabs(listTabs)])
    question5_app.run_server(mode='inline', debug=True)
    print(remainstory)


def enquiries_question6(enquiries_data, AC_enquiries, AS_enquiries, MT_enquiries, period_col='Period'):
    question6_app = JupyterDash(__name__)
    listTabs = []
    period = enquiries_data[period_col]
    ACmean = statistics.mean(enquiries_data[AC_enquiries].values)
    ASmean = statistics.mean(enquiries_data[AS_enquiries].values)
    MTmean = statistics.mean(enquiries_data[MT_enquiries].values)
    enquiriesstory = enquiries_story.render(indeNum=(np.size(period)), ACM=ACmean, ASM=ASmean, MTM=MTmean,
                                            ACE=enquiries_data[AC_enquiries].values,
                                            ASE=enquiries_data[AS_enquiries].values,
                                            MTE=enquiries_data[MT_enquiries].values, period=period)
    listTabs.append(dcc.Tab(
        label='To what extent do agencies make use of the CPR?  If they are not utilising it, what are the reasons for that?',
        children=[html.P(enquiriesstory),]), )
    question6_app.layout = html.Div([dcc.Tabs(listTabs)])
    question6_app.run_server(mode='inline', debug=True)
    print(enquiriesstory)
