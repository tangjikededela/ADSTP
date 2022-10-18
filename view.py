import numpy as np
import seaborn as sns
from pandas import DataFrame
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
from sklearn import tree
from yellowbrick.regressor import ResidualsPlot
from jinja2 import Environment, FileSystemLoader
from scipy import stats
import math
from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, dash_table, callback
import plotly.express as px
import base64
import language_tool_python

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
GAMslinear_stats = env.get_template('GAMsLinearL1')
GAMslinear_R2 = env.get_template('GAMsLinearL2')
GAMslinear_P = env.get_template('GAMsLinearL3')
GAMslinear_sum = env.get_template('GAMsLinearL4')
GB1 = env.get_template('GB1')
GB2 = env.get_template('GB2')
GB3 = env.get_template('GB3')

# variables which each load a different segmented regression template
segmented_R2P = env.get_template('testPiecewisePwlfR2P')
segmented_R2 = env.get_template('testPiecewisePwlfR2')
segmented_P = env.get_template('testPiecewisePwlfP')
segmented_B = env.get_template('testPiecewisePwlfB')
segmented_GD1 = env.get_template('drugreport1')
segmented_GC1 = env.get_template('childreport1')

# For Aberdeen City CP
register_story = env.get_template('register.txt')
risk_factor_story = env.get_template('risk_factor.txt')
reregister_story = env.get_template('reregister.txt')
remain_story = env.get_template('remain_story.txt')
enquiries_story = env.get_template('enquiries_story.txt')

# For different dependent variables compared DRD
dc1 = env.get_template('dependentmagnificationcompare')
dc2 = env.get_template('samedependentmagnificationcompare')
dc3 = env.get_template('dependentquantitycompare')
dc4 = env.get_template('trendpercentagedescription')
dct = env.get_template('trenddescription')
tppc = env.get_template('twopointpeak_child')
# for different independent variables compared
idc1 = env.get_template('independentquantitycompare')
idtpc = env.get_template('independenttwopointcomparison')
# for batch processing
bp1 = env.get_template('batchprocessing1')
bp2 = env.get_template('batchprocessing2')
# for pycaret
automodelcompare1 = env.get_template('AMC1.txt')
automodelcompare2 = env.get_template('AMC2.txt')

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


def ModelData_view(mae_metrics, rmse_metrics, ycol):
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


def start_app():
    app_name = JupyterDash(__name__)
    listTabs = []
    return (app_name, listTabs)


def run_app(app_name, listTabs):
    app_name.layout = html.Div([dcc.Tabs(listTabs)])
    app_name.run_server(mode='inline', debug=True)


def LinearModelStats_view(data, Xcol, ycol, linearData, r2, questionset, trend):
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


def LogisticModelStats_view(data, Xcol, ycol, logisticData1, logisticData2, r2, questionset):
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
    # intro = model.MicroLexicalization(intro)
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
        # conflict = model.MicroLexicalization(conflict)
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
    # summary = model.MicroLexicalization(summary)
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


def GradientBoostingModelStats_view(data, Xcol, ycol, GBmodel, r2, questionset, gbr_params):
    # Store importance figure
    plt.bar(Xcol, GBmodel.feature_importances_)

    plt.title("Importance Score")
    plt.savefig('pictures/{}.png'.format("GB1"))
    plt.clf()

    _base64 = []
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("GB1"), 'rb').read()).decode('ascii'))
    # Training & Test Deviance Figure
    test_score = np.zeros((gbr_params['n_estimators'],), dtype=np.float64)
    fig = plt.figure(figsize=(8, 8))
    plt.title('Deviance')
    plt.plot(np.arange(gbr_params['n_estimators']) + 1, GBmodel.train_score_, 'b-',
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


def RandomForestModelStats_view(data, Xcol, ycol, tree_small, rf_small, DTData, r2, mse, questionset):
    # Save the tree as a png image
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    export_graphviz(tree_small, out_file='pictures/small_tree.dot', feature_names=Xcol, rounded=True, precision=1)
    (graph,) = pydot.graph_from_dot_file('pictures/small_tree.dot')
    graph.write_png('pictures/small_tree.png', prog=['dot'])
    encoded_image = base64.b64encode(open("pictures/small_tree.png", 'rb').read()).decode('ascii')
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


def DecisionTreeModelStats_view(data, Xcol, ycol, DTData, DTmodel, r2, mse, questionset):
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
    tree.plot_tree(DTmodel,
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
    explain = TreeExplain(DTmodel, Xcol)
    # Text need to fix here
    conflict = explain
    listTabs.append(
        dcc.Tab(label='Tree Explanation', children=[html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
                                                    html.Pre(conflict)]), )
    summary = DecisionTree3.render(imp=imp, ycol=ycol, r2=round(r2, 3), qs=questionset, mse=mse)
    listTabs.append(dcc.Tab(label='Summary', children=[dcc.Graph(figure=fig), html.P(summary)]), )
    DT_app.layout = html.Div([dcc.Tabs(listTabs)])
    DT_app.run_server(mode='inline', debug=True)


def GAMs_view(gam, data, Xcol, ycol, r2, p, conflict, nss, ss, mincondition, condition):
    # Analysis and Graphs Generate
    for i, term in enumerate(gam.terms):
        if term.isintercept:
            continue
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        plt.plot(XX[:, term.feature], pdep)
        plt.plot(XX[:, term.feature], confi, c='r', ls='--')
        plt.title(Xcol[i])
        plt.savefig('pictures/{}.png'.format(i))
        plt.clf()
    # print(GAMslinear_R2.render(R=round(r2.get('explained_deviance'), 3), Xcol=Xcol, ycol=ycol,
    #                            indeNum=np.size(Xcol)))
    # print(GAMslinear_P.render(pvalue=p, Nss=nss, Ss=ss, Xcol=Xcol, ycol=ycol,
    #                           indeNum=np.size(Xcol)))
    # print(GAMslinear_sum.render(ycol=ycol, condition=condition, mincondition=mincondition, demand=1))
    gamm_app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    listTabs = []
    # Add to dashbord GAMs Model Statistics
    intro = GAMslinear_stats.render(Xcol=Xcol, ycol=ycol, trend=0, indeNum=np.size(Xcol), r2=r2['McFadden_adj'])
    # Add table
    aim = Xcol
    aim.insert(0, ycol)
    # newstory = MicroLexicalization(story)
    # listTabs.append(dcc.Tab(label='GAMs Model Stats', children=[html.P(intro),
    #                                                             dash_table.DataTable(data[aim].to_dict('records'),
    #                                                                                  [{"name": i, "id": i} for i in
    #                                                                                   data[aim].columns],
    #                                                                                  style_table={'height': '400px',
    #                                                                                               'overflowY': 'auto'})]), )
    dash_with_table(gamm_app, listTabs, intro, data[aim], 'GAMs Model Stats')
    # Fromat list with files names
    _base64 = []
    for i in range(len(Xcol)):
        _base64.append(base64.b64encode(open('pictures/{}.png'.format(i), 'rb').read()).decode('ascii'))
    aim.remove(ycol)
    # Add to dashbord values of Xcol and graphs
    for i in range(len(Xcol)):
        # other story for one independent variable add in here
        story = gamStory.render(pvalue=p[i], xcol=Xcol[i], ycol=ycol, ) + conflict[i]
        dash_with_figure(gamm_app, listTabs, story, Xcol[i], _base64[i], path='data:image/png;base64,{}')
        # listTabs.append(dcc.Tab(label=Xcol[i], children=[
        #     html.Img(src='data:image/png;base64,{}'.format(_base64[i])), html.P(story)
        # ]))
    summary = GAMslinear_P.render(pvalue=p, Nss=nss, Ss=ss, Xcol=Xcol, ycol=ycol,
                                  indeNum=np.size(Xcol)) + GAMslinear_sum.render(ycol=ycol, condition=condition,
                                                                                 mincondition=mincondition, demand=1)
    dash_only_text(gamm_app, listTabs, summary, 'Summary')
    # listTabs.append(dcc.Tab(label='Summary', children=[html.P(summary), ]), )

    gamm_app.layout = html.Div([dcc.Tabs(listTabs)])
    gamm_app.run_server(mode='inline', debug=True)


def dash_with_figure(app_name, listTabs, text, label, format, path='data:image/png;base64,{}'):
    listTabs.append(dcc.Tab(label=label, children=[
        html.Img(src=path.format(format)), html.P(text)
    ]))


def dash_with_table(app_name, listTabs, text, dataset, label):
    listTabs.append(dcc.Tab(label=label,
                            children=[html.P(text),
                                      dash_table.DataTable(dataset.to_dict('records'),
                                                           [{"name": i, "id": i} for i in
                                                            dataset.columns],
                                                           style_table={'height': '400px',
                                                                        'overflowY': 'auto'})]), )


def dash_only_text(app_name, listTabs, text, label):
    listTabs.append(dcc.Tab(label=label,
                            children=[html.P(text), ]), )


def register_question1_view(register_dataset, per1000inCity_col, diff, table_col, label, app, listTabs):
    registerstory = "The data from local comparators features in the Child Protection Register (CPR) report prepared quarterly. "
    i = 0
    for ind in per1000inCity_col:
        reslut = register_story.render(Xcol=ind, minX=min(register_dataset[ind]), maxX=max(register_dataset[ind]),
                                       diff=diff[i])
        registerstory = registerstory + reslut
        i = i + 1
    dash_with_table(app, listTabs, registerstory, register_dataset[table_col], label)


def riskfactor_question1_view(dataset, max_factor, same_factor, label, cityname, app, listTabs):
    riskstory = risk_factor_story.render(indeNum=(np.size(max_factor)), max_factor=max_factor,
                                         same_factor=same_factor,
                                         cityname=cityname)
    dash_with_table(app, listTabs, riskstory, dataset, label)


def re_register_question4_view(register_dataset, national_average_reregistration, reregister_lastyear, period,
                               table_col, label, app, listTabs):
    reregisterstory = reregister_story.render(nar=national_average_reregistration, rrly=reregister_lastyear,
                                              time=period)
    dash_with_table(app, listTabs, reregisterstory, register_dataset[table_col], label)


def remain_time_question5_view(remain_data, zero_lastdata, label, app, listTabs):
    remainstory = remain_story.render(zl=zero_lastdata)  # It can do more if I know the rule of answering this question
    dash_with_table(app, listTabs, remainstory, remain_data, label)


def enquiries_question6_view(ACmean, ASmean, MTmean, ACdata, ASdata, MTdata, period, label, app, listTabs):
    enquiriesstory = enquiries_story.render(indeNum=(np.size(period)), ACM=ACmean, ASM=ASmean, MTM=MTmean,
                                            ACE=ACdata,
                                            ASE=ASdata,
                                            MTE=MTdata, period=period)
    dash_only_text(app, listTabs, enquiriesstory, label)


def segmentedregressionsummary_CPview(X, ymax, Xmax, ylast, Xlast, diff1, diff2, Xbegin, Xend, yend, iP, dP, nP, Xcol,
                                      ycol):
    print(segmented_GC1.render(
        X=X,
        ymax=ymax,
        Xmax=Xmax,
        ylast=ylast,
        Xlast=Xlast,
        diff1=diff1,
        diff2=diff2,
        Xbegin=Xbegin,
        Xend=Xend,
        yend=yend,
        iP=iP,
        dP=dP,
        nP=nP,
        Xcol=Xcol,
        ycol=ycol, ))


def segmentedregressionsummary_DRDview(increasePart, decreasePart, notchangePart, ycolname, maxIncrease, maxDecrease):
    print(segmented_GD1.render(
        iP=increasePart,
        dP=decreasePart,
        nP=notchangePart,
        ycol=ycolname,
        mI=maxIncrease,
        mD=maxDecrease, ))


def dependentcompare_view(Xcolname, begin, end, ycolname1, ycolname2, magnification1, magnification2, X, X1, X2):
    print(dc1.render(Xcol=Xcolname, begin=begin, end=end, loopnum=end, y1name=ycolname1, y2name=ycolname2,
                     magnification1=magnification1,
                     magnification2=magnification2, X=X, X1=X1, X2=X2))


def batchprovessing_view1(m, Xcolname, X1, X2, y, allincrease, alldecrease, category_name, ycolnames, begin, end):
    story = (bp1.render(mode=m, Xcol=Xcolname, X1=0, allincrease=allincrease, alldecrease=alldecrease,
                        category_name=category_name)) + "\n"
    for i in range(np.size(ycolnames) - 1):
        ycolname = ycolnames[i]
        ydata = y[ycolname]
        y1 = ydata[begin]
        y2 = ydata[end]
        story = story + bp2.render(mode=m, ycol=ycolname, y1=y1, y2=y2, X1=X1, X2=X2, mag=0)
    print(story)


def batchprovessing_view2(m, Xcolname, X1, allincrease, alldecrease, category_name, total, ycolnames, y, point):
    story = (bp1.render(mode=m, Xcol=Xcolname, X1=X1, allincrease=False, alldecrease=False,
                        category_name=category_name)) + "\n"
    for i in range(np.size(ycolnames) - 1):
        ycolname = ycolnames[i]
        ydata = y[ycolname]
        y1 = ydata[point]
        mag = np.round(y1 / total, 2)
        story = story + bp2.render(mode=m, ycol=ycolname, y1=y1, y2=0, X1=0, X2=0, mag=mag)
    print(story)


def independenttwopointcompare_view(Xcolname, point, ycolname1, ycolname2, X, y1, y2, mode, mag):
    print(idtpc.render(Xcol=Xcolname, point=point, y1name=ycolname1, y2name=ycolname2, X=X, y1=y1, y2=y2,
                       mode=mode, mag=mag))


def two_point_and_peak_child_view(Xcolname, ycolname, Xpeak, ypeak, X1, X2, y1, y2):
    print(tppc.render(Xcol=Xcolname, ycol=ycolname, Xpeak=Xpeak, ypeak=ypeak, X1=X1, X2=X2, y1=y1, y2=y2))


def trendpercentage_view(Xcolname, begin, end, ycolname, X, y, std, samepoint):
    print(dc4.render(Xcol=Xcolname, begin=begin, end=end, ycol=ycolname, X=X, y=y, std=std, samepoint=samepoint))


def pycaret_find_one_best_model(model, detail, n, sort, exclude):
    print(automodelcompare1.render(best=model, detail=detail, n_select=n, sort=sort, exclude=exclude))


def pycaret_find_best_models(model, detail, n, sort, exclude, length):
    print(automodelcompare2.render(best=model, detail=detail, n_select=n, sort=sort, exclude=exclude, length=length))
