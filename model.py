import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from pandas import DataFrame
import heapq
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, accuracy_score
from pygam import LinearGAM, s, f, te
import statsmodels.api as sm
import scipy.signal as signal


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




def GradientBoostingDefaultModel(X, y, Xcol, gbr_params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = ensemble.GradientBoostingRegressor(**gbr_params)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    mse = mean_squared_error(y_test, model.predict(X_test))
    rmse = mse ** (1 / 2.0)
    r2 = model.score(X_test, y_test)
    return (model, mse, rmse, r2)


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

def GAMModel(data, Xcol, ycol, X,y,expect=1,epochs=100,splines=''):
    titles = Xcol
    n = np.size(titles)
    # Fitting the model
    lams = np.random.rand(epochs, np.size(titles))  # Epochs
    lams = lams * np.size(titles) - 3
    lams = np.exp(lams)
    if splines=='':
        splines=n+5
    gam = LinearGAM(n_splines=splines).gridsearch(X, y, lam=lams)
    r2 = gam.statistics_.get('pseudo_r2')
    p = gam.statistics_.get('p_values')
    # Plotting
    fig, axs = plt.subplots(1, np.size(Xcol), figsize=(40, 10))
    factor = ""
    mincondition = ""
    condition = ""
    choose=expect
    conflict = [0] * np.size(Xcol)
    # Analysis and Story Generate
    for i, ax in enumerate(axs):
        maxfirst = 0
        minfirst = 0
        XX = gam.generate_X_grid(term=i)
        Xpre = XX[:, i]
        ypre = gam.partial_dependence(term=i, X=XX)
        # Find min & max
        maxpoint = signal.argrelextrema(gam.partial_dependence(term=i, X=XX), np.greater)
        minpoint = signal.argrelextrema(gam.partial_dependence(term=i, X=XX), np.less)
        maxpoint = maxpoint[0]
        minpoint = minpoint[0]
        extremum = 0
        loopnum = int(np.size(minpoint)) + int(np.size(maxpoint))
        allpoint = np.hstack((maxpoint, minpoint))
        allpoint.sort()
        if np.size(maxpoint) != 0 and np.size(minpoint) != 0:
            if maxpoint[0] > minpoint[0] and minpoint[0] > 0:
                minfirst = 1
            elif maxpoint[0] < minpoint[0] and maxpoint[0] > 0:
                maxfirst = 1
            for j in range(loopnum):

                if j == 0:
                    if minfirst == 1:
                        factor = "With the growth of  " + Xcol[
                            i] + " , " + ycol + " first decreases to "
                        if ypre[minpoint[0]] == min(ypre):
                            factor = factor + "the minimum when " + Xcol[i] + " is " + str(
                                round(Xpre[minpoint[0]], 3))
                            mincondition = mincondition + Xcol[i] + " is around " + str(
                                round(Xpre[minpoint[0]], 3)) + ", "
                            extremum = 1
                        else:
                            factor = factor + "a relative minimum"
                    elif maxfirst == 1:
                        factor = "With the growth of  " + Xcol[i] + " , " + ycol + " first increase to "
                        if ypre[maxpoint[0]] == max(ypre):
                            factor = factor + "the maximum when " + Xcol[i] + " is " + str(
                                round(Xpre[maxpoint[0]], 3))
                            condition = condition + Xcol[i] + " is around " + str(
                                round(Xpre[maxpoint[0]], 3)) + ", "
                            extremum = 1
                        else:
                            factor = factor + "a relative maximum"
                elif j != 0:
                    if maxfirst == 1:
                        minfirst = 1
                        maxfirst = 0
                        if ypre[allpoint[j]] == min(ypre):
                            factor = factor + "\nthen decreases to the minimum when " + Xcol[
                                i] + " is " + str(round(Xpre[allpoint[j]], 3))
                            mincondition = mincondition + Xcol[i] + " is around " + str(
                                round(Xpre[allpoint[j]], 3)) + ", "
                            extremum = 1
                        else:
                            factor = factor + "\nthen decreases to a relative minimum."
                    elif minfirst == 1:
                        maxfirst = 1
                        minfirst = 0
                        if ypre[allpoint[j]] == max(ypre):
                            factor = factor + "\nthen increases to the maximum when " + Xcol[
                                i] + " is " + str(round(Xpre[allpoint[j]], 3))
                            condition = condition + Xcol[i] + " is around " + str(
                                round(Xpre[allpoint[j]], 3)) + ", "
                            extremum = 1
                        else:
                            factor = factor + "\nthen increases to a relative maximum."
            if allpoint[loopnum - 1] in maxpoint:
                factor = factor + " and finally continues to decline."
                if extremum == 0 and choose == 0:
                    condition = condition + Xcol[i] + " the less the better, "
                    # need change here
                    if ypre[0] > ypre[len(ypre) - 1]:
                        mincondition = mincondition + Xcol[i] + " the higher the better, "
                    else:
                        mincondition = mincondition + Xcol[i] + " the less the better, "
                elif extremum == 0 and choose == 1:
                    if ypre[0] > ypre[len(ypre) - 1]:
                        mincondition = mincondition + Xcol[i] + " the higher the better, "
                    else:
                        mincondition = mincondition + Xcol[i] + " the less the better, "
                elif extremum == 1 and choose == 1:
                    if ypre[0] > ypre[len(ypre) - 1]:
                        mincondition = mincondition + " or " + Xcol[i] + " the higher the better, "
                    elif ypre[0] <= min(ypre):
                        mincondition = mincondition + "or" + Xcol[i] + " the less the better, "
                elif extremum == 1 and choose == 0:
                    if ypre[0] > ypre[len(ypre) - 1]:
                        mincondition = mincondition + Xcol[i] + " the higher the better, "
                    else:
                        mincondition = mincondition + Xcol[i] + " the less the better, "
            else:
                factor = factor + " and finally continues to increase."
                if extremum == 0 and choose == 0:
                    condition = condition + Xcol[i] + " the higher the better, "
                    mincondition = mincondition + Xcol[i] + " the less the better, "
                elif extremum == 0 and choose == 1:
                    mincondition = mincondition + Xcol[i] + " the less the better, "
                elif extremum == 1 and choose == 1:
                    mincondition = mincondition + Xcol[i] + " the less the better, "
                elif extremum == 1 and choose == 0:
                    mincondition = mincondition + Xcol[i] + " the less the better, "
        elif np.size(maxpoint) != 0 and np.size(minpoint) == 0:
            factor = "With the growth of  " + Xcol[
                i] + " , the " + ycol + " first increase to the maximum when " + Xcol[i] + " is " + str(
                Xpre[maxpoint]) + " then continues to decline."
            condition = condition + Xcol[i] + " is around " + str(Xpre[maxpoint]) + ", "
            if ypre[0] > ypre[len(ypre) - 1]:
                mincondition = mincondition + Xcol[i] + " the higher the better, "
            else:
                mincondition = mincondition + Xcol[i] + " the less the better, "
        elif np.size(maxpoint) == 0 and np.size(minpoint) != 0:
            mincondition = mincondition + Xcol[i] + " is around " + str(Xpre[minpoint]) + ", "
            factor = "With the growth of  " + Xcol[
                i] + " , the " + ycol + " first decrease to the minimum when " + Xcol[i] + " is " + str(
                Xpre[minpoint]) + " then continues to increase."
        elif np.size(maxpoint) == 0 and np.size(minpoint) == 0:
            if ypre[0] < ypre[np.size(ypre) - 1]:
                factor = ycol + " keep increase as " + Xcol[i] + " increase."
                condition = condition + Xcol[i] + " the larger the better, "
                mincondition = mincondition + Xcol[i] + " the less the better, "
            else:
                factor = ycol + " keep decrease as " + Xcol[i] + " increase."
                condition = condition + Xcol[i] + " the less the better, "
                mincondition = mincondition + Xcol[i] + " the higher the better, "
        if np.size(minpoint) >= 2 and np.size(maxpoint) >= 2:
            factor = factor + " It is worth noting that as " + Xcol[
                i] + " increases, " + ycol + " tends to fluctuate periodically."
        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
        ax.set_title(Xcol[i])
        conflict[i] = factor
    nss = ""
    ss = ""
    for i in range(np.size(p) - 1):
        p[i] = round(p[i], 3)
        if p[i] > 0.05:
            nss = nss + Xcol[i] + ", "
        else:
            ss = ss + Xcol[i] + ", "
    return (gam,data,Xcol,ycol,r2,p,conflict,nss,ss,mincondition,condition)


def loop_mean_compare(dataset, Xcol,ycol):
    diff = [0] * np.size(Xcol)
    i=0
    for ind in Xcol:
        diff[i] = (statistics.mean(dataset[ind]) - statistics.mean(dataset[ycol]))
        i = i + 1
    return (diff)


def find_row_n_max(dataset, Xcol, r=0, max_num=5):

    row_data = dataset[Xcol].values[0:r+1][r]
    max_data = (heapq.nlargest(max_num, row_data))
    max_factor = []
    for ind in Xcol:
        if dataset[ind].values[0:r+1][r] in max_data:
            max_factor.append(ind)
    return (max_factor)

def detect_same_elements(list1,list2):
    same_element=0
    for i in list1:
        if i in list2:
            same_element = same_element + 1
    return (same_element)

def select_one_element(dataset, Xcol,ycol):
    datasize = np.size(dataset[Xcol])
    y = dataset[ycol].values[0:datasize][datasize - 1]
    X = dataset[Xcol].values[0:datasize][datasize - 1]
    return (X,y)

def find_all_zero_after_arow(dataset, Xcol, ycol):
    period = dataset[ycol]
    zero_lastdata = ""
    for ind in Xcol:
        remain_num = dataset[ind].values
        for i in range(np.size(remain_num)):
            if remain_num[i] == 0:
                zero_lastdata = period[i]
    return (zero_lastdata)

def find_column_mean(dataset):
    meancol=statistics.mean(dataset)
    return (meancol)

