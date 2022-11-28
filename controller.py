import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import model as MD
import view as VW
import sys

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

def create_dummy_variables(dataset, variables):
    for var in variables:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(dataset[var], prefix=var)
        datatemp = dataset.join(cat_list)
        dataset = datatemp
    data_vars = dataset.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in variables]
    data_final = dataset[to_keep]
    return (data_final)

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
        models_results.append(MD.RenderModel(i, X_train, y_train))
        mae_metrics.append(MD.MAE(models_results[current_index], X_test, y_test))
        rmse_metrics.append(MD.RMSE(models_results[current_index], X_test, y_test))
    VW.ModelData_view(mae_metrics, rmse_metrics,ycol)

class general_datastory_for_pycaret_pipelines():

    def pycaret_model_fit(dataset,types,pycaretname,comparestory,comapre_results,target):
        if types==0:
            independent_var, imp, Accuracy, AUC, imp_figure, Error_figure,SHAP_figure,imp_pos_ave,imp_pos_value_ave,imp_neg_ave,imp_neg_value_ave = MD.pycaret_create_model(types, pycaretname)
            fitstory, impstory = VW.pycaret_classification_model_summary_view(imp, Accuracy, AUC,imp_pos_ave,imp_pos_value_ave,imp_neg_ave,imp_neg_value_ave,target)
            app_name, listTabs = VW.start_app()
            VW.dash_with_table(app_name, listTabs, comparestory, dataset, "Model Compare Overview")
            _base64 = []
            _base64 = VW.read_figure(_base64, "Prediction Error")
            _base64 = VW.read_figure(_base64, "Feature Importance")
            _base64 = VW.read_figure(_base64, "SHAP summary")
            VW.dash_with_table(app_name, listTabs, fitstory, comapre_results, "Model credibility")
            # VW.dash_with_figure(app_name, listTabs, impstory, 'Variables Summary', _base64[1])
            VW.dash_with_two_figure(app_name, listTabs, impstory, 'Important Variables Summary', _base64[1],_base64[2])
            VW.run_app(app_name, listTabs)
        elif types==1:
            independent_var, imp, r2, mape, imp_figure, Error_figure,SHAP_figure,imp_pos_ave,imp_pos_value_ave,imp_neg_ave,imp_neg_value_ave = MD.pycaret_create_model(types, pycaretname)
            fitstory, impstory = VW.pycaret_model_summary_view(imp, r2, mape,imp_pos_ave,imp_pos_value_ave,imp_neg_ave,imp_neg_value_ave,target)
            app_name, listTabs = VW.start_app()
            VW.dash_with_table(app_name, listTabs, comparestory, dataset, "Model Compare Overview")
            _base64 = []
            _base64 = VW.read_figure(_base64, "Prediction Error")
            _base64 = VW.read_figure(_base64, "Feature Importance")
            _base64 = VW.read_figure(_base64, "SHAP summary")
            VW.dash_with_table(app_name, listTabs, fitstory, comapre_results, "Model credibility")
            # VW.dash_with_figure(app_name, listTabs, impstory, 'Variables Summary', _base64[1])
            VW.dash_with_two_figure(app_name, listTabs, impstory, 'Important Variables Summary', _base64[1],_base64[2])
            VW.run_app(app_name, listTabs)

    def pycaret_find_best_model(dataset,types,target,sort="",exclude=[],n=1,session_id=123,userinput="quit"):
        detail,pycaretname,comapre_results=MD.pycaret_find_best_model(dataset,types,target,sort,exclude,n,session_id)
        model = MD.model_translate(detail, n)
        excludeNum=len(exclude)
        trans_exclude=MD.inputname_to_readablename(exclude,types)
        if n ==1:
            comparestory=VW.pycaret_find_one_best_model(model, detail, n, sort, trans_exclude,excludeNum)
        elif n>1:
            comparestory=VW.pycaret_find_best_models(model, detail, n, sort, trans_exclude, excludeNum,length=len(detail))
        # print("You could use the information to fit the model or enter 'continue' the system will automatically fit the best model.")
        # userinput=input("Or enter 'quit' to end the process:")
        if userinput=="continue":
            general_datastory_for_pycaret_pipelines.pycaret_model_fit(dataset,types, pycaretname, comparestory, comapre_results,target)
        else:
            sys.exit()


class general_datastory_pipeline:
    def LinearFit(data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1, 1], trend=1):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        X = data[Xcol].values
        y = data[ycol]
        columns, linearData, predicted, mse, rmse, r2 = MD.LinearDefaultModel(X, y, Xcol)
        VW.LinearModelStats_view(data, Xcol, ycol, linearData, r2, questionset, trend)

    def LogisticFit(data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1, 1]):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        X = data[Xcol].values
        y = data[ycol]
        columns1, logisticData1, columns2, logisticData2, r2 = MD.LogisticrDefaultModel(X, y, Xcol)
        VW.LogisticModelStats_view(data, Xcol, ycol, logisticData1, logisticData2, r2, questionset)

    def GradientBoostingFit(data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1],
                                   gbr_params={'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 5,
                                               'learning_rate': 0.01, 'loss': 'ls'}):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        X = data[Xcol].values
        y = data[ycol]
        GBmodel, mse, rmse = MD.GradientBoostingDefaultModel(X, y, Xcol, gbr_params)
        VW.GradientBoostingModelStats_view(data, Xcol, ycol, GBmodel, r2, questionset, gbr_params)

    def RandomForestFit(data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1], n_estimators=10,
                               max_depth=3):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        X = data[Xcol].values
        y = data[ycol]
        tree_small, rf_small, DTData, r2, mse, rmse = MD.RandomForestDefaultModel(X, y, Xcol, n_estimators, max_depth)
        VW.RandomForestModelStats_view(data, Xcol, ycol, tree_small, rf_small, DTData, r2, mse, questionset)

    def DecisionTreeFit(data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1], max_depth=3):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        X = data[Xcol].values
        y = data[ycol]
        DTmodel, r2, mse, rmse, DTData = MD.DecisionTreeDefaultModel(X, y, Xcol, max_depth)
        VW.DecisionTreeModelStats_view(data, Xcol, ycol, DTData, DTmodel, r2, mse, questionset)

    def GAMsFit(data, Xcol, ycol, Xnewname="", ynewname="", expect=1, epochs=100, splines=''):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        X = data[Xcol].values
        y = data[ycol]
        gam, data, Xcol, ycol, r2, p, conflict, nss, ss, mincondition, condition = MD.GAMModel(data, Xcol, ycol, X, y,
                                                                                               expect, epochs, splines)
        VW.GAMs_view(gam, data, Xcol, ycol, r2, p, conflict, nss, ss, mincondition, condition)

class special_datastory_pipelines_for_ACCCP:
    def register_question1(app_name, listTabs, register_dataset, per1000inCity_col, per1000nation_col,
                           table_col=['Period', 'Registrations In Aberdeen City',
                                      'Registrations per 1000 population in Aberdeen City',
                                      'Compared with last year for Aberdeen City'],
                           label='What are the emerging trends or themes emerging from local and comparators data?'):
        diff = MD.loop_mean_compare(register_dataset, per1000inCity_col, per1000nation_col)
        VW.register_question1_view(register_dataset, per1000inCity_col, diff, table_col, label, app_name, listTabs)

    def riskfactor_question1(app_name, listTabs, risk_factor_dataset, risk_factor_col, cityname="Aberdeen City",
                             max_num=5,
                             label='What are the emerging trends or themes emerging from local single agency data?'):
        row = 0
        max_factor = MD.find_row_n_max(risk_factor_dataset, risk_factor_col, row, max_num)
        row = 1
        max_factor_lastyear = MD.find_row_n_max(risk_factor_dataset, risk_factor_col, row, max_num)
        same_factor = MD.detect_same_elements(max_factor, max_factor_lastyear)
        VW.riskfactor_question1_view(risk_factor_dataset, max_factor, same_factor, label, cityname, app_name, listTabs)

    def re_register_question4(app_name, listTabs, register_dataset, reregister_col, period_col='Period',
                              national_average_reregistration='13 - 16%',
                              table_col=['Period', 'Re-Registrations In Aberdeen City',
                                         'Re-registrations as a % of registrations in Aberdeen City',
                                         'Largest family for Aberdeen City',
                                         'Longest gap between registrations of Aberdeen City',
                                         'Shortest gap between registrations of Aberdeen City'],
                              label='To what extent is Aberdeen City consistent with the national and comparator averages for re-registration?  Can the CPC be assured that deregistered children receive at least 3 months’ post registration multi-agency support?'):
        reregister_lastyear, period = MD.select_one_element(register_dataset, reregister_col, period_col)
        VW.re_register_question4_view(register_dataset, national_average_reregistration, reregister_lastyear, period,
                                      table_col, label, app_name, listTabs)

    def remain_time_question5(app_name, listTabs, remain_data, check_col, period_col='Period',
                              label='What is the number of children remaining on the CPR for more than 1 year and can the CPC be assured that it is necessary for any child to remain on the CPR for more than 1 year?'):
        zero_lastdata = MD.find_all_zero_after_arow(remain_data, check_col, period_col)
        VW.remain_time_question5_view(remain_data, zero_lastdata, label, app_name, listTabs)

    def enquiries_question6(app_name, listTabs, enquiries_data, AC_enquiries, AS_enquiries, MT_enquiries,
                            period_col='Period',
                            label='To what extent do agencies make use of the CPR?  If they are not utilising it, what are the reasons for that?'):
        period = enquiries_data[period_col]
        ACdata = enquiries_data[AC_enquiries].values
        ASdata = enquiries_data[AS_enquiries].values
        MTdata = enquiries_data[MT_enquiries].values
        ACmean = MD.find_column_mean(ACdata)
        ASmean = MD.find_column_mean(ASdata)
        MTmean = MD.find_column_mean(MTdata)
        VW.enquiries_question6_view(ACmean, ASmean, MTmean, ACdata, ASdata, MTdata, period, label, app_name, listTabs)


class special_datastory_pipelines_for_Scottish_government_report:

    def segmentedregression_fit(X, y, Xcolname, ycolname, level, graph, base, r2, p, breakpointnum=1,
                                       governmentdrug=False, governmentchild=False):
        if governmentchild == True:
            X, ymax, Xmax, ylast, Xlast, diff1, diff2, Xbegin, Xend, yend, iP, dP, nP, Xcol, ycol = MD.segmentedregressionsummary(
                X, y, Xcolname, ycolname, level, graph, base, r2, p, breakpointnum, governmentdrug, governmentchild)
            VW.segmentedregressionsummary_CPview(X, ymax, Xmax, ylast, Xlast, diff1, diff2, Xbegin, Xend, yend, iP, dP,
                                                 nP, Xcol, ycol)
        elif governmentdrug == True:
            increasePart, decreasePart, notchangePart, ycolname, maxIncrease, maxDecrease = MD.segmentedregressionsummary(
                X, y, Xcolname, ycolname, level, graph, base, r2, p, breakpointnum, governmentdrug, governmentchild)
            VW.segmentedregressionsummary_DRDview(increasePart, decreasePart, notchangePart, ycolname, maxIncrease,
                                                  maxDecrease)

    def dependentcompare_con(m, X, y1, y2, Xcolname, ycolname1, ycolname2, begin, end):
        Xcolname, begin, end, ycolname1, ycolname2, magnification1, magnification2, X, X1, X2 = MD.NonFittingReport.dependentcompare(
            m, X, y1, y2, Xcolname, ycolname1, ycolname2, begin, end)
        VW.dependentcompare_view(Xcolname, begin, end, ycolname1, ycolname2, magnification1, magnification2, X, X1, X2)

    def batchprovessing_con(m, X, y, Xcolname, ycolnames, category_name, end, begin=0):
        if m == 1:
            m, Xcolname, X1, X2, y, allincrease, alldecrease, category_name, ycolnames, begin, end = MD.NonFittingReport.batchprovessing(
                m, X, y, Xcolname, ycolnames, category_name, end, begin)
            VW.batchprovessing_view1(m, Xcolname, X1, X2, y, allincrease, alldecrease, category_name, ycolnames, begin,
                                     end)
        elif m == 2:
            m, Xcolname, X1, allincrease, alldecrease, category_name, total, ycolnames, y, point = MD.NonFittingReport.batchprovessing(
                m, X, y, Xcolname, ycolnames, category_name, end, begin)
            VW.batchprovessing_view2(m, Xcolname, X1, allincrease, alldecrease, category_name, total, ycolnames, y,
                                     point)

    def independenttwopointcompare_con(m, X, Xcolname, y1, y2, ycolname1, ycolname2, point, mode):
        Xcolname, point, ycolname1, ycolname2, X, y1, y2, mode, mag = MD.NonFittingReport.independenttwopointcompare(m,
                                                                                                                     X,
                                                                                                                     Xcolname,
                                                                                                                     y1,
                                                                                                                     y2,
                                                                                                                     ycolname1,
                                                                                                                     ycolname2,
                                                                                                                     point,
                                                                                                                     mode)
        VW.independenttwopointcompare_view(Xcolname, point, ycolname1, ycolname2, X, y1, y2, mode, mag)

    def two_point_and_peak_child_con(m, X, y, Xcolname, ycolname, point1, point2):
        Xcolname, ycolname, Xpeak, ypeak, X1, X2, y1, y2 = MD.NonFittingReport.two_point_and_peak(m, X, y, Xcolname,
                                                                                                  ycolname, point1,
                                                                                                  point2)
        VW.two_point_and_peak_child_view(Xcolname, ycolname, Xpeak, ypeak, X1, X2, y1, y2)

    def trendpercentage_con(m, X, y, Xcolname, ycolname, begin="", end=""):
        Xcolname, begin, end, ycolname, X, y, std, samepoint = MD.NonFittingReport.samedependentcompare(m, X, y,
                                                                                                        Xcolname,
                                                                                                        ycolname, begin,
                                                                                                        end)
        VW.trendpercentage_view(Xcolname, begin, end, ycolname, X, y, std, samepoint)
