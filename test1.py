import unittest
import pandas
import numpy
import sklearn
import main

class TestCleanData(unittest.TestCase):
    def test_CleanData(self):
        # Test cleandata can remove all ? from data
        check1 = 0
        features = pandas.read_csv('attributes.csv', delim_whitespace=True)
        dataset = pandas.read_csv('communities.data', names=features['attributes'])
        dataset = dataset.drop(columns=['state', 'county', 'community', 'communityname', 'fold'], axis=1)
        dataset = dataset.drop(columns=['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'], axis=1)
        result = main.cleanData(dataset, 0.8)
        if '?' in result.values:
            check1 = 1
        self.assertEqual(check1, 0)
        return (result)

class DefaultModel(unittest.TestCase):
    features = pandas.read_csv('attributes.csv', delim_whitespace=True)
    dataset = pandas.read_csv('communities.data', names=features['attributes'])
    dataset = dataset.drop(columns=['state', 'county', 'community', 'communityname', 'fold'], axis=1)
    dataset = dataset.drop(columns=['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'], axis=1)
    global data,Xcol,X,y,gbr_params,n_estimators, max_depth
    data = main.cleanData(dataset, 0.8)
    Xcol= ['pctWPubAsst', 'PctHousLess3BR', 'PctPersOwnOccup']
    ycol = 'ViolentCrimesPerPop'
    X= data[Xcol].values
    y= data[ycol]
    gbr_params = {'n_estimators': 500,
                  'max_depth': 3,
                  'min_samples_split': 5,
                  'learning_rate': 0.01,
                  'loss': 'ls'}
    n_estimators=500
    max_depth=3
    def test_LinearDefaultModel(self):
        # Test LinearDefaultModel works well
        columns, linearData, predicted, mse, rmse, r2 = main.LinearDefaultModel(X, y, Xcol)
        self.assertEqual(type(columns), dict)
        self.assertEqual(type(linearData), pandas.core.frame.DataFrame)
        self.assertEqual(type(predicted), numpy.ndarray)
        self.assertEqual(type(mse), numpy.float64)
        self.assertEqual(type(rmse), numpy.float64)
        self.assertEqual(type(r2), numpy.float64)
    def test_LogisticrDefaultModel(self):
        # Test LogisticDefaultModel works well
        columns1, logisticData1, columns2, logisticData2, r2 = main.LogisticrDefaultModel(X, y, Xcol)
        self.assertEqual(type(columns1), dict)
        self.assertEqual(type(logisticData1), pandas.core.frame.DataFrame)
        self.assertEqual(type(columns2), dict)
        self.assertEqual(type(logisticData2), pandas.core.frame.DataFrame)
        self.assertEqual(type(r2), numpy.float64)
    def test_GradientBoostingDefaultModel(self):
        # Test GradientBoostingDefaultModel works well
        model,mse,rmse,r2 = main.GradientBoostingDefaultModel(X, y, Xcol,gbr_params)
        self.assertEqual(type(model), sklearn.ensemble._gb.GradientBoostingRegressor)
        self.assertEqual(type(mse),numpy.float64)
        self.assertEqual(type(rmse), numpy.float64)
        self.assertEqual(type(r2), numpy.float64)
    def test_RandomForestDefaultModel(self):
        # Test RandomForestDefaultModel works well
        tree_small, rf_small, DTData, r2, mse, rmse=main.RandomForestDefaultModel(X, y, Xcol, n_estimators, max_depth)
        self.assertEqual(type(tree_small), sklearn.tree._classes.DecisionTreeRegressor)
        self.assertEqual(type(rf_small),sklearn.ensemble._forest.RandomForestRegressor)
        self.assertEqual(type(DTData), pandas.core.frame.DataFrame)
        self.assertEqual(type(r2), numpy.float64)
        self.assertEqual(type(mse),numpy.float64)
        self.assertEqual(type(rmse), numpy.float64)
    def test_DecisionTreeDefaultModel(self):
        # Test DecisionTreeDefaultModel works well
        model, r2, mse, rmse, DTData=main.DecisionTreeDefaultModel(X, y, Xcol, max_depth)
        self.assertEqual(type(model), sklearn.ensemble._gb.DecisionTreeRegressor)
        self.assertEqual(type(mse),numpy.float64)
        self.assertEqual(type(rmse), numpy.float64)
        self.assertEqual(type(r2), numpy.float64)
        self.assertEqual(type(DTData), pandas.core.frame.DataFrame)

class MicroPlanning(unittest.TestCase):
    def test_MicroLexicalization(self):
        # Test if the grammar correct works well
        test_str="This is a a text,, for testting the theAutomated operations to correct grammar and word inflection."
        correct_str=main.MicroLexicalization(test_str)
        key="This is a text, for testing the automated operations to correct grammar and word inflection."
        self.assertEqual(correct_str,key)
    def test_variablenamechange(self):
        # Test if the variable names change works well
        features = pandas.read_csv('attributes.csv', delim_whitespace=True)
        dataset = pandas.read_csv('communities.data', names=features['attributes'])
        dataset = dataset.drop(columns=['state', 'county', 'community', 'communityname', 'fold'], axis=1)
        dataset = dataset.drop(columns=['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'], axis=1)
        dataset = main.cleanData(dataset, 0.8)
        Xoldname,yoldname=['pctWPubAsst','PctHousLess3BR','PctPersOwnOccup'], 'ViolentCrimesPerPop'
        Xold,yold=dataset[Xoldname].values,dataset[yoldname]
        Xkey,ykey=['percentage of households with public assistance income','percent of housing units with less than 3 bedrooms','percent of people in owner occupied households'],'total number of violent crimes per 100K popuation'
        newdataset, Xnewname, ynewname=main.variablenamechange(dataset, Xoldname,yoldname,Xkey,ykey )
        Xnew,ynew=newdataset[Xnewname].values,newdataset[ynewname]
        self.assertEqual(Xnewname,Xkey)
        self.assertEqual(ynewname,ykey)
        self.assertEqual(Xold.all(), Xnew.all())
        self.assertEqual(yold.all(), ynew.all())

if __name__ == '__main__':
    unittest.main()
