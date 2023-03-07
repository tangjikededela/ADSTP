from pandas import read_csv
import numpy
import ADSTP.IntegratedPipeline as CT

# # Set the pipelines
pipelines = CT.special_datastory_pipelines_for_Scottish_government_report

# # # Dataset of Drug-related death. Sort by gender, age, drug type.
col_names = ['years', 'drug related deaths', 'males', 'females', 'Deaths under age 14',
             'Deaths between the ages of 15 and 24', 'Deaths between the ages of 25 and 34',
             'Deaths between the ages of 35 and 44', 'Deaths between the ages of 45 and 54',
             'Deaths between the ages of 55 and 64', 'Deaths ages over 65', 'average age of death',
             'dead by Heroin/morphine 2', 'dead by Methadone', 'dead by Heroin/morphine, Methadone or Bupren-orphine',
             'dead by Codeine or a codeine-containing compound',
             'dead by Dihydro-codeine or a d.h.c-containing compound', 'dead by any opiate or opioid', ]
data = read_csv("./data/drugdeathsexagetype.csv", header=None, names=col_names)
X = data.years  # Features
Xcolname = "years"
# Q1. What is the trend in the number of drug-related deaths from 1996 to 2020?
ycolname = "drug related deaths"
y = data[ycolname]
pipelines.segmentedregression_fit(X, y, Xcolname, ycolname, level=1, graph=False, base=False, r2=False, p=False,
                                         breakpointnum=5, governmentdrug=True, governmentchild=False)
# Q2. How has the number of drug-related deaths of men and women changed from 1996 to 2020?
y1name = "males"
y2name = "females"
y1 = data[y1name]
y2 = data[y2name]
model = "magnificationcompare"
pipelines.dependentcompare_con(model, X, y1, y2, Xcolname, y1name, y2name, begin=4, end=24)

# # # Dataset of The amount of drugs found in the deceased, and the cause of death
col_names = ['Year', 'all drug-related deaths', 'more than one drug was found', 'only one drug was found',
             'more than one drug was found in %', 'more than one drug was found to be present in the body',
             'accidental poisonings']
data = read_csv("./data/onedrug.csv", header=None, names=col_names)
Xcolname = "Year"
ycolname1 = "accidental poisonings"
ycolname2 = "all drug-related deaths"
X = data.Year  # Features
y1 = data[ycolname1]
y2 = data[ycolname2]

# Q3. Accidental poisoning accounts for what percent of drug-related deaths?
m = "independenttwopointcomparison"
pipelines.independenttwopointcompare_con(m, X, Xcolname, y1, y2, ycolname1, ycolname2, point="", mode="mag")
# Q4. More than one drug was found in what percent of the dead?
ycolname1 = "more than one drug was found to be present in the body"
y1 = data[ycolname1]
pipelines.independenttwopointcompare_con(m, X, Xcolname, y1, y2, ycolname1, ycolname2, point="", mode="mag")

# # # Dataset of Drug-related death number Sort by drug type
col_names = ['Year', 'death by ‘street’ benzodiazepines (such as etizolam)',
             'death by methadone', 'death by heroin/morphine',
             'death by gabapentin and/or pregabalin',
             'death by cocaine', 'death by opiates/opioids (such as heroin/morphine and methadone)',
             'death by benzodiazepines (such as diazepam and etizolam)']
data = read_csv("./data/drugsubstancesimplicated.csv", header=None, names=col_names)
X = data.Year  # Features
Xcolname = "Year"
ycolnames = ["death by ‘street’ benzodiazepines (such as etizolam)",
             "death by methadone", "death by heroin/morphine",
             "death by gabapentin and/or pregabalin",
             "death by cocaine", "death by opiates/opioids (such as heroin/morphine and methadone)",
             "death by benzodiazepines (such as diazepam and etizolam)"]
begin = 0
end = numpy.size(X) - 1
y = data[ycolnames]
# Q5. How many of the deceased were caused by the each of the following drugs?
model = 1
category_name = " number of deaths where one or more of the following substances "
pipelines.batchprovessing_con(model, X, y, Xcolname, ycolnames, category_name, end, begin=0)
