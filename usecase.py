from pandas import read_csv,DataFrame
import numpy
import controller as CT
import UI

# # Register # Question 1-2
# register_dataset = read_csv('registrations_data.csv', header=0, converters={'Period': str})
# per1000inCity_col = ['Registrations per 1000 population in Aberdeen City',
#                  'Registrations per 1000 population in Aberdeenshire', 'Registrations per 1000 population in Moray']
# per1000nation_col = 'per 1000 population in Nation'
# # table_col=['Period','Registrations In Aberdeen City','Registrations per 1000 population in Aberdeen City','Compared with last year for Aberdeen City']
# # CT.register_question1(register_dataset, per1000inCity_col, per1000nation_col)
#
# # Risk Factor # Question 1-3
# risk_factor_dataset = read_csv('risk_factor.csv', header=0, converters={'Period': str})
# risk_factor_col = ['Emotional Abuse', 'Parental Drug Misuse', 'Domestic Abuse', 'Non-engaging Family', 'Neglect',
#                'Parent mental health', 'Parental alcohol use', 'Sexual Abuse', 'Physical Abuse', 'Other concern', 'CSE',
#                'Forced Labour', 'Placing self at risk', 'Child Trafficking']
# # CT.riskfactor_question1(risk_factor_dataset, risk_factor_col, cityname="Aberdeen City", max_num=5)
#
# # Re-Register # Question 4
# reregister_col = 'Re-Registrations In Aberdeen City'
# period_col='Period'
# national_average_reregistration = '13 - 16%' # I did not find where this data come from, it can be auto by given the data.
# # CT.re_register_question4(register_dataset, reregister_col, period_col,national_average_reregistration)
#
# # Remain time # Question 5
# remain_data = read_csv('children_remaining_Aberdeen.csv', header=0, converters={'Period': str})
# check_col = ['13-18 months', 'more than 19 months']
# period_col='Period'
# # CT.remain_time_question5(remain_data, check_col, period_col)
#
# # Enquiries # Question 6
# enquiries_data = read_csv('enquiries_data.csv', header=0, converters={'Period': str})
# AC_enquiries = 'Enquiries to the CP Register of Aberdeen City'
# AS_enquiries = 'Enquiries to the CP Register of Aberdeenshire'
# MT_enquiries = 'Enquiries to the CP Register of Moray'
# period_col='Period'
# # CT.enquiries_question6(enquiries_data, AC_enquiries, AS_enquiries, MT_enquiries, period_col)

#Aberdeen city council CP
# UI.child_protection_UI()

# drug-related death
# col_names = ['years', 'drug related deaths', 'males', 'females', 'Deaths under age 14',
#              'Deaths between the ages of 15 and 24', 'Deaths between the ages of 25 and 34',
#              'Deaths between the ages of 35 and 44', 'Deaths between the ages of 45 and 54',
#              'Deaths between the ages of 55 and 64', 'Deaths ages over 65', 'average age of death',
#              'dead by Heroin/morphine 2', 'dead by Methadone', 'dead by Heroin/morphine, Methadone or Bupren-orphine',
#              'dead by Codeine or a codeine-containing compound',
#              'dead by Dihydro-codeine or a d.h.c-containing compound', 'dead by any opiate or opioid', ]
# data = read_csv("drugdeathsexagetype.csv", header=None, names=col_names)
#
# X = data.years  # Features
# Xcolname = "years"
# ycolnames = ['drug related deaths', 'males', 'females', 'Deaths under age 14',
#              'Deaths between the ages of 15 and 24', 'Deaths between the ages of 25 and 34',
#              'Deaths between the ages of 35 and 44', 'Deaths between the ages of 45 and 54',
#              'Deaths between the ages of 55 and 64', 'Deaths ages over 65', 'average age of death',
#              'dead by Heroin/morphine 2', 'dead by Methadone', 'dead by Heroin/morphine, Methadone or Bupren-orphine',
#              'dead by Codeine or a codeine-containing compound',
#              'dead by Dihydro-codeine or a d.h.c-containing compound', 'dead by any opiate or opioid', ]
#P1
# ycolname = "drug related deaths"
# y = data[ycolname]
# level = 1
# g = False
# b = False
# r2 = False
# p = False
# CT.segmentedregressionsummary_con(X, y, Xcolname, ycolname, level, g, b, r2, p, breakpointnum=5,governmentdrug=True, governmentchild=False)
# #P2
# y1name="males"
# y2name="females"
# y1=data[y1name]
# y2=data[y2name]
# begin=4
# end=24
# model="magnificationcompare"
# CT.dependentcompare_con(model, X, y1, y2, Xcolname, y1name, y2name, begin, end)

#P3
# col_names = ['Year', 'all drug-related deaths','more than one drug was found', 'only one drug was found', 'more than one drug was found in %','more than one drug was found to be present in the body','accidental poisonings']
# data = read_csv("onedrug.csv", header=None, names=col_names)
# Xcolname="Year"
# ycolname1 = "accidental poisonings"
# ycolname2 = "all drug-related deaths"
# X = data.Year  # Features
# m="independenttwopointcomparison"
# y1=data[ycolname1]
# y2=data[ycolname2]
# CT.independenttwopointcompare_con(m, X, Xcolname, y1, y2, ycolname1, ycolname2, point="", mode="mag")
# ycolname1 = "more than one drug was found to be present in the body"
# y1=data[ycolname1]
# CT.independenttwopointcompare_con(m, X, Xcolname, y1, y2, ycolname1, ycolname2, point="", mode="mag")

# P4
# col_names = ['Year', 'death by ‘street’ benzodiazepines (such as etizolam)',
#              'death by methadone', 'death by heroin/morphine',
#              'death by gabapentin and/or pregabalin',
#              'death by cocaine', 'death by opiates/opioids (such as heroin/morphine and methadone)',
#              'death by benzodiazepines (such as diazepam and etizolam)']
# data = read_csv("drugsubstancesimplicated.csv", header=None, names=col_names)
# X = data.Year  # Features
# Xcolname = "Year"
# ycolnames = ["death by ‘street’ benzodiazepines (such as etizolam)",
#              "death by methadone", "death by heroin/morphine",
#              "death by gabapentin and/or pregabalin",
#              "death by cocaine", "death by opiates/opioids (such as heroin/morphine and methadone)",
#              "death by benzodiazepines (such as diazepam and etizolam)"]
# begin=0
# end=numpy.size(X)-1
# model = 1
# category_name=" number of deaths where one or more of the following substances "
# y=data[ycolnames]
# CT.batchprovessing_con( model, X, y, Xcolname, ycolnames, category_name, end, begin=0)

# ### Child proction
# P1
# col_names = ['Year', 'children looked after in the community (%)',
#              'children looked after at home (%)', 'children looked after with kinship carers: friends/relatives (%)',
#              'children looked after with foster carers provided by LA (%)', 'children looked after with foster carers purchased by LA (%)',
#              'children looked after with prospective adopters (%)', 'children looked after in other community (%)',
#              'children looked after in residential care settings (%)',
#              'looked-after children','children looked after with foster carers (%)']
# data = read_csv("childtable1-1.csv", header=None, names=col_names)
# X = data.Year  # Features
# Xcolname = "Year"
# ycolnames = ['children looked after in the community (%)',
#              'children looked after at home (%)', 'children looked after with kinship carers: friends/relatives (%)',
#              'children looked after with foster carers provided by LA (%)', 'children looked after with foster carers purchased by LA (%)',
#              'children looked after with prospective adopters (%)', 'children looked after in other community (%)',
#              'children looked after in residential care settings (%)',
#              'looked-after children','children looked after with foster carers (%)']
# ycolname = "looked-after children"
# y=data[ycolname] # Target variable
# level = 1
# g = False
# b = False
# r2 = False
# p = False
# CT.segmentedregressionsummary_con(X, y, Xcolname, ycolname, level, g, b, r2, p,breakpointnum=5,governmentdrug=False, governmentchild=True)

# # #P2
# ycolname="children looked after at home (%)"
# y=data[ycolname]
# m="trendpercentage"
# CT.trendpercentage_con(m, X, y, Xcolname, ycolname, begin=3, end="")
#
# ycolname="children looked after with foster carers (%)"
# y=data[ycolname]
# m="trendpercentage"
# CT.trendpercentage_con(m, X, y, Xcolname, ycolname, begin=3, end="")
# #
# ycolname="children looked after in residential care settings (%)"
# y=data[ycolname]
# m="trendpercentage"
# CT.trendpercentage_con(m, X, y, Xcolname, ycolname, begin=3, end="")


# # P3
# col_names = ['years', 'Number of children starting to be looked after under age 1',
#              'Number of children starting to be looked after age between 1 to 4',
#              'Number of children starting to be looked after age between 5 to 11',
#              'Number of children starting to be looked after age between 12 to 15',
#              'Number of children starting to be looked after age between 16 to 17',
#              'Number of children starting to be looked after age between 18 to 21',
#              'Number of children starting to be looked after with unknown age',
#              'total number of children starting to be looked after',
#              'percent of children starting to be looked after under age 5']
# data = read_csv("numberofchildrenbyage.csv", header=None, names=col_names)
#
# X = data.years  # Features
# Xcolname = "years"
# ycolnames = ['Number of children starting to be looked after under age 1',
#              'Number of children starting to be looked after age between 1 to 4',
#              'Number of children starting to be looked after age between 5 to 11',
#              'Number of children starting to be looked after age between 12 to 15',
#              'Number of children starting to be looked after age between 16 to 17',
#              'Number of children starting to be looked after age between 18 to 21',
#              'Number of children starting to be looked after with unknown age',
#              'total number of children starting to be looked after',
#              'percent of children starting to be looked after under age 5']
# ycolname = "percent of children starting to be looked after under age 5"
# y=data[ycolname] # Target variable
# begin=0
# end=numpy.size(X)-1
# model="twopointpeak_child"
# CT.two_point_and_peak_child_con(model, X, y, Xcolname, ycolname, begin,end)


## An example for general template data story.
# features = read_csv('attributes.csv', delim_whitespace=True)
# dataset = read_csv('communities.data', names=features['attributes'])
# dataset = dataset.drop(columns=['state', 'county', 'community', 'communityname', 'fold'], axis=1)
# dataset = dataset.drop(columns=['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'], axis=1)
# dataset = CT.cleanData(dataset, 0.8)  # Clear data with a threshold of 80%

# CT.LinearModelStats(dataset, ['pctWPubAsst','PctHousLess3BR','PctPersOwnOccup'], 'ViolentCrimesPerPop', ['percentage of households with public assistance income','percent of housing units with less than 3 bedrooms','percent of people in owner occupied households'],'total number of violent crimes per 100K popuation', questionset=[1, 1, 1, 1], trend=1)