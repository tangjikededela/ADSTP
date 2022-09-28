from pandas import read_csv,DataFrame
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

# UI.child_protection_UI()
# drug-related death
col_names = ['years', 'drug related deaths', 'males', 'females', 'Deaths under age 14',
             'Deaths between the ages of 15 and 24', 'Deaths between the ages of 25 and 34',
             'Deaths between the ages of 35 and 44', 'Deaths between the ages of 45 and 54',
             'Deaths between the ages of 55 and 64', 'Deaths ages over 65', 'average age of death',
             'dead by Heroin/morphine 2', 'dead by Methadone', 'dead by Heroin/morphine, Methadone or Bupren-orphine',
             'dead by Codeine or a codeine-containing compound',
             'dead by Dihydro-codeine or a d.h.c-containing compound', 'dead by any opiate or opioid', ]
data = read_csv("drugdeathsexagetype.csv", header=None, names=col_names)

X = data.years  # Features
Xcolname = "years"
ycolnames = ['drug related deaths', 'males', 'females', 'Deaths under age 14',
             'Deaths between the ages of 15 and 24', 'Deaths between the ages of 25 and 34',
             'Deaths between the ages of 35 and 44', 'Deaths between the ages of 45 and 54',
             'Deaths between the ages of 55 and 64', 'Deaths ages over 65', 'average age of death',
             'dead by Heroin/morphine 2', 'dead by Methadone', 'dead by Heroin/morphine, Methadone or Bupren-orphine',
             'dead by Codeine or a codeine-containing compound',
             'dead by Dihydro-codeine or a d.h.c-containing compound', 'dead by any opiate or opioid', ]
y1name="males"
y2name="females"
y1=data[y1name]
y2=data[y2name]
begin=4
end=24
model="magnificationcompare"
CT.dependentcompare(model, X, y1, y2, Xcolname, y1name, y2name, begin, end)

