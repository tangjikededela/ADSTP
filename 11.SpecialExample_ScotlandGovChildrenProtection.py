from pandas import read_csv
import numpy
import ADSTP.IntegratedPipeline as CT

# # Set the pipelines
pipelines = CT.special_datastory_pipelines_for_Scottish_government_report

# # # Dataset of the number of children in care. Sort by place of care.
col_names = ['Year', 'children looked after in the community (%)',
             'children looked after at home (%)', 'children looked after with kinship carers: friends/relatives (%)',
             'children looked after with foster carers provided by LA (%)',
             'children looked after with foster carers purchased by LA (%)',
             'children looked after with prospective adopters (%)', 'children looked after in other community (%)',
             'children looked after in residential care settings (%)',
             'looked-after children', 'children looked after with foster carers (%)']
data = read_csv("./data/childtable1-1.csv", header=None, names=col_names)

# Q1. What is the trend in the number of children looked after from 2006 to 2019?
X = data.Year  # Features
Xcolname = "Year"
ycolname = "looked-after children"
y = data[ycolname]  # Target variable
pipelines.segmentedregression_fit(X, y, Xcolname, ycolname, level=1, graph=False, base=False, r2=False, p=False, breakpointnum=5, governmentdrug=False, governmentchild=True)
# Q2. In 2019, how has the number of children being cared for at home compared to 2009?
ycolname = "children looked after at home (%)"
y = data[ycolname]
m = "trendpercentage"
pipelines.trendpercentage_con(m, X, y, Xcolname, ycolname, begin=3, end="")
# Q3. In 2019, how has the number of children being cared for with foster carers compared to 2009?
ycolname = "children looked after with foster carers (%)"
y = data[ycolname]
m = "trendpercentage"
pipelines.trendpercentage_con(m, X, y, Xcolname, ycolname, begin=3, end="")
# Q4. In 2019, how has the number of children being cared for in residential care settings compared to 2009?
ycolname = "children looked after in residential care settings (%)"
y = data[ycolname]
m = "trendpercentage"
pipelines.trendpercentage_con(m, X, y, Xcolname, ycolname, begin=3, end="")

# # Dataset of the number of children in care. Sort by age.
col_names = ['years', 'Number of children starting to be looked after under age 1',
             'Number of children starting to be looked after age between 1 to 4',
             'Number of children starting to be looked after age between 5 to 11',
             'Number of children starting to be looked after age between 12 to 15',
             'Number of children starting to be looked after age between 16 to 17',
             'Number of children starting to be looked after age between 18 to 21',
             'Number of children starting to be looked after with unknown age',
             'total number of children starting to be looked after',
             'percent of children starting to be looked after under age 5']
data = read_csv("./data/numberofchildrenbyage.csv", header=None, names=col_names)

# Q5. Judging from the number of children under care under the age of 5,
# what are the trends in the age of the children under care in recent years?
X = data.years  # Features
Xcolname = "years"
ycolname = "percent of children starting to be looked after under age 5"
y = data[ycolname]  # Target variable
begin = 0
end = numpy.size(X) - 1
model = "twopointpeak_child"
pipelines.two_point_and_peak_child_con(model, X, y, Xcolname, ycolname, begin, end)
