import controller as CT
import pandas

dataset=pandas.read_csv("./data/Maternal Health Risk Data Set.csv", header=0)
dataset['RiskLevel'].unique()
dataset['RiskLevel'] = dataset['RiskLevel'].replace('low risk', 0).replace('mid risk', 1).replace('high risk', 2)

Xcol=["Age","SystolicBP","DiastolicBP","BodyTemp","HeartRate","RiskLevel"]
ycol="BS"
pipeline = CT.general_datastory_pipeline
pipeline.GradientBoostingFit(dataset,Xcol,ycol)