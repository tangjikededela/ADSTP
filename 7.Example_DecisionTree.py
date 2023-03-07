import pandas
import ADSTP.IntegratedPipeline as IP

dataset=pandas.read_csv("./data/petrol_consumption.csv", header=0)
Xcol=["Petrol_tax","Average_income","Paved_Highways","Population_Driver_licence(%)"]
ycol="Petrol_Consumption"

pipeline = IP.general_datastory_pipeline
pipeline.DecisionTreeFit(dataset,Xcol,ycol,max_depth=3)
