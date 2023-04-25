import pandas as pd
import IntegratedPipeline as IP

data = pd.read_csv("./data/breastcancer.csv", header=0)
Xcol=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
ycol = 'diagnosis'
Class1='B'
Class2='M'
pipeline = IP.general_datastory_pipeline
pipeline.RidgeClassifierFit(data,Xcol,ycol,Class1,Class2)