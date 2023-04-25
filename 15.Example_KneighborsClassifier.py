import pandas as pd
import IntegratedPipeline as IP

# Load the dataset using pandas
irisdata = pd.read_csv('./data/Iris.csv',header=0)  # Replace 'your_dataset.csv' with the actual path to your dataset
Xcol=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
ycol='Species'
pipeline = IP.general_datastory_pipeline
pipeline.KNeighborsClassifierFit(irisdata,Xcol,ycol)