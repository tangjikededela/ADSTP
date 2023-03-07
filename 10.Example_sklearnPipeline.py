from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pandas
import ADSTP.IntegratedPipeline as IP

dataset=pandas.read_csv("./data/californiahousing.csv", header=0)
dataset.dropna(inplace=True)
Xcol=["housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
ycol="median_house_value"

skpipeline=Pipeline([('stdscaler', StandardScaler()),('MinMaxScaler', MinMaxScaler()),('linear', LinearRegression())])

IP.skpipeline_interpretation_con((skpipeline))
IP.skpipeline_questions_answer(skpipeline,dataset,Xcol,ycol)