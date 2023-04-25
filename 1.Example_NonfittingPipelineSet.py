from pandas import read_csv
import IntegratedPipeline as IP

# # NonfittingPipeline is mainly used for simple data presentation or exploration that is not for the purpose of model fitting.

# Read the dataset
goldprice_dataset = read_csv("./data/GoldPrice.csv", header=0)
# Let's set a pipeline without changing the variable names.
pipeline1 = IP.NonfittingPipeline(goldprice_dataset,["Date","OF_Open","DJ_open"],"Open")
# The below function will generate a basic dataset description based on the 'pipeline1' setting.
pipeline1.basic_description()

# Let's set another pipeline and change the variable names to more readable names. ("Data" to "data"; "Open" to "gold price")
pipeline2 = IP.NonfittingPipeline(goldprice_dataset,["Date"],"Open",["date"],"gold price")
# The following function will generate a simple heuristic description
# of the mean of the dependent variable and when the dependent variable reaches its maximum and minimum values,
# according to the "pipeline2" setting.
pipeline2.simple_timetrend()