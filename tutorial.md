# Overview:
This tutorial provides an overview of how you can use ADSTP in your data exploration workflow.

Note: This tutorial assumes that you have already cloned ADSTP, if you have not done so already, please check out this [page](https://github.com/tangjikededela/ADSTP).

# Quick Start Example
Before everything starts, please make sure that the Python version is 3.10.6, and Microsoft Visual C++ 14.0 or greater is required. Get it with: ["Microsoft C++ Build Tools"](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Then, please use the following commands to install the 'pygam' first:
```
pip install pygam
```
Then use the following commands to install other necessary packages:
```
pip install -r requirements.txt
```
Pandas is a powerful tool for data analysis and manipulation, and it is used for pre-work. To enable integrated pipelines in ADSTP, just create a new .py file in the ADSTP folder, simply add the import statement for ADSTP along with the Pandas import statement, without modifying existing Pandas code. (If the .py file created in another folder try to use the ADSTP, you need to pay attention to the import path.)
```
import pandas
import IntegratedPipeline
```
You could load the dataset via standard Pandas read_* commands. The following code will load a fish market dataset, you could check out this [page](https://www.kaggle.com/datasets/aungpyaeap/fish-market) to see the details of the dataset.
```
fish_dataset = pandas.read_csv('./data/fish.csv')
```
To use ADSTP to analyze your data and answer general questions based on the model, you should set the pipeline first as following code.
```
integrated_pipeline = IntegratedPipeline.general_datastory_pipeline
```
Then just choose the fitting algorithm, independent and dependent variables, a series of data analysis charts, tables and data stories to answer questions can be generated on the web page based on your selection. 
The following example uses a linear algorithm. The independent variables are length, diagonal, height and width of fish, and the dependent variable is the weight of fish.
```
Xcol = ['Length', 'Diagonal', 'Height', 'Width']
ycol = 'Weight' 
integrated_pipeline.LinearFit(fish_dataset, Xcol, ycol)
```
Run the above code, if successful, it will display "Dash is running on http://127.0.0.1:8050/"

Open this address, then you can see the content of a series of questions and answers to help you understand a linear regression model on the selected dataset.

You can also replace the “LinearFit” to other model such as “GAMsFit”, “GradientBoostingFit” etc. to see how well other models fit this dataset.
