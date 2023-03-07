# ADSTP:computer:
###### *Developed by - Ruilin Wang *
## About the application:
The automated data storytelling program (ADSTP) is a Python-based prototype that integrates many pipelines which transform data analysis results into data stories for automatic data interpretation. The prototype mainly aims to answer users' questions about the dataset and help users understand the dataset and model faster. It is also possible for the user to create some new pipelines. Since pipelines are reusable, once they are built, they are ideal for assisting in the production of monthly or annual reports.
### Main Features
| Features | Available Question  | Available at which model  |
| :---:   | :---: | :---: |
| Data Inference | Is the model credible?   | Linear set, Logistic set, Decision Tree set, Gradient Boosting set, Random Forest set, GAMs set  |
| Data Inference | Which independent variables are statistical significance?  | Linear set, Logistic set, GAMs set   |
| Data Prediction | What will be the effect on the dependent variable if an independent variable is continuously increased or decreased?   | Linear set, Logistic set, GAMs set   |
| Data Causality | Which independent variable is the most important variable that affects the dependent variable?   | Linear set, Logistic set, Decision Tree set, Gradient Boosting set, Random Forest set, GAMs set  |
| Data Causality | What are the conditions (the limit lines) for the classification?   | Logistic set, Decision Tree set, Gradient Boosting set, Random Forest set   |
| Data Description | What is the size of the dataset, and what are the independent and dependent variables?   | Non-fitting set   |
| Data Exploration | At what value of an independent variable does the dependent variable reach its maximum and minimum values?  | Non-fitting set   |
| Data Exploration | Which regression model fits the dataset best?  | PyCaret set   |
| Data Exploration | Which classification model fits the dataset best?  | PyCaret set   |
____
## System Requirements 
* Python version  - '3.6'
____

## Requirement

### The following packages are required to run the prototype:
```
pandas                          1.1.5
matplotlib                      3.3.2
numpy                           1.19.5
pwlf                            2.0.4
GPyOpt                          1.2.6
scikit-learn                    0.24.0
iteration-utilities             0.11.0
scipy                           1.5.4
pygam                           0.8.0
statsmodels                     0.12.1
numpy                           1.19.5
seaborn                         0.11.1
pydot                           1.4.2
yellowbrick                     1.3.post1
Jinja2                          3.0.3
jupyter-dash                    0.4.2
dash                            2.4.1
dash-bootstrap-components       1.1.0
plotly                          5.8.0
language-tool-python            2.7.1
```
____

## Quick Start
Here is a tutorial to help you quickly understand how ADSTP works.
### General data story for linear regression
There are several example.py files to show how the prototype works. Here will discuss one of the examples which use the linear regression model to fit some datasets as a quick start.

There are two use cases in *2_1.Example_LinearRegression.py*,  running any one of them will show a series of data stories on a dashboard that explain to the user whether the model is credible, whether the selected independent variables have a strong relationship with the dependent variable, how each independent variable affects the dependent variable and which independent variable is most important.

In fact, any user can use the following one-line command to generate a series of data stories by fitting a dataset with the linear regression model.

ADSTP.IntegratedPipeline.general_datastory_pipeline.LinearFit(*dataset, independent variables, dependent variable, more_readable_independent_variables_name="",more_readable_dependent_variable_name="", questionset=[1,1,1,1], trend=[0,1,1]*)

