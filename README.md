# ADSTP:computer:
###### *Developed by - Ruilin Wang *
## About the application:
An automated data storytelling program (ADSTP) is a Python-based prototype that integrates many pipelines which transform data analysis results into data stories for automatic data interpretation. The prototype mainly aims to answer users' questions about the dataset and help users understand the dataset and model faster. It is also possible for the user to create some new pipelines. Since pipelines are reusable, once they are built, they are ideal for assisting in the production of monthly or annual reports.
### Main Features
| Features | Available Question  | Available at which model  |
| :---:   | :---: | :---: |
| Data Inference | Is the model credible?   | Linear set, Logistic set, Decision Tree set, GAMs set  |
| Data Inference | Which independent variables are statistical significance?  | Linear set, Logistic set, GAMs set   |
| Data Prediction | What will be the effect on the dependent variable if an independent variable is continuously increased or decreased?   | Linear set, Logistic set, GAMs set   |
| Data Causality | Which independent variable is the most important variable that affect the dependent variable?   | Linear set, Logistic set, Decision Tree set, GAMs set  |
| Data Causality | What are the conditions (the limit lines) for the classification?   | Logistic set, Decision Tree set   |
| Data Description | What is the size of the dataset, and what are the independent and dependent variables?   | Non-fitting set   |
| Data Exploration | At what value of an independent variable does the dependent variable reach its maximum and minimum values?  | Non-fitting set   |
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
In the "usecase" file, there is a line called "An example for general template data story." This is an example which uses the linear regression model to fit the crime rate dataset.

Uncomment those lines after it and run will show a series of data stories on a dashboard that explain to the user whether it is credible to use the model, whether the selected independent variables have a strong relationship with the dependent variable, and how each independent variable affects the dependent variable.

In fact, any user can use the following one-line command to generate a data story for a linear model fit on any dataset.

*LinearModelStats(dataset, [indepdent variables], depdent variable, [Phrases or words to replace independent variable names],Phrases or words that replace the name of the dependent variable, questionset, trend)*

For the data story generation function of a linear model, it requies seven variables.  

| Variable | Variable Properties | Require and Effect |
| :----| ----- |:---- |
|  dataset   | Require a pandas.core.frame.DataFrame | which is the dataset input read by Pandas |
|  indepdent variables  | Require a list | It should include the indepdent variables names, usually the same as the header in the dataset |
| depdent variable  | Require a str | It is the depdent variable name, usually the same as the header in the dataset  |
| Phrases or words to replace independent variable names  | Require a list | It should have same length of indepdent variables. To make the data story more readable, the user can substitute the names of indepdent variables here. If left blank, the names in the header is used by default. |
| Phrases or words that replace the name of the dependent variable | Require a str | To make the data story more readable, the user can substitute the name of depdent variable here. If left blank, the name in the header is used by default. |
|  questionset | Require a list of length four | This array represents the type of question the user wants to know. Only 0 and 1 are allowed in it. 0 means not answering this question, and 1 means answering this question. For example, if user want to know all the questions, just use [1,1,1,1]. For linear models, four questions are currently prepared. 1. the credibility of the model. 2. whether each independent variable has a significant effect on the dependent variable. 3. under what circumstances each independent variable can make the dependent variable as large (or small) as possible. 4. the importance of each independent variable. |
|  trend | Require an int which should be 0 or 1 | It affects the focus of the description of the story. 0 means the user wants to make the dependent variable as large as possible, 1 means the user wants the dependent variable to be as small as possible  |
