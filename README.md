# ADSTP:computer:
###### *Developed by - Ruilin Wang *
### About the application:
Automated data storytelling program (ADSTP) is a Python-based prototype that integrates many pipelines which transform data analysis results into data stories for automatic interpretation. For general regression model interpretation, the system supports linear, logistic, gradient boosting, random forest, decision trees, GAMs, and segmented regression, it mainly aims to help users understand the dataset and model faster. It is also possible for the user to create a new pipeline for generating some special data stories, since pipelines are reusable, once they are built, they are ideal for assisting in the production of monthly or annual reports.

## System Requirements 
* Python version  - '3.6'
____

## Requirement

### The following packages are required to run the system:
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

Regardless of data preprocessing tasks (which is not the main task for ADSTP), any user can generate similar data stories for any dataset fitting by linear regression model with just one line of command as below.

*LinearModelStats(dataset, [indepdent variables], depdent variable, [Phrases or words to replace independent variable names],Phrases or words that replace the name of the dependent variable, questionset, trend)*

For the data story generation function of a linear model, it requies seven variables.  

| Variable | Variable Properties | Require and Effect |
| :----| ----- |:---- |
|  dataset   | Require a pandas.core.frame.DataFrame | which is the dataset input read by Pandas |
|  indepdent variables  | Require a list | It should include the indepdent variables names, usually the same as the header in the dataset |
| depdent variable  | Require a str | It is the depdent variable name, usually the same as the header in the dataset  |
| Phrases or words to replace independent variable names  | Require a list | It should have same size of indepdent variables. To make the data story more readable, the user can substitute the names of indepdent variables here. If left blank, the names in the header is used by default. |
| Phrases or words that replace the name of the dependent variable | Require a str | To make the data story more readable, the user can substitute the name of depdent variable here. If left blank, the name in the header is used by default. |
|  questionset | Require an array of length four | This array represents the type of question the user wants to know. Only 0 and 1 are allowed in it. 0 means not answering this question, and 1 means answering this question. For example, if user want to know all the questions, just use [1,1,1,1]. For linear models, four questions are currently prepared. 1. the credibility of the model. 2. whether each independent variable has a significant effect on the dependent variable. 3. under what circumstances each independent variable can make the dependent variable as large (or small) as possible. 4. the importance of each independent variable. |
|  trend | Require an int which should be 0 or 1 | It affects the focus of the description of the story. 0 means the user wants to make the dependent variable as large as possible, 1 means the user wants the dependent variable to be as small as possible  |
