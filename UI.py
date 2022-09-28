import tkinter as tk
from tkinter import filedialog
from pandas import read_csv
import numpy as np
import controller
from jupyter_dash import JupyterDash
from dash import Dash, html, dcc

# ______________________________________
def choose_data(titles='Dataset Select'):
    root = tk.Tk()
    root.title(titles)
    def get_file_path():
        global file_path
        global data_columns
        file_path = ''
        data_columns = ''
        # Open and return file path
        file_path = filedialog.askopenfilename(title="Select A File")
        tk.Label(root, text="File path: " + file_path).pack()

    def variables():
        if file_path == '':
            print("You didn't choose any of the files.")
        else:
            print(str(file_path))

    tk.Button(root, text="Open File", command=get_file_path).pack()
    tk.Button(root, text='Submit', command=variables).pack()
    destroy_button = tk.Button(root, text='Quit', command=root.destroy)
    destroy_button.pack()
    root.mainloop()

    dataset = read_csv(file_path, header=0)
    data_columns = dataset.columns.values
    return (dataset, data_columns)


# ______________________________
def single_variable_select(data_columns,titles='Single Variable Select'):
    ws = tk.Tk()
    ws.title(titles)
    tk.Label(ws, text='Please select the variable.').grid()
    # setting variable for Integers
    variable = tk.StringVar()
    variable.set(data_columns[0])
    # creating widget
    dropdown = tk.OptionMenu(
        ws,
        variable,
        *data_columns, )
    # positioning widget
    dropdown.grid()
    # infinite loop
    tk.Button(ws, text='Choose', command=ws.destroy).grid()
    tk.mainloop()
    the_variable = variable.get()
    return (the_variable)


# ~~~~~~~~~~~~~~~~~~~~
def multiple_variables_select(data_columns,titles='Multiple Variables Select'):
    master = tk.Tk()
    master.title(titles)
    var = [0] * np.size(data_columns)
    chosen_variables = [0] * np.size(data_columns)
    j = 1
    for i in range(np.size(data_columns)):
        if i % 6 == 0:
            var[i] = tk.IntVar()
            tk.Checkbutton(master, text=data_columns[i], variable=var[i]).grid(row=j, column=0, sticky='w')
        elif i % 6 == 1:
            var[i] = tk.IntVar()
            tk.Checkbutton(master, text=data_columns[i], variable=var[i]).grid(row=j, column=1, sticky='w')
        elif i % 6 == 2:
            var[i] = tk.IntVar()
            tk.Checkbutton(master, text=data_columns[i], variable=var[i]).grid(row=j, column=2, sticky='w')
        elif i % 6 == 3:
            var[i] = tk.IntVar()
            tk.Checkbutton(master, text=data_columns[i], variable=var[i]).grid(row=j, column=3, sticky='w')
        elif i % 6 == 4:
            var[i] = tk.IntVar()
            tk.Checkbutton(master, text=data_columns[i], variable=var[i]).grid(row=j, column=4, sticky='w')
        elif i % 6 == 5:
            var[i] = tk.IntVar()
            tk.Checkbutton(master, text=data_columns[i], variable=var[i]).grid(row=j, column=5, sticky='w')
            j = j + 1
    tk.Button(master, text='Quit', command=master.destroy).grid()
    tk.mainloop()
    m = 0
    for i in range(np.size(data_columns)):
        chosen_variables[i] = var[i].get()
        if chosen_variables[i] == 1:
            m = m + 1
    some_variables = [0] * m
    m = 0
    for i in range(np.size(data_columns)):
        if chosen_variables[i] == 1:
            some_variables[m] = data_columns[i]
            m = m + 1
    return (some_variables)
#________________________________________________

def start_app():
    app_name = JupyterDash(__name__)
    listTabs = []
    return (app_name,listTabs)

def run_app(app_name,listTabs):
    app_name.layout = html.Div([dcc.Tabs(listTabs)])
    app_name.run_server(mode='inline', debug=True)

def child_protection_UI():
    question_set = ['register question 1-2', 'riskfactor question 1-3', 'reregister question 4', 'remain time question 5',
                    'enquiries question 6', ]
    question_select = multiple_variables_select(question_set)
    print(question_select)

    app_name, listTabs = start_app()

    if question_set[0] in question_select:
        register_dataset, data_columns = choose_data()
        per1000inCity_col = multiple_variables_select(data_columns,'Please select the register per 1000 in city column')
        per1000nation_col = single_variable_select(data_columns,'Please select the register per 1000 in nation column')
        controller.register_question1(app_name, listTabs, register_dataset, per1000inCity_col, per1000nation_col)
    if question_set[1] in question_select:
        risk_factor_dataset, data_columns = choose_data()
        risk_factor_col = multiple_variables_select(data_columns,'Please select the risk factors')
        controller.riskfactor_question1(app_name, listTabs, risk_factor_dataset, risk_factor_col, cityname="Aberdeen City", max_num=5)
    if question_set[2] in question_select:
        register_dataset, data_columns = choose_data()
        reregister_col = single_variable_select(data_columns,'Please select the reregister number of a city')
        period_col = single_variable_select(data_columns,'Please select the period column')
        controller.re_register_question4(app_name, listTabs, register_dataset, reregister_col, period_col, national_average_reregistration= '13 - 16%')
    if question_set[3] in question_select:
        remain_data, data_columns = choose_data()
        check_col = multiple_variables_select(data_columns,'Please select after which remain time columns')
        period_col = single_variable_select(data_columns,'Please select the period column')
        controller.remain_time_question5(app_name, listTabs, remain_data, check_col, period_col)
    if question_set[4] in question_select:
        enquiries_data, data_columns = choose_data()
        AC_enquiries = single_variable_select(data_columns,'Please select the Aberdeen City enquiries column')
        AS_enquiries = single_variable_select(data_columns,'Please select the Aberdeenshire enquiries column')
        MT_enquiries=single_variable_select(data_columns,'Please select the Morty enquiries column')
        period_col = single_variable_select(data_columns,'Please select the period column')
        controller.enquiries_question6(app_name, listTabs, enquiries_data, AC_enquiries, AS_enquiries, MT_enquiries, period_col)

    run_app(app_name,listTabs)
