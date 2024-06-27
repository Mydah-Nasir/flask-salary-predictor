# Import libraries
import numpy as np
import re
import string
import pandas as pd

from flask import Flask, request, render_template
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    A view for rendering results on HTML GUI
    '''
    # html -> py
    input_generator = request.form.values()

    # Gets values from form via POST request as a generator
    print(type(input_generator))

    # To get the value from a generator
    text = next(input_generator)
    Age = int(request.form['Age'])
    Annual_Income = int(request.form['Annual_Income'])
    Num_Bank_Accounts = int(request.form['Num_Bank_Accounts'])
    Num_Credit_Card = int(request.form['Num_Credit_Card'])
    Interest_Rate = int(request.form['Interest_Rate'])
    Num_of_Loan = int(request.form['Num_of_Loan'])
    Delay_from_due_date = int(request.form['Delay_from_due_date'])
    Num_of_Delayed_Payment = int(request.form['Num_of_Delayed_Payment'])
    Credit_Mix = int(request.form['Credit_Mix'])
    Outstanding_Debt = int(request.form['Outstanding_Debt'])
    Credit_Utilization_Ratio = int(request.form['Credit_Utilization_Ratio'])
    Credit_History_Age = int(request.form['Credit_History_Age'])
    Total_EMI_per_month = int(request.form['Total_EMI_per_month'])
    Payment_Behaviour = int(request.form['Payment_Behaviour'])
    Amount_invested_monthly = int(request.form['Amount_invested_monthly'])
    Monthly_Balance = int(request.form['Monthly_Balance'])
    # Credit_Score = int(request.form['Credit_Score'])
    Payment_of_Min_Amount = request.form['Payment_of_Min_Amount']
    if (Payment_of_Min_Amount == 'No'):
        PMA_NM = 0
        PMA_No = 1
        PMA_Yes = 0
    elif (Payment_of_Min_Amount == 'Yes'):
        PMA_NM = 0
        PMA_No = 0
        PMA_Yes = 1
    elif (Payment_of_Min_Amount == 'NM'):
        PMA_NM = 1
        PMA_No = 0
        PMA_Yes = 0

    # PMA_NM = int(request.form['PMA_NM'])
    # PMA_No = int(request.form['PMA_No'])
    # PMA_Yes = int(request.form['PMA_Yes'])

    # Age = next(input_generator) works as well
    print(type(text))
    scaler = StandardScaler()

    # Load a saved XGBoost model
    loaded_model = xgb.Booster()
    loaded_model.load_model('model.h5')
    datapoint = pd.DataFrame({'Age': [Age],
                              'Annual_Income': [Annual_Income], 'Num_Bank_Accounts': [Num_Bank_Accounts], 'Num_Credit_Card': [Num_Credit_Card],
                              'Interest_Rate': [Interest_Rate], 'Num_of_Loan': [Num_of_Loan],
                              'Delay_from_due_date': [Delay_from_due_date], 'Num_of_Delayed_Payment': [Num_of_Delayed_Payment],  'Credit_Mix': [Credit_Mix],
                              'Outstanding_Debt': [Outstanding_Debt], 'Credit_Utilization_Ratio': [Credit_Utilization_Ratio], 'Credit_History_Age': [Credit_History_Age],
                              'Total_EMI_per_month': [Total_EMI_per_month],
                              'Amount_invested_monthly': [Amount_invested_monthly], 'Payment_Behaviour': [Payment_Behaviour], 'Monthly_Balance': [Monthly_Balance],
                              'PMA_NM': [PMA_NM], 'PMA_No': [PMA_No], 'PMA_Yes': [PMA_Yes]})
    scalerfit = pd.DataFrame(scaler.fit_transform(
        datapoint), columns=datapoint.columns)
    dmatrix = xgb.DMatrix(scalerfit)
    prediction = loaded_model.predict(dmatrix)
    return render_template('index1.html', prediction_placeholder=prediction, age_value=Age)
    # py -> html
    # return render_template('index.html', prediction_placeholder=prediction)


if __name__ == "__main__":
    # debug=True means you won't have to run the server again & again, it'll update directly for you
    app.run(debug=True)
