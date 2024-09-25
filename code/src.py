import pandas as pd

def transform_resp(resp):
    def yes_no(value):
        if value not in ['Yes', 'No']:
            raise ValueError(f"Unexpected value: {value}. Expected 'Yes' or 'No'.")
        return 1 if value == 'Yes' else 0

    
    Debt_to_Income_Ratio = resp['emi_monthly'] * 12 / resp['annual_income'] if resp['annual_income'] > 0 else 0

    # Prepare output DataFrame
    output = {
        'Age': resp['age'],
        'Annual_Income': resp['annual_income'],
        'Num_Bank_Accounts': resp['accounts'],
        'Num_Credit_Card': resp['credit_cards'],
        'Num_of_Delayed_Payment': resp['delayed_payments'],
        'Credit_Utilization_Ratio': resp['credit_card_ratio'],
        'Total_EMI_per_month': resp['emi_monthly'],
        'Credit_History_Age_Formated': resp['credit_history'],
        'Mortgage_Loan': yes_no(resp['mortgage_loan']),
        'Missed_Payment_Day': yes_no(resp['missed_payment']),
        'Debt_to_Income_Ratio': Debt_to_Income_Ratio,
        'Payment_of_Min_Amount_Yes': yes_no(resp['minimum_payment'])
    }

    return pd.DataFrame([output])  # Return a DataFrame with one row.
