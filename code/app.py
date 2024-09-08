import streamlit as st
from new2 import data_show
import pickle
import pandas as pd
from zipfile import ZipFile
import os
from src import transform_resp
import seaborn as sns 
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    menu_items={
        "About": "For More Information\n" + "https://github.com/EsraKorkmazz/FINANCE-STREAMLIT-APP"
    }
)

path = os.path.dirname(__file__)
folder_path = os.path.join(path,'../models')
@st.cache_data
def unzip_load(name):
    try:
        # Path to the zip file
        path_zip = os.path.join(path, '../models/' + name + '.pkl.zip')
        
        # Extract the zip file
        with ZipFile(path_zip, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        
        # Path to the extracted .obj (or .pkl) file
        path_obj = os.path.join(folder_path, name + '.pkl')
        
        # Load the pickled object
        with open(path_obj, 'rb') as f:
            return pickle.load(f)
    
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

st.sidebar.header("Menü")
menu = st.sidebar.radio(
    "Select an Option",
    ["HOME PAGE", "PORTFOLIO OPTIMIZATION", "CREDIT SCORE"],
    format_func=lambda x: {"HOME PAGE": "🏠 HOME PAGE","PORTFOLIO OPTIMIZATION": "💲PORTFOLIO OPTIMIZATION","CREDIT SCORE": "🏦 Credit Score"}[x]
)
if menu == "HOME PAGE":
    st.title("Welcome to Our Financial Dashboard!")
    st.write("""
    Welcome to our comprehensive financial dashboard, designed to provide you with the tools you need for effective financial management and investment decision-making.

    **Features:**

    - **Portfolio Optimization:** Dive into our Portfolio Optimization tool to balance risk and return according to your preferences. Whether you are conservative or aggressive in your investment approach, our tool helps you optimize your portfolio with stocks from the BIST 100 index. Adjust your risk tolerance and let our system suggest the optimal stock weights for you.

    - **Credit Score Insights:** Explore insights and predictions related to credit scores. This tool aims to help you understand and manage your credit health better.
    """)
    st.image("/Users/esra/Desktop/FINANCE-STREAMLIT-APP/images/home.png",  width=700)

elif menu == "PORTFOLIO OPTIMIZATION":
    st.title("Portfolio Optimization Page")
    data_show()

elif menu == "CREDIT SCORE":
    st.title('Credit Score Analysis')
    scaler = unzip_load('scaler')
    model = unzip_load('model')    

    age_default = None
    annual_income_default = 0.00
    accounts_default = 0
    credit_cards_default = 0
    delayed_payments_default = 0
    credit_card_ratio_default = 0.00
    emi_monthly_default = 0.00
    credit_history_default = 0
    loans_default = None
    missed_payment_default = 0
    minimum_payment_default = 0

    st.markdown('''
                The purpose of our project is to allow the user to see how different factors affect their credit score. The project predicts the credit score using the data entered by the user through a form. Users can fill out the form to calculate their credit score. All data is temporary and not saved. For more information, you can check the project repository on GitHub.
    ''')

    st.header('Credit Score Form')
    age = st.slider('What is your age?', min_value=18, max_value=100, step=1, value=age_default)
    annual_income = st.number_input('What is your Annual Income?', min_value=0.00, max_value=300000.00, value=annual_income_default)
    accounts = st.number_input('How many bank accounts do you have?', min_value=0, max_value=20, step=1, value=accounts_default)
    credit_cards = st.number_input('How many credit cards do you have?', min_value=0, max_value=12, step=1, value=credit_cards_default)
    delayed_payments = st.number_input('How many delayed payments do you have?', min_value=0, max_value=20, step=1, value=delayed_payments_default)
    credit_card_ratio = st.slider('What is your credit card utilization ratio?', min_value=0.00, max_value=100.00, value=credit_card_ratio_default)
    emi_monthly = st.number_input('How much EMI do you pay monthly?', min_value=0.00, max_value=5000.00, value=emi_monthly_default)
    credit_history = st.number_input('How many months old is your credit history?', min_value=0, max_value=500, step=1, value=credit_history_default)
    loans = st.multiselect('Which loans do you have?', ['Auto Loan', 'Credit-Builder Loan', 'Personal Loan',
                                                'Home Equity Loan', 'Mortgage Loan', 'Student Loan',
                                                'Debt Consolidation Loan', 'Payday Loan'], default=loans_default)
    missed_payment = st.radio('Have you missed any payments in the last 12 months?', ['Yes', 'No'], index=missed_payment_default)
    minimum_payment = st.radio('Have you paid the minimum amount on at least one of your credit cards?', ['Yes', 'No'], index=minimum_payment_default)

    run = st.button( 'Run the numbers!')

    st.header('Credit Score Results')

    col1, col2 = st.columns([3, 2])

    with col2:
        x1 = [0, 6, 0]
        x2 = [0, 4, 0]
        x3 = [0, 2, 0]
        y = ['0', '1', '2']

        f, ax = plt.subplots(figsize=(5,2))

        p1 = sns.barplot(x=x1, y=y, color='#3EC300')
        p1.set(xticklabels=[], yticklabels=[])
        p1.tick_params(bottom=False, left=False)
        p2 = sns.barplot(x=x2, y=y, color='#FAA300')
        p2.set(xticklabels=[], yticklabels=[])
        p2.tick_params(bottom=False, left=False)
        p3 = sns.barplot(x=x3, y=y, color='#FF331F')
        p3.set(xticklabels=[], yticklabels=[])
        p3.tick_params(bottom=False, left=False)

        plt.text(0.7, 1.05, "POOR", horizontalalignment='left', size='medium', color='white', weight='semibold')
        plt.text(2.5, 1.05, "REGULAR", horizontalalignment='left', size='medium', color='white', weight='semibold')
        plt.text(4.7, 1.05, "GOOD", horizontalalignment='left', size='medium', color='white', weight='semibold')

        ax.set(xlim=(0, 6))
        sns.despine(left=True, bottom=True)

        figure = st.pyplot(f)

    with col1:

        placeholder = st.empty()

        if run:
            resp = {
                'age': age,
                'annual_income': annual_income,
                'accounts': accounts,
                'credit_cards': credit_cards,
                'delayed_payments': delayed_payments,
                'credit_card_ratio': credit_card_ratio,
                'emi_monthly': emi_monthly,
                'credit_history': credit_history,
                'loans': loans,
                'missed_payment': missed_payment,
                'minimum_payment': minimum_payment
            }
            output = transform_resp(resp)
            output = pd.DataFrame(output, index=[0])
            output.loc[:,:] = scaler.transform(output)

            credit_score = model.predict(output)[0]
            
            if credit_score == 1:
                st.balloons()
                t1 = plt.Polygon([[5, 0.5], [5.5, 0], [4.5, 0]], color='black')
                placeholder.markdown('Your credit score is **GOOD**! Congratulations!')
                st.markdown('This credit score indicates that this person is likely to repay a loan, so the risk of giving them credit is low.')
            elif credit_score == 0:
                t1 = plt.Polygon([[3, 0.5], [3.5, 0], [2.5, 0]], color='black')
                placeholder.markdown('Your credit score is **REGULAR**.')
                st.markdown('This credit score indicates that this person is likely to repay a loan, but can occasionally miss some payments. Meaning that the risk of giving them credit is medium.')
            elif credit_score == -1:
                t1 = plt.Polygon([[1, 0.5], [1.5, 0], [0.5, 0]], color='black')
                placeholder.markdown('Your credit score is **POOR**.')
                st.markdown('This credit score indicates that this person is unlikely to repay a loan, so the risk of lending them credit is high.')
            plt.gca().add_patch(t1)
            figure.pyplot(f)
            prob_fig, ax = plt.subplots()

            with st.expander('Click to see how certain the algorithm was'):
                plt.pie(model.predict_proba(output)[0], labels=['Poor', 'Regular', 'Good'], autopct='%.0f%%')
                st.pyplot(prob_fig)
            
            with st.expander('Click to see how much each feature weight'):
                importance = model.feature_importances_
                importance = pd.DataFrame(importance)
                columns = pd.DataFrame(['Age', 'Annual_Income', 'Num_Bank_Accounts',
                                        'Num_Credit_Card', 'Num_of_Delayed_Payment',
                                        'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                                        'Credit_History_Age_Formated', 'Auto_Loan',
                                        'Credit-Builder_Loan', 'Personal_Loan', 'Home_Equity_Loan',
                                        'Mortgage_Loan', 'Student_Loan', 'Debt_Consolidation_Loan',
                                        'Payday_Loan', 'Missed_Payment_Day', 'Payment_of_Min_Amount_Yes'])

                importance = pd.concat([importance, columns], axis=1)
                importance.columns = ['importance', 'index']
                importance_fig = round(importance.set_index('index')*100.00, 2)
                loans = ['Auto_Loan', 'Credit-Builder_Loan', 'Personal_Loan', 
                        'Home_Equity_Loan', 'Mortgage_Loan', 'Student_Loan',
                        'Debt_Consolidation_Loan', 'Payday_Loan']

                # summing the loans
                Loans = importance_fig.loc[loans].sum().reset_index()
                Loans['index'] = 'Loans'
                Loans.columns=['index','importance']
                importance_fig = importance_fig.drop(loans, axis=0).reset_index()
                importance_fig = pd.concat([importance_fig, Loans], axis=0)
                importance_fig.sort_values(by='importance', ascending=True, inplace=True)

                # plotting the figure
                importance_figure, ax = plt.subplots()
                bars = ax.barh('index', 'importance', data=importance_fig)
                ax.bar_label(bars)
                plt.ylabel('')
                plt.xlabel('')
                plt.xlim(0,20)
                sns.despine(right=True, top=True)
                st.pyplot(importance_figure)