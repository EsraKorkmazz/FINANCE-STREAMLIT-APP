import streamlit as st
from new2 import data_show
import pickle
import pandas as pd
import zipfile
import os
from src import transform_resp
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    menu_items={
        "About": "For More Information\n" + "https://github.com/EsraKorkmazz/FINANCE-STREAMLIT-APP"
    }
)

st.markdown(
    """
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/EsraKorkmazz/FINANCE-STREAMLIT-APP/)
    """
)

path = os.path.dirname(__file__)

@st.cache_resource
def unzip_load(name):
    try:
        path_zip = os.path.join(path, '../models/' + name + '.pkl.zip')
        
        if not os.path.exists(path_zip):
            st.error(f"Zip file not found: {path_zip}")
            return None
        
        with zipfile.ZipFile(path_zip, 'r') as zip_ref:
            zip_ref.extractall('models')
        
        path_obj = os.path.join('models', name + '.pkl')
        
        with open(path_obj, 'rb') as f:
            return pickle.load(f)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

st.sidebar.header("Menü")
menu = st.sidebar.radio(
    "Select an Option",
    ["HOME PAGE", "PORTFOLIO OPTIMIZATION", "CREDIT SCORE"],
    format_func=lambda x: {
        "HOME PAGE": "🏠 HOME PAGE",
        "PORTFOLIO OPTIMIZATION": "💲PORTFOLIO OPTIMIZATION",
        "CREDIT SCORE": "🏦 CREDIT SCORE"
    }[x]
)

if menu == "HOME PAGE":
    st.title("Welcome to Our Financial Dashboard!")
    st.write("""
    Welcome to our comprehensive financial dashboard, designed to provide you with the tools you need for effective financial management and investment decision-making.

    **Features:**

    - **Portfolio Optimization:** Dive into our Portfolio Optimization tool to balance risk and return according to your preferences.
    
    - **Credit Score Insights:** Explore insights and predictions related to credit scores.
    """)
    st.image("images/home.png", width=700)

elif menu == "PORTFOLIO OPTIMIZATION":
    st.title("Portfolio Optimization")
    st.markdown('''
                Portfolio optimization is about choosing the best mix of investments to maximize returns while minimizing risk. This app helps you optimize your portfolio based on the BIST30.
    ''')
    data_show()

elif menu == "CREDIT SCORE":
    st.title('Credit Score Analysis')
    best_model = unzip_load('best_model')

    st.markdown('''
                The purpose of our project is to allow the user to see how different factors affect their credit score.
    ''')

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

    st.header('Credit Score Form')
    age = st.slider('What is your age?', min_value=18, max_value=100, step=1)
    annual_income = st.number_input('What is your Annual Income?', min_value=0.00, max_value=300000.00)
    accounts = st.number_input('How many bank accounts do you have?', min_value=0, max_value=20, step=1)
    credit_cards = st.number_input('How many credit cards do you have?', min_value=0, max_value=12, step=1)
    delayed_payments = st.number_input('How many delayed payments do you have?', min_value=0, max_value=20, step=1)
    credit_card_ratio = st.slider('What is your credit card utilization ratio?', min_value=0.00, max_value=100.00)
    emi_monthly = st.number_input('How much EMI do you pay monthly?', min_value=0.00, max_value=5000.00)
    credit_history = st.number_input('How many months old is your credit history?', min_value=0, max_value=500, step=1)
    loans = st.multiselect('Which loans do you have?', ['Auto Loan', 'Credit-Builder Loan', 'Personal Loan',
                                                'Home Equity Loan', 'Mortgage Loan', 'Student Loan',
                                                'Debt Consolidation Loan', 'Payday Loan'])
    missed_payment = st.radio('Have you missed any payments in the last 12 months?', ['Yes', 'No'], index=0)
    minimum_payment = st.radio('Have you paid the minimum amount on at least one of your credit cards?', ['Yes', 'No'], index=0)

    run = st.button('Run the numbers!')

    st.header('Credit Score Results')
    
    col1, col2 = st.columns([3, 2])  
    with col2:
        x1 = [0, 6, 0]
        x2 = [0, 4, 0]
        x3 = [0, 2, 0]
        y = ['0', '-1', '1']

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

            credit_score = best_model.predict(output)[0]
            
            if credit_score == 1:
                st.balloons()
                t1 = plt.Polygon([[5, 0.5], [5.5, 0], [4.5, 0]], color='black')
                placeholder.markdown('Your credit score is **GOOD**! Congratulations!')
                st.markdown('This credit score indicates that this person is likely to repay a loan, so the risk of giving them credit is low.')
            elif credit_score == 0:
                t1 = plt.Polygon([[3, 0.5], [3.5, 0], [2.5, 0]], color='black')
                placeholder.markdown('Your credit score is **REGULAR**.')
                st.markdown('This credit score indicates that this person is likely to repay a loan, but can occasionally miss some payments. Meaning that the risk of giving them credit is medium.')
            elif credit_score == -1 :
                t1 = plt.Polygon([[1, 0.5], [1.5, 0], [0.5, 0]], color='black')
                placeholder.markdown('Your credit score is **POOR**.')
                st.markdown('This credit score indicates that this person is unlikely to repay a loan, so the risk of lending them credit is high.')
            f.gca().add_patch(t1)
            st.pyplot(f)

            with st.expander('Click to see how certain the algorithm was'):
                probabilities = best_model.predict_proba(output)[0]
                labels = ['Poor', 'Regular', 'Good']
                prob_fig = go.Figure(data=[go.Pie(labels=labels, values=probabilities)])
                prob_fig.update_layout(title_text='Prediction Probabilities')
                st.plotly_chart(prob_fig, use_container_width=True)

            with st.expander('Click to see feature importance'):
                importance = best_model.feature_importances_
                importance_fig = pd.DataFrame({'Importance': importance, 'Feature': ['Age', 'Annual Income', 'Accounts',
                                                                                    'Credit Cards', 'Delayed Payments', 
                                                                                    'Credit Card Ratio', 'EMI Monthly',
                                                                                    'Credit History', 'Loans',
                                                                                    'Missed Payment', 'Minimum Payment']})
                importance_fig.sort_values(by='Importance', ascending=True, inplace=True)

                importance_figure = go.Figure()
                importance_figure.add_trace(go.Bar(
                    y=importance_fig['Feature'],
                    x=importance_fig['Importance'],
                    orientation='h',
                    marker=dict(color='royalblue'),
                    text=importance_fig['Importance'].apply(lambda x: f'{x:.2f}%'),
                    textposition='inside'
                ))
                importance_figure.update_layout(title_text='Feature Importance', xaxis_title='Importance (%)')
                st.plotly_chart(importance_figure, use_container_width=True)

