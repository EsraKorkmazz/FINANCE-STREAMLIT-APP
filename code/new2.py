import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def data_show():
    nasdaq_stocks = [
    'AAPL',  # Apple Inc.
    'AMZN',  # Amazon.com, Inc.
    'GOOGL', # Alphabet Inc. (Class A)
    'MSFT',  # Microsoft Corporation
    'TSLA',  # Tesla, Inc.
    'NFLX',  # Netflix, Inc.
    'NVDA',  # NVIDIA Corporation
    'INTC',  # Intel Corporation
    'CSCO',  # Cisco Systems, Inc.
    'CMCSA', # Comcast Corporation
    'WBD',   # Warner Bros. Discovery, Inc.
    'CSX',   # CSX Corporation
    'KHC',   # The Kraft Heinz Company
    'BKR',   # Baker Hughes Company
    'KDP',   # Keurig Dr Pepper Inc.
    'GFS',   # GlobalFoundries Inc.
    'EXC',   # Exelon Corporation
    'MNST',  # Monster Beverage Corporation
    'CPRT',  # Copart, Inc.
    'XEL',   # Xcel Energy Inc.
    'MRNA',  # Moderna, Inc.
    'ON',    # ON Semiconductor Corporation
    'DXCM',  # DexCom, Inc.
    'FAST',  # Fastenal Company
    'DLTR',  # Dollar Tree, Inc.
    'MRVL',  # Marvell Technology, Inc.
    'MDLZ',  # Mondelez International, Inc.
    'CTSH',  # Cognizant Technology Solutions
    'MCHP',  # Microchip Technology Inc.
    'FTNT',  # Fortinet, Inc.
    'PYPL',  # PayPal Holdings, Inc.
    'CSGP',  # CoStar Group, Inc.
    'AZN',   # AstraZeneca PLC
    'CCEP',  # Coca-Cola Europacific Partners PLC
    'GILD',  # Gilead Sciences, Inc.
    'MU',    # Micron Technology, Inc.
    'GEHC',  # GE HealthCare Technologies Inc.
    'SBUX',  # Starbucks Corporation
    'PDD',   # PDD Holdings Inc.
    'AEP',   # American Electric Power Company
    'TTD',   # The Trade Desk, Inc.
    'DDOG',  # Datadog, Inc.
    'ABNB',  # Airbnb, Inc.
    'ILMN',  # Illumina, Inc.
    'PAYX',  # Paychex, Inc.
    'DASH',  # DoorDash, Inc.
    'ARM',   # Arm Holdings PLC
    'EA',    # Electronic Arts Inc.
    'TTWO',  # Take-Two Interactive Software, Inc.
    'ROST',  # Ross Stores, Inc.
    'AMD',   # Advanced Micro Devices, Inc.
    'TEAM',  # Atlassian Corporation
    'GOOG',  # Alphabet Inc. (Class C)
    'QCOM',  # Qualcomm Incorporated
    'PEP',   # PepsiCo, Inc.
    'AVGO',  # Broadcom Inc.
    'ZS',    # Zscaler, Inc.
    'FANG',  # Diamondback Energy, Inc.
    'ODFL',  # Old Dominion Freight Line, Inc.
    'BIIB',  # Biogen Inc.
    'TMUS',  # T-Mobile US, Inc.
    'TXN',   # Texas Instruments Incorporated
    'HON',   # Honeywell International Inc.
    'CTAS',  # Cintas Corporation
    'CDW',   # CDW Corporation
    'ADBE',  # Adobe Inc.
    'AMGN',  # Amgen Inc.
    'NKE',   # NIKE, Inc.
    'COST',  # Costco Wholesale Corporation
    'INTU',  # Intuit Inc.
    'ISRG',  # Intuitive Surgical, Inc.
    'LRCX',  # Lam Research Corporation
    'ASML',  # ASML Holding N.V.
    'VRTX',  # Vertex Pharmaceuticals Incorporated
    'REGN',  # Regeneron Pharmaceuticals, Inc.
    'BIDU',  # Baidu, Inc.
    'ZM',    # Zoom Video Communications, Inc.
    'ADP'    # Automatic Data Processing, Inc.
]


    # Get stock data from yfinance
    try:
        data = yf.download(nasdaq_stocks, start='2023-01-01', end='2024-01-01', progress=False)['Adj Close']
    except Exception as e:
        st.error(f"An error occurred: {e}")
        data = pd.DataFrame()

    # Remove columns with missing data
    data = data.dropna(axis=1, how='all')

    # Check if we have enough valid data
    if data.empty:
        st.error("All downloaded data is empty. Please check the stock symbols or data source.")
        return

    missing_stocks = set(nasdaq_stocks) - set(data.columns)
    if missing_stocks:
        st.warning(f"Failed to download data for: {', '.join(missing_stocks)}")

    # Daily returns
    returns = data.pct_change(fill_method=None).dropna()

    if returns.empty:
        st.error("Calculated returns data is empty. Check the downloaded data.")
        return

    # Risk and return calculation
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    st.markdown("Please choose your risk tolerance level and click the 'Optimize Portfolio' button!")
    risk_tolerance = st.slider('Risk Tolerance Level (0 = Conservative, 100 = Aggressive)', 0, 100, 50)

    if 'optimized_weights' not in st.session_state:
        st.session_state.optimized_weights = np.zeros(len(data.columns))

    if st.button('Optimize Portfolio'):
        risk_aversion = 1 - (risk_tolerance / 100)

        def portfolio_objective(weights, mean_returns, cov_matrix, risk_aversion):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return - (portfolio_return - risk_aversion * portfolio_volatility)

        num_assets = len(data.columns)
        args = (mean_returns, cov_matrix, risk_aversion)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))

        initial_weights = num_assets * [1. / num_assets]

        opt_result = minimize(portfolio_objective, initial_weights, args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)

        if opt_result.success:
            st.session_state.optimized_weights = opt_result.x
        else:
            st.error("Optimization failed. Check the constraints or bounds.")
            return

        st.markdown("**Optimized Portfolio Weights:**")
        num_stocks = len(data.columns)

        for i in range(num_stocks):
            weight = float(st.session_state.optimized_weights[i])
            if weight > 0:
                st.write(f"{data.columns[i]}: {weight * 100:.2f}%")

        # Create a dataframe for downloading using data.columns
        portfolio_data = pd.DataFrame({
            'Stock': data.columns,
            'Weight': st.session_state.optimized_weights[:len(data.columns)]
        })

        # Convert dataframe to CSV
        csv_data = portfolio_data.to_csv(index=False)

        # Provide the download link
        st.download_button(
            label="Download Portfolio Weights as CSV",
            data=csv_data,
            file_name="optimized_portfolio_weights.csv",
            mime="text/csv"
        )

    st.subheader("Expected Portfolio Metrics")
    st.markdown('''
                The "Expected Portfolio Metrics" section shows key indicators of your portfolio’s performance, including the expected return and volatility. The expected return represents the potential profit your portfolio might generate based on historical data, while volatility measures the risk or fluctuations in your portfolio's value over time. The graph helps visualize these metrics, giving you insight into the balance between risk and return for your chosen portfolio.
                ''')
    if st.button('Show Expected Portfolio Metrics'):
        portfolio_return = np.sum(mean_returns * st.session_state.optimized_weights) * 252
        portfolio_volatility = np.sqrt(np.dot(st.session_state.optimized_weights.T, np.dot(cov_matrix, st.session_state.optimized_weights))) * np.sqrt(252)
        st.write(f"Expected Portfolio Return: {portfolio_return:.2%}")
        st.write(f"Expected Portfolio Volatility: {portfolio_volatility:.2%}")
        
        # Plot Expected Portfolio Return vs Volatility
        plt.figure(figsize=(8, 6))
        plt.bar(['Return', 'Volatility'], [portfolio_return, portfolio_volatility], color=['blue', 'orange'])
        plt.ylabel('Value')
        plt.title('Expected Portfolio Return and Volatility')
        st.pyplot(plt)

    st.subheader("Visualizing Portfolio")
    st.markdown('''
                When you visualize your portfolio, you get a graphical representation of how your investments are distributed across different assets. This helps you understand your portfolio’s balance and the relative weight of each asset.
                ''')
    if st.button('Show Visualizing Portfolio'):
        weights_df = pd.DataFrame(st.session_state.optimized_weights, index=data.columns, columns=['Weight'])
        fig = px.pie(weights_df, values='Weight', names=weights_df.index, title='Optimized Portfolio Weights')
        st.plotly_chart(fig)

    st.subheader("Efficient Frontier")
    st.markdown('''
                The Efficient Frontier is a key concept in portfolio optimization. It shows the set of optimal portfolios that offer the highest expected return for a given level of risk. By plotting your portfolio against the efficient frontier, you can see how well your portfolio is performing compared to the best possible combinations of risk and return. The goal is to adjust your portfolio so that it lies on or near the efficient frontier for maximum efficiency.
                ''')
    if st.button('Show Efficient Frontier'):
        def efficient_frontier():
            num_portfolios = 10000
            results = np.zeros((3, num_portfolios))
            weights_record = []

            for i in range(num_portfolios):
                weights = np.random.random(len(data.columns))
                weights /= np.sum(weights)
                weights_record.append(weights)

                portfolio_return = np.sum(mean_returns * weights) * 252
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0

                results[0,i] = portfolio_return
                results[1,i] = portfolio_volatility
                results[2,i] = sharpe_ratio

            # Locate the portfolio with the highest Sharpe ratio
            max_sharpe_idx = np.argmax(results[2])
            max_sharpe_return = results[0,max_sharpe_idx]
            max_sharpe_volatility = results[1,max_sharpe_idx]

            # Locate the portfolio with the minimum volatility
            min_vol_idx = np.argmin(results[1])
            min_vol_return = results[0,min_vol_idx]
            min_vol_volatility = results[1,min_vol_idx]

            # Plot Efficient Frontier
            plt.figure(figsize=(10, 7))
            plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o')
            plt.colorbar(label='Sharpe Ratio')
            plt.scatter(max_sharpe_volatility, max_sharpe_return, color='r', marker='*', s=200, label='Max Sharpe Ratio')
            plt.scatter(min_vol_volatility, min_vol_return, color='g', marker='*', s=200, label='Min Volatility')
            plt.xlabel('Volatility')
            plt.ylabel('Return')
            plt.title('Efficient Frontier')
            plt.legend()
            st.pyplot(plt)

        efficient_frontier()

