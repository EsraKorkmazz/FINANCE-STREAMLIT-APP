import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

def data_show():
    nasdaq_stocks = [
        'AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA', 'NFLX', 'NVDA', 'INTC', 'CSCO', 'CMCSA',
        'WBD', 'CSX', 'KHC', 'BKR', 'KDP', 'GFS', 'EXC', 'MNST', 'CPRT', 'MRNA',
        'ON', 'DXCM', 'FAST', 'DLTR', 'MRVL', 'MDLZ', 'CTSH', 'MCHP', 'FTNT', 'PYPL',
        'CSGP', 'AZN', 'CCEP', 'GILD', 'MU', 'GEHC', 'SBUX', 'PDD', 'AEP', 'TTD',
        'DDOG', 'ILMN', 'PAYX', 'DASH', 'EA', 'TTWO', 'ROST', 'AMD', 'TEAM', 'GOOG',
        'QCOM', 'PEP', 'AVGO', 'ZS', 'FANG', 'ODFL', 'BIIB', 'TMUS', 'TXN', 'HON',
        'CTAS', 'CDW', 'ADBE', 'AMGN', 'NKE', 'COST', 'INTU', 'ISRG', 'LRCX', 'ASML',
        'VRTX', 'REGN', 'BIDU', 'ZM', 'ADP'
    ]
    for attempt in range(3):
        try:
            data = yf.download(
                nasdaq_stocks,
                start='2023-01-01',
                end='2024-01-01',
                progress=False,
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True,
                timeout=30
            )
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    close_data = {}
                    for stock in nasdaq_stocks:
                        try:
                            if stock in data.columns.levels[0]:
                                stock_data = data[stock]['Close']
                                if len(stock_data.dropna()) > 200:
                                    close_data[stock] = stock_data
                        except:
                            continue
                    data = pd.DataFrame(close_data)
                else:
                    if 'Close' in data.columns:
                        data = data[['Close']].rename(columns={'Close': nasdaq_stocks[0]})
                break
            else:
                st.warning(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(2)

        except Exception as e:
            st.warning(f"Bulk download attempt {attempt + 1} failed: {str(e)}")
            if attempt == 2:
                raise e
            time.sleep(3)
    data = data.dropna(axis=1, how='all')

    if data.empty:
        st.error("All downloaded data is empty. Please check the stock symbols or data source.")
        return

    missing_stocks = set(nasdaq_stocks) - set(data.columns)
    if missing_stocks:
        st.warning(f"Failed to download data for: {', '.join(missing_stocks)}")

    returns = data.pct_change(fill_method=None).dropna()

    if returns.empty:
        st.error("Calculated returns data is empty. Check the downloaded data.")
        return

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
            threshold = 1e-5
            positive_weights = {
                data.columns[i]: w for i, w in enumerate(st.session_state.optimized_weights) if w > threshold
            }

            if positive_weights:
                portfolio_data = pd.DataFrame({
                    'Stock': list(positive_weights.keys()),
                    'Weight': list(positive_weights.values())
                })

                portfolio_data['Weight'] = portfolio_data['Weight'].apply(lambda x: f"{x*100:.2f}%")

                st.markdown("**Optimized Portfolio Weights:**")
                st.table(portfolio_data)

                csv_data = pd.DataFrame({
                    'Stock': list(positive_weights.keys()),
                    'Weight': list(positive_weights.values())
                }).to_csv(index=False)

                st.download_button(
                    label="Download Portfolio Weights as CSV",
                    data=csv_data,
                    file_name="optimized_portfolio_weights.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No stocks selected in the optimized portfolio.")

        else:
            st.error("Optimization failed. Check the constraints or bounds.")

    st.subheader("Expected Portfolio Metrics")
    st.markdown('''
                The "Expected Portfolio Metrics" section shows key indicators of your portfolio’s performance, including the expected return and volatility. The expected return represents the potential profit your portfolio might generate based on historical data, while volatility measures the risk or fluctuations in your portfolio's value over time. The graph helps visualize these metrics, giving you insight into the balance between risk and return for your chosen portfolio.
                ''')
    if st.button('Show Expected Portfolio Metrics'):
        portfolio_return = np.sum(mean_returns * st.session_state.optimized_weights) * 252
        portfolio_volatility = np.sqrt(np.dot(st.session_state.optimized_weights.T, np.dot(cov_matrix, st.session_state.optimized_weights))) * np.sqrt(252)
        st.write(f"Expected Portfolio Return: {portfolio_return:.2%}")
        st.write(f"Expected Portfolio Volatility: {portfolio_volatility:.2%}")
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

            max_sharpe_idx = np.argmax(results[2])
            max_sharpe_return = results[0,max_sharpe_idx]
            max_sharpe_volatility = results[1,max_sharpe_idx]

            min_vol_idx = np.argmin(results[1])
            min_vol_return = results[0,min_vol_idx]
            min_vol_volatility = results[1,min_vol_idx]

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

if __name__ == '__main__':
    st.title("NASDAQ Portfolio Optimizer")
    data_show()



