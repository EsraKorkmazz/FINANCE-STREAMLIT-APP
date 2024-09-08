import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def data_show():
    bist100symbols = [
        "GARAN.IS", "KCHOL.IS", "THYAO.IS", "ISCTR.IS", "BIMAS.IS", "TUPRS.IS", "FROTO.IS",
        "AKBNK.IS", "ASELS.IS", "ENKAI.IS", "YKBNK.IS", "SASA.IS", "TCELL.IS", "CCOLA.IS",
        "SAHOL.IS", "VAKBN.IS", "TTKOM.IS", "EREGL.IS", "AEFES.IS", "SISE.IS", "TOASO.IS",
        "PGSUS.IS", "HALKB.IS", "ARCLK.IS", "MGROS.IS", "AGHOL.IS", "OYAKC.IS", "TAVHL.IS",
        "ASTOR.IS", "KOZAL.IS", "ENJSA.IS", "TTRAK.IS", "TURSG.IS", "ULKER.IS", "ISMEN.IS",
        "GUBRF.IS", "PETKM.IS", "BRSAN.IS", "BRYAT.IS", "DOAS.IS", "AKSEN.IS", "TABGD.IS",
        "ALARK.IS", "MAVI.IS", "DOHOL.IS", "EKGYO.IS", "AKSA.IS", "SOKM.IS", "ECILC.IS",
        "BTCIM.IS", "KONYA.IS", "EGEEN.IS", "TSKB.IS", "KONTR.IS", "REEDR.IS", "CIMSA.IS",
        "VESBE.IS", "HEKTS.IS", "ENERY.IS", "KCAER.IS", "SMRTG.IS", "CWENE.IS", "KRDMD.IS",
        "KOZAA.IS", "MIATK.IS", "ZOREN.IS", "VESTL.IS", "AKFYE.IS", "BFREN.IS", "ALFAS.IS",
        "KLSER.IS", "ECZYT.IS", "AGROT.IS", "GESAN.IS", "EUPWR.IS", "KLMSN.IS", "OSTIM.IS"
    ]
    
    # get stock data from yfinance
    data = yf.download(bist100symbols, start = '2022-01-01', end = '2024-01-01')['Adj Close']
    if data.empty:
        st.write("No data available.")
        return
 
    # daily returns
    returns = data.pct_change().dropna()
    
    # risk and return calculation
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    risk_tolerance = st.slider('Risk Tolerance Level (0 = Conservative, 100 = Aggressive)', 0, 100, 50)

    if 'optimized_weights' not in st.session_state:
        st.session_state.optimized_weights = np.zeros(len(bist100symbols))

    if st.button('Optimize Portfolio'):
        risk_aversion = 1 - (risk_tolerance / 100)

        def portfolio_objective(weights, mean_returns, cov_matrix, risk_aversion):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return - (portfolio_return - risk_aversion * portfolio_volatility)

        num_assets = len(bist100symbols)
        args = (mean_returns, cov_matrix, risk_aversion)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))

        initial_weights = num_assets * [1. / num_assets]

        opt_result = minimize(portfolio_objective, initial_weights, args=args,
                            method='SLSQP', bounds=bounds, constraints=constraints)

        st.session_state.optimized_weights = opt_result.x

        st.markdown("**Optimized Portfolio Weights:**")
        for i, stock in enumerate(bist100symbols):
            weight = float(st.session_state.optimized_weights[i])  
            if weight > 0:
                st.write(f"{stock}: {weight * 100:.2f}%")

    st.subheader("Expected Portfolio Metrics")
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
    if st.button('Show Visualizing Portfolio'):
        weights_df = pd.DataFrame(st.session_state.optimized_weights, index=bist100symbols, columns=['Weight'])
        fig = px.pie(weights_df, values='Weight', names=weights_df.index, title='Optimized Portfolio Weights')
        st.plotly_chart(fig)

    st.subheader("Efficient Frontier")
    if st.button('Show Efficient Frontier'):
        def efficient_frontier():
            num_portfolios = 10000
            results = np.zeros((3, num_portfolios))
            weights_record = []

            for i in range(num_portfolios):
                weights = np.random.random(len(bist100symbols))
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

