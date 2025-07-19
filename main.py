import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from models.black_scholes import EuropeanBlackScholesPricer
from derivatives.vanilla import EuropeanOption
from models.monte_carlo import TerminalMonteCarloPricer
from models.path_generators import GeometricBrownianMotion
from models.binomial_tree import EuropeanBinomialTreePricer

tab = st.radio("Select Mode", ["Single Option", "Heatmap"])

if tab == "Heatmap":

    with st.sidebar:
        S = st.slider("Spot price", 75, 125, 100)
        r = st.slider("Risk-free rate", float(0), 0.1, 0.05)
        sigma = st.slider("Volatility", float(0), float(1), 0.4)

        option_type = st.selectbox("Option Type", ("Call", "Put"))
        option_map = {"Call": "call", "Put": "put"}

    plot_value = st.selectbox("Plotting variable",
                              ["Option Price", "Delta", "Gamma", "Theta", "Vega", "Rho"])

    time = np.arange(0.1, 1, 0.1)
    strike = np.arange(75, 125, 5)

    time_grid, strike_grid = np.meshgrid(time, strike, indexing="xy")

    def values(S, K, T, r, sigma, type):
        option = EuropeanOption(K, T, type)
        pricer = EuropeanBlackScholesPricer(option, S, sigma, r)
        plot_map = {"Option Price": pricer.price,
                    "Delta": pricer.delta,
                    "Gamma": pricer.gamma,
                    "Theta": pricer.theta,
                    "Vega": pricer.vega,
                    "Rho": pricer.rho}
        return plot_map[plot_value]

    vec_func = np.vectorize(values)

    prices = vec_func(S, strike_grid, time_grid, r, sigma, option_map[option_type])[::-1][::-1]
    st.title('Black-Scholes Model Heatmap')
    plt.figure(figsize=(10, 8))
    format = ".2f" if plot_value == "Option Price" else ".4f"
    sns.heatmap(prices, annot=True, fmt=format, cmap='viridis', xticklabels=np.round(time, 2), 
                yticklabels=np.flip(strike), cbar_kws={'label': plot_value})
    plt.xlabel("Time to expiration (years)")
    plt.ylabel("Strike price ($)")
    st.pyplot(plt)
    plt.close()


if tab == "Single Option":

    with st.sidebar:

        S = st.slider("Spot price", 75, 125, 100)
        r = st.slider("Risk-free rate", float(0), 0.1, 0.05)
        sigma = st.slider("Volatility", float(0), float(1), 0.4)
        strike = st.slider("Strike price", 75, 125, 100)
        expiry = st.slider("Time to expiration", 0.1, float(2), float(1))

        option_type = st.selectbox("Option Type", ("Call", "Put"))
        option_map = {"Call": "call", "Put": "put"}

    tabs = st.tabs(["Black-Scholes Model", "Binomial Tree", "Monte Carlo"])

    option = EuropeanOption(strike, expiry, option_map[option_type])

    with tabs[0]:
        pricer = EuropeanBlackScholesPricer(option, S, sigma, r)
        price = round(pricer.price, 4)
        a = st.columns(3)
        b = st.columns(3)
        a[0].metric("Option Price", price)
        cols = a[1:] + b

        for (greek, value), col in zip(pricer.greeks.items(), cols):
            rounded_value = round(value, 4)
            col.metric(f"{greek.capitalize()}", rounded_value)

    with tabs[1]:
        n = st.slider("Number of time steps", 1, 500, 10)
        pricer = EuropeanBinomialTreePricer(option, S, sigma, r, q=0, steps=n)
        price = round(pricer.price, 4)
        a = st.columns(3)
        b = st.columns(3)
        a[0].metric("Option Price", price)
        cols = a[1:] + b

        for (greek, value), col in zip(pricer.greeks.items(), cols):
            rounded_value = round(value, 4)
            col.metric(f"{greek.capitalize()}", rounded_value)
        
    with tabs[2]:
        N = st.slider("Number of paths", 1, 1000000, 10000)
        n = st.slider("Number of time steps", 1, 1000, 250)

        pricer = TerminalMonteCarloPricer(option, GeometricBrownianMotion(S, r, sigma, expiry, N, n), r, N, n)
        price = round(pricer.price, 4)

        st.metric("Option Price", price)
        
        
