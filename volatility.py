from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
import os
import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, minimize, brentq
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm

from market_data import MarketData, get_risk_free_rates
from options import EuropeanOption, AmericanOption
from black_scholes import EuropeanBlackScholesPricer
from binomial_tree import AmericanBinomialTreePricer


class VolatilityModel(ABC):

    def __init__(self, data, ticker):
        self.data = data
        self.ticker = ticker

    @abstractmethod
    def model_volatility(self):
        pass

class ImpliedVolatilityModel:

    def __init__(self, data, ticker, type, style):
        self.data = data
        self.ticker = ticker
        self.type = type
        self.style = style
        self.stock_data = data.stocks[ticker]
        self.option_data = self.get_option_data()
        self.date = self.get_date()
        self.stock_price = self.get_stock_price()
        self.rates = self.get_rates()

        self.option_data = self.filter_data()
        self.add_risk_free_rate()
        self.get_IV()

    def get_option_data(self):
        if self.type == "call":
            return self.data.calls[self.ticker]
        elif self.type == "put":
            return self.data.puts[self.ticker]

    def get_date(self):
        self.option_data["lastTradeDate"] = pd.to_datetime(self.option_data["lastTradeDate"], utc=True)
        return max(self.option_data["lastTradeDate"]).date()

    def get_stock_price(self):
        self.stock_data.index = pd.to_datetime(self.stock_data.index, utc=True).date
        try:
            price = self.stock_data.loc[self.date].Close
        except KeyError:
            price = self.stock_data.iloc[-1].Close
        return price
    
    def get_rates(self):
        rates = get_risk_free_rates()
        return rates

    def filter_data(self):
        '''
        Removing deep in-the-money and out-of-the-money options,
        and stale options with no volume/open interest
        '''

        df = self.option_data

        df["lastTradeDate"] = pd.to_datetime(df["lastTradeDate"], utc=True).dt.date
        date = pd.Timestamp(max(self.option_data["lastTradeDate"]))
        df["T"] = (pd.to_datetime(df["expirationDate"]) - date).dt.days
        df["T"] = df["T"]/365

        time_filter = (df["T"] > 7/365) & (df["T"] < 2)
        df = df.loc[time_filter]

        moneyness_filter = np.abs((self.stock_price - df["strike"])/self.stock_price) < 0.2
        df = df.loc[moneyness_filter]

        stale_filter = (df['volume'] > 0) & (df['openInterest'] > 0)
        df = df.loc[stale_filter]

        df = df[(
            (df["ask"] + df["bid"])/2 > EuropeanBlackScholesPricer(
                EuropeanOption(df['strike'], df['T'], self.type),
                self.stock_price, 1e-8, 0.04).price
        )]
        
        return df
    
    @staticmethod
    def interpolate_rate(row, rates, date):
        try:
            rates = rates.loc[date - timedelta(days=1)]
        except KeyError:
            rates = rates.iloc[-1]
        x = rates.to_numpy()
        y = rates.index.to_numpy()
        T = row.loc['T']
        return np.interp(T, y, x).round(2)
    
    def add_risk_free_rate(self):
        rates = np.log(1 + self.rates/100)
        date = self.date
        self.option_data['riskFreeRate'] = self.option_data.agg(
            lambda row: self.interpolate_rate(row, rates, date), axis=1
        )

    @staticmethod
    def find_IV(row, price, type):
        def cost_func(sigma, price, type):
            S = price
            option = EuropeanOption(row["strike"], row["T"], type)
            pricer = EuropeanBlackScholesPricer(option, S, sigma, row['riskFreeRate'])
            model_price = pricer.price
            market_price = (row["ask"] + row["bid"])/2
            return (market_price - model_price)
        return brentq(cost_func, 1e-6, 5, args=(price, type), maxiter=500)
    
    def get_IV(self):
        self.option_data["estimatedIV"] = self.option_data.aggregate(
            func=(lambda row: self.find_IV(row, self.stock_price, self.type)), axis=1)


    def plot_IV_surface(self, save_figure=False):
        df = self.option_data
        x = df["strike"]/self.stock_price
        y = df["T"]
        z = df["estimatedIV"]

        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method="cubic")

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.plot_surface(xi, yi, zi, cmap=cm.winter)
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("Time to Maturity (Years)")
        ax.set_zlabel("Implied Volatility")
        ax.set_title(f"Implied Volatility Surface for {self.ticker} as of {self.date}")
        plt.show()

        if save_figure:
            try:
                os.mkdir('figures')
            except FileExistsError:
                pass
            base_dir = os.path.dirname(__file__)
            data_dir = os.path.join(base_dir, "figures")
            file_name = f"IV_surface_{self.ticker}_{self.date}.pdf"
            file_path = os.path.join(data_dir, file_name)
            plt.savefig(file_path)

class VolatilityModel:

    def __init__(self, data: MarketData, ticker, start=None, end=None):
        self.data = data
        self.ticker = ticker
        self.start = date(start) if start else date(2024,1,1)
        self.end = date(end) if start else date.today()

        self.select_data()

    def select_data(self):
        df = self.data.stocks[self.ticker]

        columns = ["Volume", "Dividends", "Stock Splits"]
        df.drop(columns=columns, inplace=True)

        df.index = pd.to_datetime(df.index, utc=True).date
        filter = (df.index > self.start) & (df.index < self.end)
        df.loc[filter]

        self.data = df 

    def simple_vol(self):
        df = self.data
        log_return = np.log(df["Close"]/df["Close"].shift(-1))
        volatility = log_return.rolling(window=252, center=False).std().dropna()
        return volatility
    
    def EWMA_vol(self, lambda_=0.94):
        df = self.data
        df["LogReturn"] = np.log(df["Close"]/df["Close"].shift(-1)).dropna()
        df["EWMA"] = (df["LogReturn"].rolling(window=252, center=False).std()).shift(1)
        df.dropna(axis=0, inplace=True)
        for i in range(2, len(df)):
            df.iloc[i, df.columns.get_loc("EWMA")] = np.sqrt((lambda_ * ((df.iloc[i-1, df.columns.get_loc("EWMA")])**2) 
                                    + (1 - lambda_) * (df.iloc[i-1, df.columns.get_loc("LogReturn")])**2))
        return df["EWMA"]
    
    def estimate_EWMA_param(self):
        df = self.data
        df["LogReturn"] = np.log(df["Close"]/df["Close"].shift(-1))
        df["EWMA"] = (df["LogReturn"].rolling(window=252, center=False).std()).shift(1)
        df.dropna(axis=0, inplace=True)
         
        def likelihood(lambda_, df):
            for i in range(2, len(df)):
                df.iloc[i, df.columns.get_loc("EWMA")] = np.sqrt((lambda_ * ((df.iloc[i-1, df.columns.get_loc("EWMA")])**2) 
                                        + (1 - lambda_) * (df.iloc[i-1, df.columns.get_loc("LogReturn")])**2))
            likelihood = -(-np.log(df["EWMA"]**2) - (df["LogReturn"]**2)/df["EWMA"]**2).sum()
            return likelihood 
        
        bounds = [(.8, .999)]
        lambda_ = minimize(likelihood, [0.95], args=(df), bounds=bounds).x[0]
        return lambda_
    
    def GARCH11_vol(self, omega, alpha, beta):
        df = self.data
        df["LogReturn"] = np.log(df["Close"]/df["Close"].shift(-1))
        df["stdev"] = df["LogReturn"].rolling(window=252, center=False).std().shift(1)
        df["vol"] = np.nan
        for i in range(2, len(df)):
                df.iloc[i, df.columns.get_loc("vol")] = np.sqrt(omega 
                                                         + alpha*(df.iloc[i-1, df.columns.get_loc("LogReturn")]**2) 
                                                         + beta*(df.iloc[i-1, df.columns.get_loc("stdev")]**2))
        return df["vol"]

    def estimate_GARCH11_params(self):
        df = self.data
        df["LogReturn"] = np.log(df["Close"]/df["Close"].shift(-1))
        df["stdev"] = df["LogReturn"].rolling(window=252, center=False).std().shift(1)
        df["vol"] = np.nan

        def likelihood(params, df):
            omega, alpha, beta = params
            for i in range(2, len(df)):
                df.iloc[i, df.columns.get_loc("vol")] = np.sqrt(omega/100000 
                                                         + (alpha/10)*(df.iloc[i-1, df.columns.get_loc("LogReturn")]**2) 
                                                         + beta*(df.iloc[i-1, df.columns.get_loc("stdev")]**2))
            likelihood = -(-np.log(df["vol"]) - (df["LogReturn"]**2)/df["vol"]**2).sum()
            return likelihood 
        
        constraint = NonlinearConstraint(lambda x: x[0]/100000 + x[1]/10 + x[2], 0.95, 1.05)
        
        bounds = [(0, 1), (0, 1), (0, 1)]
        omega, alpha, beta = minimize(likelihood, [1, 1, 0.9], args=(df), bounds=bounds, constraints=constraint).x
        return omega/100000, alpha/10, beta


def main():
    data = MarketData(["NVDA"])
    x = VolatilityModel(data, "NVDA")
    fig, ax = plt.subplots()
    ax.plot(x.EWMA_vol(x.estimate_EWMA_param()), label="EWMA")
    ax.plot(x.GARCH11_vol(*x.estimate_GARCH11_params()), label="GARCH(1,1)")
    ax.plot(x.simple_vol(), label="Unweighted")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
