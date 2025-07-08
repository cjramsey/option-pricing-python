from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from numbers import Real
import os

import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, minimize, brentq
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm

from core.interface import HistoricalVolatilityModel, VolatilitySurfaceModel
from marketdata.market_data import MarketData, get_risk_free_rates
from derivatives.vanilla import EuropeanOption, AmericanOption
from models.black_scholes import EuropeanBlackScholesPricer
from models.binomial_tree import AmericanBinomialTreePricer

class ConstantVolatility(VolatilitySurfaceModel):

    def __init__(self, sigma):
        if not isinstance(sigma, Real):
            raise TypeError(f"Parameter sigma must be a +ve real value, got {sigma} ({type(sigma)}).")
        if sigma < 0:
            raise ValueError(f"Parameter sigma must not be -ve, got {sigma}.")
        self.sigma = sigma

    def get_volatility(self):
        return self.sigma
    
class RollingUnweightedVolatilityModel(HistoricalVolatilityModel):

    def __init__(self, data, ticker, start=None, end=None, window=252):
        super().__init__(data, ticker, start, end, window)
        
        self.data = self.filter_data()

    def filter_data(self):
        df = self.data.stocks[self.ticker]
        drop_columns = ["Volume", "Dividends", "Stock Splits"]
        df.drop(columns=drop_columns, inplace=True)
        df.index = pd.to_datetime(df.index, utc=True).date
        filter = (df.index >= self.start) & (df.index <= self.end)
        return df.loc[filter]

    def get_volatility(self):
        df = self.data
        df["LogReturn"] = np.log(df["Close"]/df["Close"].shift(1))
        df["Volatility"] = df["LogReturn"].rolling(self.window, center=False).std()
        df.dropna(axis=0, inplace=True)
        return df["Volatility"]
    
    def forecast_volatility(self, horizon):
        return super().forecast_volatility(horizon)

class EWMAVolatilityModel(HistoricalVolatilityModel):

    def __init__(self, data, ticker, start=None, end=None, window=252):
        super().__init__(data,ticker, start, end, window)

        self.data = self.filter_data()

    def filter_data(self):
        df = self.data.stocks[self.ticker]
        drop_columns = ["Volume", "Dividends", "Stock Splits"]
        df.drop(columns=drop_columns, inplace=True)
        df.index = pd.to_datetime(df.index, utc=True).date
        filter = (df.index >= self.start) & (df.index <= self.end)
        return df.loc[filter]
    
    def get_volatility(self, lambda_=0.95):
        df = self.data.copy()
        df["LogReturn"] = np.log(df["Close"]/df["Close"].shift(1))
        df = df.iloc[1:]
        df.loc[:, "EWMA"] = (df["LogReturn"].rolling(window=252, center=False).std()).shift(-1)
        df.dropna(axis=0, inplace=True)

        for i in range(2, len(df)):
            df.iloc[i, df.columns.get_loc("EWMA")] = np.sqrt((lambda_ * ((df.iloc[i-1, df.columns.get_loc("EWMA")])**2) 
                                    + (1 - lambda_) * (df.iloc[i-1, df.columns.get_loc("LogReturn")])**2))
        return df["EWMA"]
    
    def forecast_volatility(self, horizon):
        return super().forecast_volatility(horizon)
    
    def estimate_param(self):
        df = self.data.copy()
        df["LogReturn"] = np.log(df["Close"]/df["Close"].shift(1))
        df = df.iloc[1:].copy()
        df.loc[:, "EWMA"] = (df["LogReturn"].rolling(window=252, center=False).std()).shift(-1)
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
    
class GARCHVolatilityModel(HistoricalVolatilityModel):

    def __init__(self, data, ticker, start=None, end=None, window=252):
        super().__init__(data, ticker, start, end)

        self.data = self.filter_data()

    def filter_data(self):
        df = self.data.stocks[self.ticker]
        columns = ["Volume", "Dividends", "Stock Splits"]
        df.drop(columns=columns, inplace=True)
        df.index = pd.to_datetime(df.index, utc=True).date
        filter = (df.index > self.start) & (df.index < self.end)
        return df.loc[filter]
    
    def calculate_volatility(self, omega=.000005, alpha=.1, beta=.9):
        df = self.data.copy()
        df["LogReturn"] = np.log(df["Close"]/df["Close"].shift(1))
        df["stdev"] = df["LogReturn"].rolling(window=252, center=False).std()
        df["vol"] = np.nan
        for i in range(2, len(df)):
                df.iloc[i, df.columns.get_loc("vol")] = np.sqrt(omega 
                                                         + alpha*(df.iloc[i-1, df.columns.get_loc("LogReturn")]**2) 
                                                         + beta*(df.iloc[i-1, df.columns.get_loc("stdev")]**2))
        df.dropna(axis=0, inplace=True)
        return df
    
    def get_volatility(self, omega=.000005, alpha=.1, beta=.9):
        df = self.calculate_volatility(omega, alpha, beta)
        return df["vol"]
    
    def estimate_params(self):
        df = self.data.copy()
        df["LogReturn"] = np.log(df["Close"]/df["Close"].shift(1))
        df["stdev"] = df["LogReturn"].rolling(window=252, center=False).std()
        df["vol"] = np.nan

        def likelihood(params, df):
            omega, alpha, beta = params
            for i in range(2, len(df)):
                df.iloc[i, df.columns.get_loc("vol")] = np.sqrt(omega
                                                         + (alpha)*(df.iloc[i-1, df.columns.get_loc("LogReturn")]**2) 
                                                         + beta*(df.iloc[i-1, df.columns.get_loc("stdev")]**2))
            likelihood = -(-np.log(df["vol"]) - (df["LogReturn"]**2)/df["vol"]**2).sum()
            return likelihood 
        
        constraint = NonlinearConstraint(lambda x:  x[1] + x[2], lb=0.9, ub=1)
        
        bounds = [(0, 1), (0, 1), (0, 1)]
        omega, alpha, beta = minimize(likelihood, [1e-5, 1, 1], args=(df), bounds=bounds, constraints=constraint).x
        return omega, alpha, beta
    
    def forecast_volatility(self, horizon):
        return super().forecast_volatility(horizon)
    
class ImpliedVolatilityModel(VolatilitySurfaceModel):

    VALID_TYPES = {"call", "put"}
    VALID_STYLES = {"european", "american"}

    def __init__(self, data, ticker, option_type, option_style="american"):
        if not isinstance(data, MarketData):
            raise TypeError(f"Parameter data must be a MarketData instance, got {data} ({type(data)}).")
        if not isinstance(ticker, str):
            raise TypeError(f"Parameter ticker must be a string, got {ticker}, ({type(ticker)}).")
        if option_type not in self.VALID_TYPES:
            raise ValueError(f"Parameter option_type must be one of {self.VALID_TYPES}, got {option_type}.")
        if option_style not in self.VALID_STYLES and option_style is not None:
            raise ValueError(f"Parameter option_style must be one of {self.VALID_STYLES}, got {option_style}.")
        
        self.data = data
        self.ticker = ticker
        self.option_type = option_type
        self.option_style = option_style

        self.option_data = self.get_option_data()
        self.stock_price = self.get_stock_price()
        self.risk_free_rates = get_risk_free_rates()
        
        self.option_data = self.filter_data()
        self.add_risk_free_rate()

    def get_option_data(self):
        if self.option_type == "call":
            return self.data.calls[self.ticker]
        elif self.option_type == "put":
            return self.data.puts[self.ticker]

    def get_date(self):
        self.option_data["lastTradeDate"] = pd.to_datetime(self.option_data["lastTradeDate"], utc=True)
        return max(self.option_data["lastTradeDate"]).date()

    def get_stock_price(self):
        self.data.stocks[self.ticker].index = pd.to_datetime(self.data.stocks[self.ticker].index, utc=True).date
        try:
            price = self.data.stocks[self.ticker].loc[self.get_date()].Close
        except KeyError:
            price = self.data.stocks[self.ticker].iloc[-1].Close
        return price
    
    def filter_data(self):
        '''
        Removing deep in-the-money and out-of-the-money options,
        and stale options with no volume/open interest
        '''

        df = self.option_data.copy()

        df["lastTradeDate"] = pd.to_datetime(df["lastTradeDate"], utc=True).dt.date
        date = pd.Timestamp(max(self.option_data["lastTradeDate"]))
        df["T"] = (pd.to_datetime(df["expirationDate"], utc=True) - date).dt.days
        df["T"] = df["T"]/365

        time_filter = (df["T"] > 5/365) & (df["T"] < 2)
        df = df.loc[time_filter]

        moneyness_filter = np.abs((self.stock_price - df["strike"])/self.stock_price) < 0.3
        df = df.loc[moneyness_filter]

        stale_filter = (df["volume"] > 0) | (df["openInterest"] > 0)
        df = df.loc[stale_filter]
        
        if self.option_style == "american":
            pricing_function = lambda row: AmericanBinomialTreePricer(AmericanOption(
                row["strike"], row["T"], self.option_type), self.stock_price, 1e-8, 0.04).control_variate_technique()
            filter  = (df["ask"] + df["bid"])/2 > df.agg(pricing_function, axis=1)
            df.loc[filter]
        else:
            filter = ((df["ask"] + df["bid"])/2 > EuropeanBlackScholesPricer(
                EuropeanOption(df["strike"], df["T"], self.option_type),
                self.stock_price, 1e-8, 0.04).price
                )
        
        return df
        

    @staticmethod
    def interpolate_rate(row, rates, date):
        try:
            rates = rates.loc[date - timedelta(days=1)]
        except KeyError:
            rates = rates.iloc[-1]
        x = rates.to_numpy()
        y = rates.index.to_numpy()
        T = row.loc["T"]
        return np.interp(T, y, x).round(2)
    
    def add_risk_free_rate(self):
        rates = np.log(1 + self.risk_free_rates/100)
        date = self.get_date()
        self.option_data["riskFreeRate"] = self.option_data.agg(
            lambda row: self.interpolate_rate(row, rates, date), axis=1
        )

    @staticmethod
    def find_IV(row, price, type):
        def cost_func(sigma, price, type):
            sigma = sigma[0]
            S = price
            option = EuropeanOption(row["strike"], row["T"], type)
            pricer = EuropeanBlackScholesPricer(option, S, sigma, row["riskFreeRate"])
            model_price = pricer.price
            market_price = (row["lastPrice"])
            return np.abs(market_price - model_price)
        bounds = [(1e-8, 5)]
        return minimize(cost_func, 0.3, args=(price, type), bounds=bounds).x
    
    def get_IV(self):
        self.option_data["estimatedIV"] = self.option_data.aggregate(
            func=(lambda row: self.find_IV(row, self.stock_price, self.option_type)), axis=1)
        
    def get_volatility(self, strike, expiry):
        return super().get_volatility(strike, expiry)
        
    def plot_volatility_surface(self, save_figure=False):
        df = self.option_data
        x = df["strike"]/self.stock_price
        y = df["T"]
        z = df["estimatedIV"]

        xi = np.linspace(x.min(), x.max(), 25)
        yi = np.linspace(y.min(), y.max(), 25)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method="cubic")

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.plot_surface(xi, yi, zi, cmap=cm.winter)
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("Time to Maturity (Years)")
        ax.set_zlabel("Implied Volatility")
        ax.set_title(f"Implied Volatility Surface for {self.ticker} as of {self.get_date()}")
        plt.show()

        if save_figure:
            try:
                os.mkdir("figures")
            except FileExistsError:
                pass
            base_dir = os.path.dirname(__file__)
            data_dir = os.path.join(base_dir, "figures")
            file_name = f"IV_surface_{self.ticker}_{self.get_date()}.pdf"
            file_path = os.path.join(data_dir, file_name)
            plt.savefig(file_path)


def main():
    x = ImpliedVolatilityModel(MarketData([]), "TSLA", "put")
    x.get_IV()
    x.plot_volatility_surface()

if __name__ == "__main__":
    main()
