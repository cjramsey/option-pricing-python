from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm

from market_data import MarketData, get_risk_free_rates
from options import EuropeanOption, AmericanOption
from black_scholes import EuropeanBlackScholesPricer
from binomial_tree import AmericanBinomialTreePricer

class ImpliedVolatilitySurface:

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


def main():
    data = MarketData([])
    x = ImpliedVolatilitySurface(data, "GOOG", "call", "American")
    print(x.option_data)
    x.plot_IV_surface(save_figure=True)

if __name__ == "__main__":
    main()
