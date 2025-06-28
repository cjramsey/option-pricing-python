import datetime
import os
import pandas as pd
import numpy as np
import yfinance as yf

class MarketData:

    def __init__(self, tickers, start=None, end=None, interval=None,
                save_stock_data=False, save_option_data=False):
        self.tickers = tickers
        self.start = start if start else datetime.datetime(2025, 1, 1)
        self.end = end if end else datetime.datetime.today()
        self.interval = interval
        self.save_stock_data = save_stock_data
        self.save_option_data = save_option_data
        self.stocks = self.get_stock_data()
        self.options = self.get_options_data()

        if save_stock_data or save_option_data:
            self.save_data()

    def get_stock_data(self):
        self.stocks = {}
        for ticker in self.tickers:
            tk = yf.Ticker(ticker)
            self.stocks.update({f"{ticker}" : tk.history(period="max")})
        return self.stocks

    def get_options_data(self):
        self.options = {}
        for ticker in self.tickers:
            tk = yf.Ticker(ticker)
            exps = tk.options
            options = pd.DataFrame()
            for e in exps:
                opt = tk.option_chain(e)
                opt = pd.concat([opt.calls, opt.puts])
                opt["expirationDate"] = e
                options = pd.concat([options, opt], ignore_index=True)
            options.drop(["change", "percentChange", "openInterest", 
                        "inTheMoney", "contractSize", "currency"], 
                        axis=1, inplace=True)
            self.options.update({ticker: options})
        return self.options

    def save_data(self):
        base_dir = os.path.dirname(__file__)
        try:
            os.mkdir("data")
        except FileExistsError:
            pass
        if self.save_stock_data:
            for ticker in self.tickers:
                file_name1 = f"{ticker}_stock_{datetime.date.today()}.csv"
                file_path1 = os.path.join(base_dir, "data", file_name1)
                self.stocks[ticker].to_csv(file_path1)
            
        if self.save_option_data:
            for ticker in self.tickers:
                file_name2 = f"{ticker}_options_{datetime.date.today()}.csv"
                file_path2 = os.path.join(base_dir, "data", file_name2)
                self.options[ticker].to_csv(file_path2)

    @staticmethod
    def parse_file_name(file_name, delimiter="_"):
        ticker, type, *date = file_name.split("_")
        return ticker, type, date

    def load_data(self, file_names=None):
        base_dir = os.path.dirname(__file__)
        if not file_names:
            for entry in os.scandir(base_dir):
                if entry.is_file():
                    df = pd.read_csv(entry.path, index_col=[0])
            return
            
        for file_name in file_names:
            path = os.path.join(base_dir, "data", file_name)
            try:
                df = pd.read_csv(path, index_col=[0])
            except FileNotFoundError:
                print(f"{file_name} not found.")
            else:
                ticker, type, *_ = self.parse_file_name(file_name)
            if type == "stock":
                self.stocks.update({ticker: df})
            elif type == "options":
                self.options.update({ticker: df})
                    

if __name__ == "__main__":
    pass


