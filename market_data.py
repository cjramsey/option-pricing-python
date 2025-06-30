import datetime
import os
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf

class MarketData:

    def __init__(self, tickers, save_stock_data=False, save_option_data=False):
        self.tickers = tickers
        self.save_stock_data = save_stock_data
        self.save_option_data = save_option_data
        self.stocks = {}
        self.calls = {}
        self.puts = {}

        if self.tickers:
            self.get_stock_data()
            self.get_options_data()
        else:
            self.load_data()

        if save_stock_data or save_option_data:
            self.save_data()

    def get_stock_data(self):
        for ticker in self.tickers:
            tk = yf.Ticker(ticker)
            df = tk.history(period="max")
            df.index = pd.to_datetime(df.index, utc=True)
            self.stocks.update({f"{ticker}" : df})

    def get_options_data(self):
        for ticker in self.tickers:
            tk = yf.Ticker(ticker)
            exps = tk.options
            calls, puts = pd.DataFrame(), pd.DataFrame()
            for e in exps:
                opt = tk.option_chain(e)
                opt.calls["expirationDate"] = e
                opt.puts["expirationDate"] = e
                calls = pd.concat([calls, opt.calls], ignore_index=True)
                puts = pd.concat([puts, opt.puts], ignore_index=True)

            drop_columns = ["change", "percentChange", "inTheMoney", 
                            "contractSize", "currency"]
            calls.drop(drop_columns, axis=1, inplace=True)
            puts.drop(drop_columns, axis=1, inplace=True)

            calls["lastTradeDate"] = pd.to_datetime(calls["lastTradeDate"], utc=True)
            puts["lastTradeDate"] = pd.to_datetime(puts["lastTradeDate"], utc=True)

            self.calls.update({ticker: calls})
            self.puts.update({ticker: puts})

    def save_data(self):
        base_dir = os.path.dirname(__file__)
        try:
            os.mkdir("data")
        except FileExistsError:
            pass
        if self.save_stock_data:
            for ticker in self.tickers:
                file_name = f"{ticker}_stock_{datetime.date.today()}.csv"
                file_path = os.path.join(base_dir, "data", file_name)
                self.stocks[ticker].to_csv(file_path)
            
        if self.save_option_data:
            for ticker in self.tickers:
                file_name = f"{ticker}_calls_{datetime.date.today()}.csv"
                file_path = os.path.join(base_dir, "data", file_name)
                self.calls[ticker].to_csv(file_path)

                file_name = f"{ticker}_puts_{datetime.date.today()}.csv"
                file_path = os.path.join(base_dir, "data", file_name)
                self.puts[ticker].to_csv(file_path)

    @staticmethod
    def parse_file_name(file_name, delimiter="_"):
        ticker, type, *date = file_name.split("_")
        return ticker, type, date

    def load_data(self, file_names=None):
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, "data")
        if not file_names:
            for entry in os.scandir(data_dir):
                if not entry.is_file():
                    continue
                df = pd.read_csv(entry.path, index_col=[0])
                ticker, type, *_ = self.parse_file_name(entry.name)
                if type == "stock":
                    self.stocks.update({ticker: df})
                elif type == "calls":
                    self.calls.update({ticker: df})
                elif type == "puts":
                    self.puts.update({ticker: df})
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
            elif type == "call":
                self.calls.update({ticker: df})
            elif type == "puts":
                self.puts.update({ticker: df})


def get_risk_free_rates(start=None, durations=None):
    start = start if start else datetime.datetime(2025, 5, 1)
    durations =  durations if durations else ["DGS1MO", "DGS3MO", "DGS6MO", 
                                              "GS1", "GS2", "GS3"]

    risk_free_rates = pdr.DataReader("DGS1MO", "fred", start)
    for duration in durations[1:]:
        risk_free_rates = pd.merge(risk_free_rates, pdr.DataReader(duration, "fred", start), 
                         left_index=True, right_index=True, how="outer")

    sofr = pdr.DataReader("SOFR", "fred", start)
    risk_free_rates = pd.merge(sofr, risk_free_rates, left_index=True, 
                               right_index=True, how="inner")

    risk_free_rates.rename(columns={"DGS1MO": 1/12, "DGS3MO": 1/4, "DGS6MO": 1/2, 
                                    "GS1": 1, "GS2": 2, "GS3": 4, "SOFR": 3/365}, 
                                    inplace=True)
    
    risk_free_rates = risk_free_rates.interpolate(method="linear").dropna().round(2)

    risk_free_rates.index = risk_free_rates.index.date
    
    return risk_free_rates

                   
def main():
    tickers = ["AAPL", "MSFT", "SPY", "NVDA", "PLTR",
               "AMZN", "GOOG", "TSLA", "TSM", "TQQQ"]
    data = MarketData([])

if __name__ == "__main__":
    main()

