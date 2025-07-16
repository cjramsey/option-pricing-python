import datetime
import os
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf

class MarketData:
    '''
    Class for downloading, saving, and loading historical stock and option chain data
    using Yahoo Finance via the yfinance library.
    '''

    def __init__(self, tickers, save_stock_data=False, save_option_data=False):
        '''
        Constructor for MarketData objects. Fetches data or loads from disk depending
        on whether tickers are provided. Optionally saves data to disk.

        args:
            tickers (list[str]): list of ticker symbols to fetch.
            save_stock_data (bool): whether to save stock data.
            save_option_data (bool): whether to save option data.
        '''

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
        '''
        Downloads historical daily price data for each ticker using yfinance.
        Stores the data in the stocks attribute.
        '''

        for ticker in self.tickers:
            tk = yf.Ticker(ticker)
            df = tk.history(period="max")
            df.index = pd.to_datetime(df.index, utc=True)
            self.stocks.update({f"{ticker}" : df})

    def get_options_data(self):
        '''
        Downloads complete option chains (calls and puts) for all available
        expiration dates for each ticker using yfinance.

        Processes and cleans the data, storing in the calls and puts attributes.
        '''

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

    @staticmethod
    def parse_file_name(file_name, delimiter="_"):
        '''
        Parses a file name of the format {ticker}_{type}_{date}.csv.

        args:
            file_name (str): File name to parse.
            delimiter (str): Delimiter used in the file name (default is "_").

        returns:
            tuple[str, str, list[str]]: ticker, type ("stock", "calls", "puts"), and remainder (e.g. date).
        '''

        ticker, type, *date = file_name.split(delimiter)
        return ticker, type, date

    def save_data(self):
        '''
        Saves stock and/or option data to data/ directory.

        - Stock data: Overwrites existing stock CSVs for a ticker.
        - Option data: Saves calls and puts CSVs with today's date.

        Data is saved in the format: {ticker}_{type}_{YYYY-MM-DD}.csv
        '''

        base_dir = os.path.join(os.path.dirname(__file__), "..")
        data_dir = os.path.join(base_dir, "data")
        try:
            os.mkdir("data")
        except FileExistsError:
            pass
        if not self.save_stock_data:
            return
        
        for ticker in self.tickers:
            for entry in os.scandir(data_dir):
                if not entry.is_file():
                    continue
                file_ticker, type, *_ = self.parse_file_name(entry.name)
                if file_ticker == ticker and type == "stock" :
                    os.remove(entry.path)
            file_name = f"{ticker}_stock_{datetime.date.today()}.csv"
            file_path = os.path.join(data_dir, file_name)
            self.stocks[ticker].to_csv(file_path)
            
        if self.save_option_data:
            for ticker in self.tickers:
                file_name = f"{ticker}_calls_{datetime.date.today()}.csv"
                file_path = os.path.join(base_dir, "data", file_name)
                self.calls[ticker].to_csv(file_path)

                file_name = f"{ticker}_puts_{datetime.date.today()}.csv"
                file_path = os.path.join(base_dir, "data", file_name)
                self.puts[ticker].to_csv(file_path)

    def load_data(self, file_names=None):
        '''
        Loads stock and option data from CSV files in the data/ directory.

        If no file names are provided, loads all available data.
        Otherwise, loads the specified files.

        args:
            file_names (list[str], optional): Specific filenames to load.
        '''

        base_dir = os.path.join(os.path.dirname(__file__), "..")
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
    """
    Retrieves historical risk-free rates from FRED using pandas_datareader.

    Includes SOFR and various Treasury yields (e.g., 1M, 3M, 6M, 1Y, 2Y, 3Y).
    Interpolates missing values and renames columns with maturities in years.

    args:
        start (datetime.datetime, optional): Start date for data retrieval. Defaults to May 1, 2025.
        durations (list[str], optional): FRED codes for interest rate series.

    returns:
        pd.DataFrame: Risk-free rates indexed by date with columns as maturity in years.
    """

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
    tickers = ["AAPL", "AMZN", "GOOG", "MSFT", "NVDA",
               "PLTR", "SPY", "TQQQ", "TSLA", "TSM"]
    MarketData(tickers, save_option_data=True, save_stock_data=True)

if __name__ == "__main__":
    main()

