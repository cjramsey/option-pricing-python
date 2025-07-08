from abc import ABC, abstractmethod
from datetime import date, datetime
from numbers import Real

import numpy as np

from marketdata.market_data import MarketData

class Option(ABC):

    VALID_TYPES = {"call", "put"}

    def __init__(self, strike, expiry, option_type):
        if not isinstance(strike, Real) or strike < 0:
            raise TypeError(f"Parameter strike must be a +ve real value, got {strike} ({type(strike)}).")
        if not isinstance(expiry, Real) or expiry < 0:
            raise TypeError(f"Parameter expiry must be a +ve real value, got {expiry} ({type(expiry)}).")
        if option_type not in self.VALID_TYPES:
            raise ValueError(f"Parameter option_type must be one of {self.VALID_TYPES}, got {option_type}.")
        
        self.strike = strike
        self.expiry = expiry
        self.option_type = option_type
    
    @abstractmethod
    def payoff(self, *args, **kwargs):
        pass

    @property
    def early_exercise(self):
        return False

class Pricer(ABC):

    @property
    @abstractmethod
    def price(self, *args, **kwargs):
        pass

    @property
    def delta(self):
        raise NotImplementedError
    
    @property
    def theta(self):
        raise NotImplementedError
    
    @property
    def gamma(self):
        raise NotImplementedError
    
    @property
    def vega(self):
        raise NotImplementedError
    
    @property
    def rho(self):
        raise NotImplementedError
    
class HistoricalVolatilityModel(ABC):

    def __init__(self, data, ticker, start=None, end=None, window=252):
        if not isinstance(data, MarketData):
            raise TypeError(f"Parameter data must be a MarketData instance, got {data} ({type(data)}).")
        if not isinstance(ticker, str):
            raise TypeError(f"Parameter ticker must be a string, got {ticker}, ({type(ticker)}).")
        if not isinstance(start, (str, datetime, date)) and start is not None:
            raise TypeError(f"Parameter start must be a string, datetime or date, got {start}, ({type(start)}).")
        if not isinstance(end, (str, datetime, date)) and end is not None:
            raise TypeError(f"Parameter end must be a string, datetime or date, got {end}, ({type(end)}).")
        if not isinstance(window, (int, np.integer)):
            raise TypeError(f"Parameter window must be a positive integer, got {window} ({type(window)}).")
        if window < 1:
            raise ValueError(f"Parameter window must be >= 1, got {window}.")
        
        try:    
            self.start = datetime.strptime(start, "%Y-%m-%d")
        except ValueError:
            print("Parameter start does not match the format '%Y-%m-%d'.")
        except TypeError:
            if start:
                self.start = start.date()
            else:
                self.start = date(1980, 1, 1)
        
        try:    
            self.end = datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            print("Parameter end does not match the format '%Y-%m-%d'.")
        except TypeError:
            if end:
                self.end = end.date()
            else:
                self.end = date.today()
                
        self.data = data
        self.ticker = ticker
        self.window = window

    @abstractmethod
    def get_volatility(self):
        pass
    
    @abstractmethod
    def forecast_volatility(self, horizon):
        pass
    
class VolatilitySurfaceModel(ABC):

    @abstractmethod
    def get_volatility(self, strike, expiry):
        pass

    def plot_volatility_surface(self):
        raise NotImplementedError

class PathGenerator(ABC):

    def __init__(self, no_of_paths, no_of_steps):
        if not isinstance(no_of_paths, Real) or no_of_paths < 1:
            raise TypeError(f"Parameter no_of_paths must be >= 1, got {no_of_paths} ({type(no_of_paths)}).")
        if not isinstance(no_of_steps, Real) or no_of_steps < 0:
            raise TypeError(f"Parameter no_of_steps must be a +ve real value, got {no_of_steps} ({type(no_of_steps)}).")

        self.no_of_paths = no_of_paths
        self.no_of_steps = no_of_steps

    @abstractmethod
    def generate_paths(self):
        pass

    