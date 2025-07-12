from abc import ABC, abstractmethod
from datetime import date, datetime
from numbers import Real

import numpy as np

from marketdata.market_data import MarketData

class Option(ABC):
    '''Abstract base class for an option object.'''

    VALID_TYPES = {"call", "put"}

    def __init__(self, strike, expiry, option_type):
        '''
        Constructor for Option instances.

        Attributes:
            strike (float): strike price of option.
            expiry (float): time to expiration of option.
            option_type (str): either "call" or "put".
        '''

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
        '''
        Calculate payoff for option given spot/path.
        Must be implemented by subclasses.
        '''
        pass

    @property
    def early_exercise(self):
        '''Defines whether early exercise is allowed.'''
        return False

class Pricer(ABC):
    '''Abstract base class for option pricers.'''

    @property
    @abstractmethod
    def price(self, *args, **kwargs):
        '''
        Calculates price of the option give parameters relevant to model being used.
        Must be implemented by subclasses.
        '''
        pass

    @property
    def delta(self):
        '''Change in option price with respect to change in spot price.'''
        raise NotImplementedError
    
    @property
    def theta(self):
        '''Change in option price with respect to change in time.'''
        raise NotImplementedError
    
    @property
    def gamma(self):
        '''Change in delta with respect to change in spot price.'''
        raise NotImplementedError
    
    @property
    def vega(self):
        '''Change in option price with respect to change in volatility.'''
        raise NotImplementedError
    
    @property
    def rho(self):
        '''Change in option price with respect to change in risk-free-rate.'''
        raise NotImplementedError
    
class HistoricalVolatilityModel(ABC):
    '''Abstract base class for historical volatility models.'''

    def __init__(self, data, ticker, start=None, end=None, window=252):
        '''
        Constructor for HistoricalVolatilityModel objects.

        Attributes:
            data (MarketData): a MarketData object containing historical stock prices.
            ticker (str): ticker for which historical data is used.
            start (str, date/datetime): start date for historical data.
            end (str, date/datetime): end date for historical data.
            window (int): number of periods for rolling window.
        '''

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
        
        # converting start to date object, defaults to 1980-01-01
        try:    
            self.start = datetime.strptime(start, "%Y-%m-%d")
        except ValueError:
            print("Parameter start does not match the format '%Y-%m-%d'.")
        except TypeError:
            if start:
                self.start = start.date()
            else:
                self.start = date(1980, 1, 1)
    
        # converting start to date object, defaults to today
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
    def get_volatility(self, *args, **kwargs):
        '''
        Calculates volatility based off historical data.
        Must be implemented by subclasses.
        '''
        pass
    
    @abstractmethod
    def forecast_volatility(self, horizon):
        '''
        Predicts volatility at specific time in the future.
        Must be implemented by subclasses.

        args:
            horizon (float): Time ahead for which to predict volatility.
        '''
        pass
    
class VolatilitySurfaceModel(ABC):
    '''Abstract base class for volatility surface models.'''

    @abstractmethod
    def get_volatility(self, strike, expiry):
        '''
        Interpolate volatility from surface from specified strike and expiry.
        Must be implemented by subclasses.
        
        args:
            strike (float): strike price.
            expiry (float): time to expiration.
        '''
        pass

    def plot_volatility_surface(self):
        '''Plot 3D volatility surface.'''
        raise NotImplementedError

class PathGenerator(ABC):
    '''Abstract base class for path generators.'''

    def __init__(self, no_of_paths, no_of_steps):
        '''
        Constructor for PathGenerator objects.

        args:
            no_of_paths (int): number of simulated paths to generate.
            no_of_steps (int): number of time steps in a single path.
        '''

        if not isinstance(no_of_paths, np.integer) or no_of_paths < 1:
            raise TypeError(f"Parameter no_of_paths must be an integer >= 1, got {no_of_paths} ({type(no_of_paths)}).")
        if not isinstance(no_of_steps, np.integer) or no_of_steps < 1:
            raise TypeError(f"Parameter no_of_steps must be an integer >=1, got {no_of_steps} ({type(no_of_steps)}).")

        self.no_of_paths = no_of_paths
        self.no_of_steps = no_of_steps

    @abstractmethod
    def generate_paths(self):
        '''
        Generates paths based on parameters passed to constructor.
        Must be implemented by subclasses.
        '''
        pass

    