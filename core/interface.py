from abc import ABC, abstractmethod
from numbers import Real

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
    
class Model(ABC):
    pass
    
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

    