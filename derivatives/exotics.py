from collections.abc import Iterable
from numbers import Real

import numpy as np

from core.interface import Option

class PerpetualAmericanOption(Option):
    
    def __init__(self, strike, option_type):
        super().__init__(strike, np.inf, option_type)

    def payoff(self, spot):
        if self.option_type == "call":
            return max(spot - self.strike, 0)
        else:
            return max(self.strike - spot, 0)

class GapOption(Option):

    def __init__(self, strike1, strike2, expiry, option_type):
        if not isinstance(strike1, Real) or strike1 < 0:
            raise TypeError(f"Parameter strike1 must be a +ve real value, got {strike1} ({type(strike1)}).")
        if not isinstance(strike2, Real) or strike2 < 0:
            raise TypeError(f"Parameter strike2 must be +ve real value, got {strike2} ({type(strike2)}).")
        if not isinstance(expiry, Real) or expiry < 0:
            raise TypeError(f"Parameter expiry must be +ve real value, got {expiry} ({type(expiry)}).")
        if option_type not in self.VALID_TYPES:
            raise ValueError(f"Parameter option_type must be one of {self.VALID_TYPES}, got {option_type}.")
        
        self.strike1 = strike1
        self.strike2 = strike2
        self.expiry = expiry
        self.option_type = option_type
        
    def payoff(self, spot):
        if self.option_type == "call":
             return spot - self.strike1 if spot > self.strike2 else 0
        else:
            return self.strike1 - spot if spot < self.strike2 else 0
        
class ForwardStartOption(Option):
    '''
    Forward start at-the-money European options.
    Strike price is equal to spot price at T1.
    '''

    def __init__(self, T1, T2, option_type):
        if not isinstance(T1, Real) or T1 < 0:
            raise TypeError(f"Parameter T1 must be +ve real value, got {T1} ({type(T1)}).")
        if not isinstance(T2, Real) or T2 < 0:
            raise TypeError(f"Parameter T2 must be +ve real value, got {T2} ({type(T2)}).")
        if option_type not in self.VALID_TYPES:
            raise ValueError(f"Parameter option_type must be one of {self.VALID_TYPES}, got {option_type}.")
    
        self.T1 = T1
        self.T2 = T2
        self.option_type = option_type

    def payoff(self, spot, strike):
        if self.option_type == "call":
            return max(spot - strike, 0)
        else:
            return max(strike - spot, 0)

class BasicCliquetOption(Option):

    def __init__(self, strike, expirations, option_types):
        if not isinstance(strike, Real) or strike < 0:
            raise TypeError(f"Parameter strikes must be a +ve real value, got {strike} ({type(strike)}).")
        if not isinstance(expirations, Iterable) or isinstance(expirations, (str, bytes)):
            raise TypeError(f"Parameter expirations must be an iterable object, got {type(expirations)}.")
        if not isinstance(option_types, Iterable) or isinstance(option_types, (str, bytes)):
            raise TypeError(f"Parameter option_types must be an iterable object, got {type(option_types)}.")
        for option_type in option_types:
            if option_type not in self.VALID_TYPES:
                raise ValueError(f"Parameter option_types must contain values from {self.VALID_TYPES}, got {option_type}.")
        
        self.strike = strike
        self.expirations = expirations
        self.option_types = option_types

    def payoff(self, spots):
        c = 0
        K = self.strike
        for i, spot in enumerate(spots):
            if self.option_types[i] == "call":
                c += max(spot - K, 0)
            elif self.option_types[i] == "put":
                c += max(K - spot, 0)
            K = spot
        return c
    
class AsianOption(Option):

    AVG_TYPES = {"arithmetic", "geometric"}

    def __init__(self, strike, expiry, option_type, avg_type="arithmetic"):
        if avg_type not in self.AVG_TYPES:
            raise ValueError(f"Parameter average must be one of {self.AVG_TYPES}, got {avg_type}.")
        
        super().__init__(strike, expiry, option_type)
        self.avg_type = avg_type

    def payoff(self, path):
        if self.avg_type == "arithmetic":
            average = np.mean(path)
        elif self.avg_type == "geometric":
            average = np.prod(path)**(1/len(path))

        if self.option_type == "call":
            return max(average - self.strike, 0)
        elif self.option_type == "put":
            return max(self.strike - average, 0)
        
class ChooserOption(Option):
    
    def __init__(self, strike, T1, T2):
        if not isinstance(strike, Real) or strike < 0:
            raise TypeError(f"Parameter strikes must be an +ve real value, got {strike} ({type(strike)}).")
        if not isinstance(T1, Real) or T1 < 0:
            raise TypeError(f"Parameter strike2 must be +ve real value, got {T1} ({type(T1)}).")
        if not isinstance(T2, Real) or T2 < 0:
            raise TypeError(f"Parameter expiry must be +ve real value, got {T2} ({type(T2)}).")
        
        self.strike = strike
        self.T1 = T1
        self.T2 = T2

    def payoff(self, spot, choice):
        if choice == "call":
            return max(spot - self.strike)
        elif choice == "put":
            return max(self.strike - spot)
        else:
            raise ValueError(f"Parameter choice must be one of {self.VALID_TYPES}, got {choice}.")

class BarrierOption(Option):

    VALID_TYPES = {"down_and_out_call", "down_and_in_call",
                   "up_and_out_call", "up_and_in_call",
                   "down_and_out_put", "down_and_in_put",
                   "up_and_out_put", "up_and_in_put"}

    def __init__(self, strike, expiry, option_type, barrier):
        if not isinstance(barrier, Real) or barrier < 0:
            raise TypeError(f"Parameter barrier must be a +ve real valu, got {barrier} ({type(barrier)}).")
        super().__init__(strike, expiry, option_type)
        self.barrier = barrier

    def payoff(self, path):
        if self.option_type == "down_and_out_call":
            if min(path) > self.barrier:
                return max(path[-1] - self.strike, 0)
        elif self.option_type == "down_and_in_call":
            if min(path) < self.barrier:
                return max(path[-1] - self.strike, 0)
        elif self.option_type == "up_and_out_call":
            if max(path) < self.barrier:
                return max(path[-1] - self.strike, 0)
        elif self.option_type == "up_and_in_call":
            if max(path) > self.barrier:
                return max(path[-1] - self.strike, 0)
        elif self.option_type == "down_and_out_put":
            if min(path) > self.barrier:
                return min(self.strike - path[-1], 0)
        elif self.option_type == "down_and_in_put":
            if min(path) < self.barrier:
                return max(self.strike - path[-1], 0)
        elif self.option_type == "up_and_out_put":
            if max(path) < self.barrier:
                return max(self.strike - path[-1], 0)
        elif self.option_type == "up_and_in_put":
            if max(path) > self.barrier:
                return max(self.strike - path[-1], 0)
            
        return 0


def main():
    pass

if __name__ == "__main__":
    main()