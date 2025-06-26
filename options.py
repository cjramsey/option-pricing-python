from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class Option(ABC):

    def __init__(self, K, T, type):
        self.K = K
        self.T = T
        self.type = type

    @abstractmethod
    def payoff(self, spot):
        pass

class EuropeanOption(Option):

    def payoff(self, spot):
        if self.type == 'call':
            return max(spot - self.K, 0)
        elif self.type == 'put':
            return max(self.K - spot, 0)
        
class AmericanOption(Option):

    def payoff(self, spot):
        if self.type == "call":
            return max(spot - self.K, 0)
        elif self.type == "put":
            return max(self.K - spot, 0)
        
# Exotics

class PerpetualAmericanOption(AmericanOption):
    
    def __init__(self, K, type):
        super().__init__(K, np.inf, type)


class GapOption(EuropeanOption):

    def __init__(self, K1, K2, T, type):
        self.K1 = K1
        self.K2 = K2
        self.T = T
        self.type = type
        
    def payoff(self, S):
        if self.type == "call":
             return S - self.K1 if S > self.K2 else 0
        elif self.type == "put":
            return self.K1 - S if S < self.K2 else 0


if __name__ == "__main__":
    pass