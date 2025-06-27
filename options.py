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
        
class ForwardStartOption(EuropeanOption):

    def __init__(self, K, T1, T2, type):
        self.K = K
        self.T1 = T1
        self.T2 = T2
        self.type = type

class CliquetOption(Option):

    def __init__(self, K, T, types):
        self.K = K
        self.T = T
        self.types = types

    def payoff(self, S):
        c = 0
        K = self.K
        for i, spot in enumerate(S):
            if self.types[i] == "call":
                c += max(spot - K, 0)
            elif self.types[i] == "put":
                c += max(K - spot, 0)
            K = spot
        return c


if __name__ == "__main__":
    pass
