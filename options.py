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

    def payoff(self, spots):
        c = 0
        K = self.K
        for i, spot in enumerate(spots):
            if self.types[i] == "call":
                c += max(spot - K, 0)
            elif self.types[i] == "put":
                c += max(K - spot, 0)
            K = spot
        return c
    
class AsianOption(Option):

    def payoff(self, average):
        if self.type == "call":
            return max(average - self.K, 0)
        elif self.type == "put":
            return max(self.K - average, 0)
        
class ChooserOption(Option):
    
    def __init__(self, K, T1, T2):
        self.K = K
        self.T1 = T1
        self.T2 = T2

    def payoff(self, S, choice):
        if choice == "call":
            return max(S - self.K)
        elif choice == "put":
            return max(self.K - S)

class BarrierOption(Option):

    def __init__(self, K, T, type, barrier):
        super().__init__(K, T, type)
        self.barrier = barrier

    def payoff(self, path):
        if self.type == "do_call":
            if path.min() > self.barrier:
                return max(path[-1] - self.K, 0)
        elif self.type == "di_call":
            if path.min() < self.barrier:
                return max(path[-1] - self.K, 0)
        elif self.type == "uo_call":
            if path.max() < self.barrier:
                return max(path[-1] - self.K, 0)
        elif self.type == "ui_call":
            if path.max() > self.barrier:
                return max(path[-1] - self.K, 0)
        
        if self.type == "do_put":
            if path.min() > self.barrier:
                return min(self.K - path[-1], 0)
        elif self.type == "di_put":
            if path.min() < self.barrier:
                return max(self.K - path[-1], 0)
        elif self.type == "uo_put":
            if path.max() < self.barrier:
                return max(self.K - path[-1], 0)
        elif self.type == "ui_put":
            if path.max() > self.barrier:
                return max(self.K - path[-1], 0)
        
    

def main():
    pass

if __name__ == "__main__":
    pass
