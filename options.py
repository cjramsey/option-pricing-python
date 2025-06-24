from abc import ABC, abstractmethod

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
        

if __name__ == "__main__":
    option = EuropeanOption(105, 1, "put")
    print(option.payoff(90))