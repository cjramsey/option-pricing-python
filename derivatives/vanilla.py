from core.interface import Option

class EuropeanOption(Option):

    def __init__(self, strike, expiry, option_type):
        super().__init__(strike, expiry, option_type)

    def payoff(self, spot):
        if self.option_type == "call":
            return max(spot - self.strike, 0)
        else:
            return max(self.strike - spot, 0)
        
class AmericanOption(Option):

    def __init__(self, strike, expiry, option_type):
        super().__init__(strike, expiry, option_type)

    def payoff(self, spot):
        if self.option_type == "call":
            return max(spot - self.strike, 0)
        else:
            return max(self.strike - spot, 0)
        
    @property
    def early_exercise(self):
        return True
                

def main():
    pass

if __name__ == "__main__":
    main()
