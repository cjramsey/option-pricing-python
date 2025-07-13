from core.interface import Option

class EuropeanOption(Option):
    '''
    Class for European-style options which can only be exercised at expiration date.
    Passed to Pricer objects to calculate price of the option.
    '''

    def __init__(self, strike, expiry, option_type):
        '''
        Constructor for EuropeanOption objects.
        Inherits constructor from Option abstract base class.
        '''
        super().__init__(strike, expiry, option_type)

    def payoff(self, spot):
        '''
        Calculates payoff of option for specifed spot price.

        args:
            spot (float): spot price.
        '''

        if self.option_type == "call":
            return max(spot - self.strike, 0)
        else:
            return max(self.strike - spot, 0)
        
class AmericanOption(Option):
    '''
    Class for American-style options which allows for early exercise up to the expiration date.
    Passed to Pricer objects to calculate price of the option.
    '''

    def __init__(self, strike, expiry, option_type):
        '''
        Constructor for AmericanOption objects.
        Inherits constructor from Option abstract base class.
        '''
        super().__init__(strike, expiry, option_type)

    def payoff(self, spot):
        '''
        Calculates payoff of option for specifed spot price.

        args:
            spot (float): spot price.
        '''
        
        if self.option_type == "call":
            return max(spot - self.strike, 0)
        else:
            return max(self.strike - spot, 0)
        
    @property
    def early_exercise(self):
        '''Defines early exercise to be True.'''
        return True
                

def main():
    pass

if __name__ == "__main__":
    main()
