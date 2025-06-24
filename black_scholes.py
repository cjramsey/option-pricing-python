import numpy as np
from scipy.stats import norm

import greeks
from options import EuropeanOption

# closed-form solutions for European options

class BlackScholesPricer:

    def __init__(self, option, S, sigma, r, q=0):
        self.option = option
        self.S = S
        self.sigma = sigma
        self.r = r
        self.q = q
        self.price = self.get_price()
        
    def get_price(self):
        K = self.option.K
        T = self.option.T
        S = self.S
        r = self.r
        q = self.q
        sigma = self.sigma

        if self.option.type == "call":
            d1 = (np.log(S/K) + (r - q + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            c = S*np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            return c
        
        elif self.option.type == "put":
            d1 = (np.log(S/K) + (r - q - (sigma**2)/2)*T)/(sigma*np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            p = K * np.exp(-r*T) * norm.cdf(-d1) - S * norm.cdf(-d2)
            return p
        
    @property
    def greeks(self):

        args = [self.S, self.option.K, self.sigma, self.r, self.option.T, self.q]

        if self.option.type == "call":
            delta = greeks.delta_call(*args)
            theta = greeks.theta_call(*args)
            rho = greeks.rho_call(*args[:-1])
        elif self.option.type == "put":
            delta =  greeks.delta_put(*args)
            theta = greeks.theta_put(*args)
            rho = greeks.rho_put(*args[:-1])
        gamma = greeks.gamma(*args)
        vega = greeks.vega(*args)

        greeks_dict = {
            "delta": delta,
            "theta": theta,
            "rho": rho,
            "gamma": gamma,
            "vega": vega
        }

        return greeks_dict
        

if __name__ == "__main__":
    
    K = 95
    T = 1
    type = "call"
    S = 100
    sigma = 0.1
    r = 0.05

    option = EuropeanOption(K, T, type)
    pricer = BlackScholesPricer(option, S, sigma, r)
    print(pricer.price)
    print(pricer.greeks)