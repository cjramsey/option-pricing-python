import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod

import greeks
from options import EuropeanOption, PerpetualAmericanOption, GapOption

# closed-form solutions for European options

class BlackScholesPricer(ABC):

    def __init__(self, option, S, sigma, r, q=0):
        self.option = option
        self.S = S
        self.sigma = sigma
        self.r = r
        self.q = q
        self.price = self.get_price()

    @abstractmethod
    def get_price(self):
        pass

class EuropeanBlackScholesPricer(BlackScholesPricer):

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
            p = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
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
    
class PerpetualAmericanBlackSholesPricer(BlackScholesPricer):

    def get_price(self):
        K = self.option.K
        S = self.S
        r = self.r
        q = self.q
        sigma = self.sigma

        w = r - q - (sigma**2)/2
        if self.option.type == "call":
            alpha = (-w + np.sqrt(w**2 +2*sigma**2*r))/sigma**2
            H = K*alpha/(alpha - 1)
            if S < H:
                c = K/(alpha - 1) * (((alpha - 1)*S)/(alpha*K))**alpha
            else: 
                c = S - K
            return c
        
        elif self.option.type == "put":
            alpha = (w + np.sqrt(w**2 + 2*sigma**2*r))/sigma**2
            H = K*alpha/(alpha + 1)
            if S > H:
                p = K/(alpha + 1) * (((alpha+1)*S)/(alpha*K))**(-alpha)
            else:
                p = K - S
            return p
    
class GapBlackScolesPricer(BlackScholesPricer):

    def get_price(self):
        K1 = self.option.K1
        K2 = self.option.K2
        T = self.option.T
        S = self.S
        r = self.r
        q = self.q
        sigma = self.sigma
        
        d1 = (np.log(S/K2) + (r - q + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if self.option.type == "call":
            c = S*np.exp(-q*T)*norm.cdf(d1) - K1*np.exp(-r*T)*norm.cdf(d2)
            return c
        
        elif self.option.type == "put":
            p = K1*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
            return p

if __name__ == "__main__":
    
    K1 = 400000
    K2 = 350000
    T = 1
    type = "put"
    S = 500000
    sigma = 0.2
    r = 0.05
    q = 0

    option = GapOption(K1, K2, 1, type)
    pricer = GapBlackScolesPricer(option, S, sigma, r, q)
    print(pricer.price)
