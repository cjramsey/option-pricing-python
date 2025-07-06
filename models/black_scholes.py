from numbers import Real

import numpy as np
from scipy.stats import norm
from abc import ABC

from core.interface import Option, Pricer
from derivatives.vanilla import EuropeanOption
from derivatives import exotics

# Closed-form solutions for European options

class BlackScholesPricer(Pricer, ABC):

    def __init__(self, option, spot, sigma, r, q=0):
        if not isinstance(option, Option):
            raise TypeError(f"Parameter option must be an Option instance, got {type(option)}.")
        if not isinstance(spot, Real) or spot < 0:
            raise TypeError(f"Parameter spot must be a +ve real value, got {spot} ({type(spot)}).")
        if not isinstance(sigma, Real) or sigma < 0:
            raise TypeError(f"Parameter sigma must be a +ve real value, got {sigma} ({type(sigma)}).")
        if not isinstance(r, Real):
            raise TypeError(f"Parameter r must be a real value, got {r} ({type(r)}).")
        if not isinstance(q, Real) or q < 0:
            raise TypeError(f"Parameter q must be a +ve real value, got {q} ({type(q)}).")
        
        self.option = option
        self.spot = spot
        self.sigma = sigma
        self.r = r
        self.q = q

    @property
    def d1(self):
        d1 = (np.log(self.spot/self.option.strike) 
                   + (self.r - self.q + (self.sigma**2)/2)*self.option.expiry)/(self.sigma*np.sqrt(self.option.expiry))
        return d1
    
    @property
    def d2(self):
        d2 = self.d1 - self.sigma * np.sqrt(self.option.expiry)
        return d2
    
class EuropeanBlackScholesPricer(BlackScholesPricer):

    def __init__(self, option, spot, sigma, r, q=0):
        if not isinstance(option, EuropeanOption):
            raise TypeError(f"Parameter option must be EuropeanOption instance, got {type(option)}.")
        super().__init__(option, spot, sigma, r, q)

    @property
    def price(self):
        K = self.option.strike
        T = self.option.expiry
        S = self.spot
        r = self.r
        q = self.q

        if self.option.option_type == "call":
            c = S*np.exp(-q*T) * norm.cdf(self.d1) - K * np.exp(-r*T) * norm.cdf(self.d2)
            return c
        elif self.option.option_type == "put":
            p = K * np.exp(-r*T) * norm.cdf(-self.d2) - S * norm.cdf(-self.d1)
            return p
        
    @property
    def delta(self):
        T = self.option.expiry
        q = self.q

        if self.option.option_type == "call":
            delta = norm.cdf(self.d1)*np.exp(-q*T)
        elif self.option.option_type == "put":
            delta = (norm.cdf(self.d1) - 1)*np.exp(-q*T)
        return delta
    
    @property
    def theta(self):
        K = self.option.strike
        T = self.option.expiry
        S = self.spot
        sigma = self.sigma
        r = self.r
        q = self.q

        if self.option.option_type == "call":
            theta = (-(S*norm.pdf(self.d1)*sigma*np.exp(-q*T))/(2*np.sqrt(T)) 
                     + q*S*norm.cdf(self.d1)*np.exp(-q*T) - r*K*np.exp(-r*T)*norm.cdf(self.d2))
        elif self.option.option_type == "put":
            theta = (-(S*norm.pdf(self.d1)*sigma*np.exp(-q*T))/(2*np.sqrt(T)) 
                     - q*S*norm.cdf(-self.d1)*np.exp(-q*T) + r*K*np.exp(-r*T)*norm.cdf(-self.d2))
        return theta
    
    @property
    def gamma(self):
        T = self.option.expiry
        S = self.spot
        sigma = self.sigma
        q = self.q

        gamma = (norm.pdf(self.d1)*np.exp(-q*T))/(S*sigma*np.sqrt(T))
        return gamma
    
    @property
    def vega(self):
        T = self.option.expiry
        S = self.spot
        q = self.q

        vega = S*np.sqrt(T)*norm.pdf(self.d1)*np.exp(-q*T)
        return vega
    
    @property
    def rho(self):
        K = self.option.strike
        T = self.option.expiry
        r = self.r

        if self.option.option_type == "call":
            rho = K*T*np.exp(-r*T)*norm.cdf(self.d2)
        elif self.option.option_type == "put":
            rho = -K*T*np.exp(-r*T)*(norm.cdf(-self.d2))
        return rho
            
    @property
    def greeks(self):
        greeks_dict = {
            "delta": self.delta,
            "theta": self.theta,
            "gamma": self.gamma,
            "vega": self.vega,
            "rho": self.rho
        }
        return greeks_dict
    
# Closed-form solutions for exotics

class PerpetualAmericanPricer(BlackScholesPricer):

    def __init__(self, option, spot, sigma, r, q=0):
        if not isinstance(option, exotics.PerpetualAmericanOption):
            raise TypeError(f"Parameter option must be PerpetualAmericanOption instance, got {type(option)}.")
        super().__init__(option, spot, sigma, r, q)

    @property
    def price(self):
        K = self.option.strike
        S = self.spot
        sigma = self.sigma
        r = self.r
        q = self.q

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
    
class GapOptionPricer(BlackScholesPricer):

    def __init__(self, option, spot, sigma, r, q=0):
        if not isinstance(option, exotics.GapOption):
            raise TypeError(f"Parameter option must be GapOption instance, got {type(option)}.")
        super().__init__(option, spot, sigma, r, q)

    @property
    def price(self):
        K1 = self.option.strike1
        K2 = self.option.strike2
        T = self.option.expiry
        S = self.spot
        sigma = self.sigma
        r = self.r
        q = self.q
        
        d1 = (np.log(S/K2) + (r - q + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if self.option.option_type == "call":
            c = S*np.exp(-q*T)*norm.cdf(d1) - K1*np.exp(-r*T)*norm.cdf(d2)
            return c
        
        elif self.option.option_type == "put":
            p = K1*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
            return p

class ForwardStartPricer(BlackScholesPricer):

    def __init__(self, option, spot, sigma, r):
        if not isinstance(option, exotics.ForwardStartOption):
            raise TypeError(f"Parameter option must be ForwardStartOption instance, got {type(option)}.")
        super().__init__(option, spot, sigma, r, q=0)

    @property
    def price(self):
        T = self.option.T2 - self.option.T1
        euro_option = EuropeanOption(self.spot, T, self.option.option_type)
        pricer = EuropeanBlackScholesPricer(euro_option, self.spot, self.sigma, self.r, self.q)
        c = pricer.price
        value = c*np.exp(-self.q*self.option.T1)
        return value

class BasicCliquetPricer(BlackScholesPricer):
    '''
    We assume the term structure is simple with fixed reset times,
    specified type of option between each period, and the strike 
    price of next option resets to the current spot price at the 
    expiry of the previous option.

    Moreover, we make the assumption that in a risk-neutral world, 
    the spot price at time T is equal to spot price at time 0, continuously 
    compounded at the risk-free rate, r, discounted by the yield q.
    '''

    def __init__(self, option, spot, sigma, r, q=0):
        if not isinstance(option, exotics.BasicCliquetOption):
            raise TypeError(f"Parameter option must be BasicCliquetOption instance, got {type(option)}.")
        super().__init__(option, spot, sigma, r, q)

    @property
    def price(self):
        K = self.option.strike
        S = self.spot
        sigma = self.sigma
        r = self.r
        q = self.q

        T1 = 0
        c = 0
        for T2, type in zip(self.option.expirations, self.option.option_types):
            option = exotics.ForwardStartOption(K, T1, T2, type)
            pricer = ForwardStartPricer(option, S*np.exp((r-q)*T1), sigma, r) 
            c += pricer.price
            T1 = T2
            K = S*np.exp((r-q)*T1)
        return c
    

def main():
    pass

if __name__ == "__main__":
    main()