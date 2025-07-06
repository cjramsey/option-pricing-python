import numpy as np
from scipy.stats import norm

# Greeks for European options on stocks providing a yield of q (0 if non-dividend paying)

def delta_call(S0, K, sigma, r, T, q=0):
    d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    delta = norm.cdf(d1)*np.exp(-q*T)
    return delta

def delta_put(S0, K, sigma, r, T, q=0):
    d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    delta = (norm.cdf(d1) - 1)*np.exp(-q*T)
    return delta


def theta_call(S0, K, sigma, r, T, q=0):
    d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = -(S0*norm.pdf(d1)*sigma*np.exp(-q*T))/(2*np.sqrt(T)) + q*S0*norm.cdf(d1)*np.exp(-q*T) - r*K*np.exp(-r*T)*norm.cdf(d2)
    return theta

def theta_put(S0, K, sigma, r, T, q=0):
    d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = -(S0*norm.pdf(d1)*sigma*np.exp(-q*T))/(2*np.sqrt(T)) - q*S0*norm.cdf(-d1)*np.exp(-q*T) + r*K*np.exp(-r*T)*norm.cdf(-d2)
    return theta


def gamma(S0, K, sigma, r, T, q=0):
    d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    gamma = (norm.pdf(d1)*np.exp(-q*T))/(S0*sigma*np.sqrt(T))
    return gamma


def vega(S0, K, sigma, r, T, q=0):
    d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    vega = S0*np.sqrt(T)*norm.pdf(d1)*np.exp(-q*T)
    return vega
    

def rho_call(S0, K, sigma, r, T):
    d2 = (np.log(S0/K) + (r - (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    rho = K*T*np.exp(-r*T)*norm.cdf(d2)
    return rho

def rho_put(S0, K, sigma, r, T):
    d2 = (np.log(S0/K) + (r - (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    rho = -K*T*np.exp(-r*T)*(norm.cdf(-d2))
    return rho