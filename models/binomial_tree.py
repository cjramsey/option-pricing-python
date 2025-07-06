from abc import ABC, abstractmethod
import datetime
from numbers import Real
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from core.interface import Pricer
from derivatives.vanilla import Option, EuropeanOption, AmericanOption
from models.black_scholes import EuropeanBlackScholesPricer

class BinomialTreePricer(Pricer, ABC):

    def __init__(self, option, spot, sigma, r, q=0, steps=10):
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
        if not isinstance(steps, Real) or steps < 1:
            raise TypeError(f"Parameter steps must be greater than or equal to 1")
        
        self.option = option
        self.spot = spot
        self.sigma = sigma
        self.r = r
        self.q = q
        self.steps = steps
        self.trees = self.get_trees()

    @abstractmethod
    def get_trees(self):
        pass

    @property
    def delta(self):
        dt = self.option.expiry/self.steps
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        f = self.trees[1]
        delta = (f[0,1] - f[1,1])/(self.spot*u - self.spot*d)
        return delta
    
    @property
    def theta(self):
        dt = self.option.expiry/self.steps
        f = self.trees[1]
        theta = (f[1,2] - f[0,0])/(2*dt)
        return theta
    
    @property
    def gamma(self):
        S = self.spot
        dt = self.option.expiry/self.steps
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        f = self.trees[1]
        h = 0.5*(S*u**2 - S*d**2)
        gamma = ((f[2,2] - f[1,2])/(S*u**2 - S) - (f[1,2] - f[0,2])/(S - S*d**2))/h
        return gamma
    
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

    def plot_binomial_tree(self, annotate_values=True, save_figure=False, file_name=""):
        if self.steps > 10:
            print("Cannot plot tree with greater than 10 time steps.")
            return
            
        N = self.steps
        stock_tree, option_tree = self.trees
        plt.figure(figsize=(1.2*N, 1.2*N))
        
        for j in range(N+1):
            for i in range(j+1):
                if not np.isnan(option_tree[i,j]):
                    x = j
                    y = -2*i + j
                    plt.plot(x, y, c="k", marker='o')

                    if annotate_values:
                        plt.text(x, y - 0.4, f"{option_tree[i,j]:.2f}",
                                ha="center", fontsize=8, c="green")
                        plt.text(x, y + 0.2, f"{stock_tree[i,j]:.2f}",
                                 ha="center", fontsize=8, c="blue")

                    if j < N:
                        if not np.isnan(option_tree[i, j+1]):
                            plt.plot([x, x+1], [y, y+1], c="k", linestyle="-", linewidth=0.5)

                        if not np.isnan(option_tree[i+1, j+1]):
                            plt.plot([x, x+1], [y, y-1], c="k", linestyle="-", linewidth=0.5)

        legend_elements = [
            Line2D([0], [0], marker="o", c="w", label="Stock Price",
                markerfacecolor="blue", markersize=10),
            Line2D([0], [0], marker="o", c="w", label="Option Value",
                markerfacecolor="green", markersize=10)
        ]

        title1 = f"Binomial Tree: N = {N}, S={self.spot} $\\sigma$={self.sigma}, r={self.r}"
        title2 = f"Option: K={self.option.strike}, T={self.option.expiry} "

        plt.title(title1 + "\n" + title2)
        plt.legend(handles=legend_elements)
        plt.axis("off")
        plt.tight_layout()

        if save_figure:
            base_dir = os.path.dirname(__file__)
            try:
                os.mkdir('figures')
            except FileExistsError:
                pass
            if not file_name:
                file_name = f"BinomialTree_{datetime.datetime.now().strftime('%d%m%y%H%M%S')}.pdf"
            file_path = os.path.join(base_dir, "figures", file_name)
            plt.savefig(file_path)

        plt.show()


class EuropeanBinomialTreePricer(BinomialTreePricer):

    def __init__(self, option, spot, sigma, r, q=0, steps=10):
        if not isinstance(option, EuropeanOption):
            raise TypeError(f"Parameter option must be an EuropeanOption instance, got {type(option)}.")
        super().__init__(option, spot, sigma, r, q, steps)

    def get_trees(self):
        K = self.option.strike
        N = self.steps
        dt = self.option.expiry/N
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        p = (np.exp(self.r * dt) - d)/(u - d)

        stock_tree = np.full((N+1, N+1), np.nan)
        for i in range(N+1):
            for j in range(i+1):
                stock_tree[j,i] = self.spot * u**(i-j) * d**j

        options_tree = np.full_like(stock_tree, np.nan)
        if self.option.option_type == "call":
            options_tree[:,-1] = np.maximum(stock_tree[:,-1] - K, 0)
        elif self.option.option_type == "put":
            options_tree[:,-1] = np.maximum(K - stock_tree[:,-1], 0)

        for i in range(N-1, -1, -1):
            for j in range(i+1):
                options_tree[j, i] = np.exp(-self.r*dt)*p*options_tree[j, i+1] + (1-p)*options_tree[j+1, i+1]

        return np.array([stock_tree, options_tree])
    
    @property
    def price(self):
        return self.trees[1,0,0]
    
    @property
    def vega(self):
        dt = self.option.expiry/self.steps
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        f = self.trees[1]
        delta_sigma = 0.001
        f_prime = EuropeanBinomialTreePricer(self.option, self.spot, self.sigma + delta_sigma, self.r, self.steps).get_trees()[1]
        vega = (f_prime[0,0] - f[0,0])/delta_sigma
        return vega
    
    @property
    def rho(self):
        f = self.trees[1]
        delta_r = 0.001
        f_prime = EuropeanBinomialTreePricer(self.option, self.spot, self.sigma, self.r + delta_r, self.steps).get_trees()[1]
        rho = (f_prime[0,0] - f[0,0])/delta_r
        return rho


class AmericanBinomialTreePricer(BinomialTreePricer):

    def __init__(self, option, spot, sigma, r, q=0, steps=10):
        if not isinstance(option, AmericanOption):
            raise TypeError(f"Parameter option must be an AmericanOption instance, got {type(option)}.")
        super().__init__(option, spot, sigma, r, q, steps)

    def get_trees(self):
        K = self.option.strike
        N = self.steps
        dt = self.option.expiry/N
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        p = (np.exp(self.r * dt) - d)/(u - d)
        
        stock_tree = np.full((N+1, N+1), np.nan)
        for i in range(N+1):
            for j in range(i+1):
                stock_tree[j,i] = self.spot * u**(i-j) * d**j

        options_tree = np.full_like(stock_tree, np.nan)
        if self.option.option_type == "call":
            options_tree[:,-1] = np.maximum(stock_tree[:,-1] - K, 0)
            for i in range(N-1, -1, -1):
                for j in range(i+1):
                    exercise_val = stock_tree[j,i] - K
                    hold_val = np.exp(-self.r*dt)*(p*options_tree[j, i+1] + (1-p)*options_tree[j+1, i+1])
                    options_tree[j,i] = max(exercise_val, hold_val)

        elif self.option.option_type == "put":
            options_tree[:,-1] = np.maximum(K - stock_tree[:,-1], 0)
            for i in range(N-1, -1, -1):
                for j in range(i+1):
                    exercise_val = K -stock_tree[j,i] 
                    hold_val = np.exp(-self.r*dt)*(p*options_tree[j, i+1] + (1-p)*options_tree[j+1, i+1])
                    options_tree[j,i] = max(exercise_val, hold_val)

        return np.array([stock_tree, options_tree])
    
    @property
    def price(self):
        return self.trees[1,0,0]
    
    @property
    def vega(self):
        dt = self.option.expiry/self.steps
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        f = self.trees[1]
        delta_sigma = 0.001
        f_prime = AmericanBinomialTreePricer(self.option, self.spot, self.sigma + delta_sigma, self.r, self.steps).get_trees()[1]
        vega = (f_prime[0,0] - f[0,0])/delta_sigma
        return vega
    
    @property
    def rho(self):
        f = self.trees[1]
        delta_r = 0.001
        f_prime = AmericanBinomialTreePricer(self.option, self.spot, self.sigma, self.r + delta_r, self.steps).get_trees()[1]
        rho = (f_prime[0,0] - f[0,0])/delta_r
        return rho
    
    def control_variate_technique(self):
        f_A = self.price
        euro_option = EuropeanOption(self.option.strike, self.option.expiry, self.option.option_type)
        f_E = EuropeanBinomialTreePricer(euro_option, self.spot, self.sigma, self.r, self.steps).price
        f_BSM = EuropeanBlackScholesPricer(euro_option, self.spot, self.sigma, self.r).price

        return f_A + (f_BSM - f_E)


def main():
    pass

if __name__ == "__main__":
    main()
