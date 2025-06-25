import math
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from abc import ABC, abstractmethod
from options import Option, EuropeanOption, AmericanOption

class BinomialTreePricer(ABC):

    def __init__(self, option, S, sigma, r, N):
        self.option = option
        self.S = S
        self.sigma = sigma
        self.r = r
        self.N = N
        self.trees = self.get_trees()
        self.price = self.get_price()

    @abstractmethod
    def get_trees(self):
        pass

    @abstractmethod
    def get_price(self):
        pass

    def plot_binomial_tree(self, annotate_values=True, save_figure=False, file_name=""):
        if self.N > 10:
            print("Cannot plot tree with greater than 10 time steps.")
            return
            
        N = self.N
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

        title1 = f"Binomial Tree: N = {N}, S={self.S} $\\sigma$={self.sigma}, r={self.r}"
        title2 = f"Option: K={self.option.K}, T={self.option.T} "

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

    def get_trees(self):
        K = self.option.K
        dt = self.option.T/self.N
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        p = (np.exp(self.r * dt) - d)/(u - d)

        stock_tree = np.full((self.N+1, self.N+1), np.nan)
        for i in range(self.N+1):
            for j in range(i+1):
                stock_tree[j,i] = self.S * u**(i-j) * d**j

        options_tree = np.full_like(stock_tree, np.nan)
        if self.option.type == "call":
            options_tree[:,-1] = np.maximum(stock_tree[:,-1] - K, 0)
        elif self.option.type == "put":
            options_tree[:,-1] = np.maximum(K - stock_tree[:,-1], 0)

        for i in range(self.N-1, -1, -1):
            for j in range(i+1):
                options_tree[j, i] = np.exp(-self.r*dt)*p*options_tree[j, i+1] + (1-p)*options_tree[j+1, i+1]

        return np.array([stock_tree, options_tree])
    
    def get_price(self):
        return self.trees[1,0,0]


class AmericanBinomialTreePricer(BinomialTreePricer):

    def get_trees(self):
        K = self.option.K
        dt = self.option.T/self.N
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        p = (np.exp(self.r * dt) - d)/(u - d)
        
        stock_tree = np.full((self.N+1,self.N+1), np.nan)
        for i in range(self.N+1):
            for j in range(i+1):
                stock_tree[j,i] = self.S * u**(i-j) * d**j

        options_tree = np.full_like(stock_tree, np.nan)
        if self.option.type == "call":
            options_tree[:,-1] = np.maximum(stock_tree[:,-1] - K, 0)
            for i in range(self.N-1, -1, -1):
                for j in range(i+1):
                    exercise_val = stock_tree[j,i] - K
                    hold_val = np.exp(-self.r*dt)*(p*options_tree[j, i+1] + (1-p)*options_tree[j+1, i+1])
                    options_tree[j,i] = max(exercise_val, hold_val)

        elif self.option.type == "put":
            options_tree[:,-1] = np.maximum(K - stock_tree[:,-1], 0)
            for i in range(self.N-1, -1, -1):
                for j in range(i+1):
                    exercise_val = K -stock_tree[j,i] 
                    hold_val = np.exp(-self.r*dt)*(p*options_tree[j, i+1] + (1-p)*options_tree[j+1, i+1])
                    options_tree[j,i] = max(exercise_val, hold_val)

        return np.array([stock_tree, options_tree])
    
    def get_price(self):
        return self.trees[1,0,0]
    
    @property
    def greeks(self):
        dt = self.option.T/self.N
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        f = self.trees[1]

        delta = (f[0,1] - f[1,1])/(self.S*u - self.S*d)
        
        h = 0.5*(self.S*u**2 - self.S*d**2)
        gamma = ((f[2,2] - f[1,2])/(self.S*u**2 - self.S) - (f[1,2] - f[0,2])/(self.S - self.S*d**2))/h
        
        theta = (f[1,2] - f[0,0])/(2*dt)
        
        # valuing new option with same parameters except small change in sigma
        delta_sigma = 0.01
        f_prime = AmericanBinomialTreePricer(self.option, self.S, self.sigma + delta_sigma, self.r, self.N).get_trees()[1]
        vega = (f_prime[0,0] - f[0,0])/delta_sigma
    
        # valuing new option with same parameters except small change in r
        delta_r = 0.01
        f_prime = AmericanBinomialTreePricer(self.option, self.S, self.sigma, self.r + delta_r, self.N).get_trees()[1]
        rho = (f_prime[0,0] - f[0,0])/delta_r
        
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
    type = "put"
    S = 100
    sigma = 0.1
    r = 0.05
    N = 10

    option = AmericanOption(K, T, type)
    pricer = AmericanBinomialTreePricer(option, S, sigma, r, N)
    pricer.plot_binomial_tree()
