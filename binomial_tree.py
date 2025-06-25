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
    def get_price(self):
        pass

    @abstractmethod
    def get_trees(self):
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

    def get_price(self):
        T = self.option.T
        K = self.option.K
        dt = T/self.N
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        p = (np.exp(self.r * dt) - d)/(u - d)

        price = 0
        for k in range(1, self.N+2):
            ST = self.S * u**k * d**(self.N-k)
            p_ = math.comb(self.N, k) * p**k * (1-p)**(self.N-k)
            if self.option.type == "call":
                price += max(ST - K, 0) * p_
            elif self.option.type == "put":
                price += max(K - ST, 0) * p_

        price = np.exp(-self.r*T) * price

        return price

    def get_trees(self):
        T = self.option.T
        K = self.option.K
        dt = T/self.N
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
        
class AmericanBinomialTreePricer(BinomialTreePricer):
    
    def get_price(self):
        return self.trees[1,0,0]

    def get_trees(self):
        T = self.option.T
        K = self.option.K
        S = self.S
        N = self.N
        r = self.r
        dt = T/N
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        p = (np.exp(self.r * dt) - d)/(u - d)

        stock_tree = np.full((N+1,N+1), np.nan)
        for i in range(N+1):
            for j in range(i+1):
                stock_tree[j,i] = S * u**(i-j) * d**j

        options_tree = np.full_like(stock_tree, np.nan)
        if self.option.type == "call":
            options_tree[:,-1] = np.maximum(stock_tree[:,-1] - K, 0)
            for i in range(N-1, -1, -1):
                for j in range(i+1):
                    exercise_val = stock_tree[j,i] - K
                    hold_val = np.exp(-r*dt)*(p*options_tree[j, i+1] + (1-p)*options_tree[j+1, i+1])
                    options_tree[j,i] = max(exercise_val, hold_val)

        elif self.option.type == "put":
            options_tree[:,-1] = np.maximum(K - stock_tree[:,-1], 0)
            for i in range(N-1, -1, -1):
                for j in range(i+1):
                    exercise_val = K -stock_tree[j,i] 
                    hold_val = np.exp(-r*dt)*(p*options_tree[j, i+1] + (1-p)*options_tree[j+1, i+1])
                    options_tree[j,i] = max(exercise_val, hold_val)

        return np.array([stock_tree, options_tree])
    

if __name__ == "__main__":

    K = 95
    T = 1
    type = "call"
    S = 100
    sigma = 0.1
    r = 0.05
    N = 10

    option = AmericanOption(K, T, type)
    pricer = AmericanBinomialTreePricer(option, S, sigma, r, N)
    pricer.plot_binomial_tree()
