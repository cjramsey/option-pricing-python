from abc import ABC, abstractmethod
import datetime
from numbers import Real
import os

import numpy as np
import matplotlib.pyplot as plt

from core.interface import Option, Pricer, PathGenerator
from derivatives.vanilla import EuropeanOption
from derivatives import exotics
from models.path_generators import GeometricBrownianMotion

class MonteCarloPricer(Pricer, ABC):

    def __init__(self, option, path_generator, r,
                 no_of_trials=10000, no_of_steps=250):
        if not isinstance(option, Option):
            raise TypeError(f"Parameter option must be an Option instance, got {type(option)}.")
        if not isinstance(path_generator, PathGenerator):
            raise TypeError(f"Parameter path_generator must be a PathGenerator instance, got {type(path_generator)}.")
        if not isinstance(r, Real):
            raise TypeError(f"Parameter r must be a real value, got {r} ({type(r)}).")
        if not isinstance(no_of_trials, (int, np.integer)) or no_of_trials < 1:
            raise TypeError(f"Parameter no_of_trials must be an integer >= 1, got {no_of_trials} ({type(no_of_trials)}).")
        if not isinstance(no_of_steps, (int, np.integer)) or no_of_steps < 1:
            raise TypeError(f"Parameter no_of_steps must be an integer >=1, got {no_of_steps} ({type(no_of_steps)}).")
        
        self.option = option
        self.path_generator = path_generator
        self.r = r
        self.T = self.option.expiry
        self.no_of_trials = no_of_trials
        self.no_of_steps = no_of_steps

    @property
    @abstractmethod
    def price(self):
        pass

    @property
    def paths(self):
        return self.generate_paths()

    def generate_paths(self):
        paths = self.path_generator.paths
        return paths
    
    def plot_paths(self, save_figure=False, file_name=""):
        N = self.no_of_trials
        n = self.no_of_steps
        S = self.path_generator.spot
        drift_rate = self.path_generator.drift_rate
        vol = self.path_generator.volatility

        fig, ax = plt.subplots()
        ax.plot(np.arange(0, self.T + self.T/n, self.T/n), self.paths.T)

        title1 = f"Monte Carlo Simulation"
        title2 = f"Parameters: N={N}, S={S}, $\\mu=${drift_rate}, $\\sigma=${vol}, T={self.T}, $\\Delta t=${self.T/n:.5f}"
        ax.set_title(title1 + "\n" + title2)
        ax.set_xlabel("Time (Yrs)")
        ax.set_ylabel("Asset Price ($)")

        if save_figure:
            base_dir = os.path.dirname(__file__)
            try:
                os.mkdir("figures")
            except FileExistsError:
                pass
            if not file_name:
                file_name = f"MonteCarloSim_{datetime.datetime.now().strftime('%d%m%y%H%M%S')}.pdf"
            file_path = os.path.join(base_dir, "figures", file_name)
            fig.savefig(file_path)
            
        plt.show()

class TerminalMonteCarloPricer(MonteCarloPricer):
    
    def __init__(self, option, path_generator, r,
                 no_of_trials=10000, no_of_steps=250 ):
        super().__init__(option, path_generator, r, no_of_trials, no_of_steps)

    @property
    def price(self):
        terminal_values = self.paths[:,-1]
        f = np.vectorize(lambda x: self.option.payoff(x))
        payoffs = f(terminal_values)
        avg_payoff = np.mean(payoffs)
        price = np.exp(-self.r * self.option.expiry) * avg_payoff
        return price

class PathDependentMonteCarloPricer(MonteCarloPricer):

    def __init__(self, option, path_generator, r,
                 no_of_trials=10000, no_of_steps=250 ):
        super().__init__(option, path_generator, r, no_of_trials, no_of_steps)

    @property
    def price(self):
        paths = self.paths
        f = np.vectorize(lambda x: self.option.payoff(x))
        payoffs = f(paths)
        avg_payoff = np.mean(payoffs)
        price = np.exp(-self.r * self.option.expiry) * avg_payoff
        return price

def main():
    pass

if __name__ == "__main__":
    main()

