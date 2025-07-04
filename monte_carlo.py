import datetime
import os
import numpy as np
import matplotlib.pyplot as plt


class MonteCarloSimualtion:
    
    def __init__(self, N, S, drift_rate, volatility, T, n):
        self.N = N
        self.S = S
        self.drift_rate = drift_rate
        self.volatility = volatility
        self.T = T
        self.n = n
        self.paths = self.generate_paths()
        self.average = self.get_average()

    def generate_paths(self):
        dt = self.T/self.n
        self.paths = np.zeros((self.N, self.n+1))
        random = np.random.standard_normal((self.N, self.n+1))
        self.paths[:,0] = self.S
        for i in range(1, self.n+1):
            self.paths[:,i] = (self.paths[:,i-1] + self.paths[:,i-1]*self.drift_rate*dt + 
                                self.paths[:,i-1]*self.volatility*random[:,i]*np.sqrt(dt))
        return self.paths
    
    def plot_paths(self, save_figure=False, file_name=""):
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, self.T + self.T/self.n, self.T/self.n), self.paths.T)

        title1 = f"Monte Carlo Simulation"
        title2 = f"Parameters: N={self.N}, S={self.S}, $\\mu=${self.drift_rate}, $\\sigma=${self.volatility}, T={self.T}, $\\Delta t=${self.T/self.n:.5f}"
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

    def get_average(self):
        self.average = self.paths[:,-1].mean()
        return self.average

class FastMonteCarloSimualtion(MonteCarloSimualtion):

    def __init__(self,  N, S, drift_rate, volatility, T, n):
        super().__init__(N, S, drift_rate, volatility, T, n)

    def generate_paths(self):
        dt = self.T/self.n
        dz = np.random.standard_normal(size=(self.N, self.n+1))
        dz[:,0] = 0
        dW = np.cumsum(dz*np.sqrt(dt), axis=1)
        t = np.linspace(0, self.T, self.n+1)
        self.paths = self.S*np.exp((self.drift_rate - (self.volatility**2)/2)*t +self.volatility*dW)
        return self.paths
    
class TerminalMonteCarloPricer:
    
    def __init__(self, option, S, drift_rate, volatility, N=10000, n=250):
        self.option = option
        self.S = S
        self.drift_rate = drift_rate
        self.volatility = volatility
        self.N = N
        self.n = n
        self.price = self.get_price()

    def get_price(self):
        sims = FastMonteCarloSimualtion(self.N, self.S, self.drift_rate, self.volatility, self.option.T, self.n)
        final_values = sims.paths[:,-1]
        f = np.vectorize(lambda x: self.option.payoff(x))
        avg_payoff = f(final_values).mean()
        return avg_payoff
    
class AverageMonteCarloPricer:

    def __init__(self, option, S, drift_rate, volatility, N=10000, n=250):
        self.option = option
        self.S = S
        self.drift_rate = drift_rate
        self.volatility = volatility
        self.N = N
        self.n = n
        self.price = self.get_price()

    def get_price(self):
        sims = FastMonteCarloSimualtion(self.N, self.S, self.drift_rate, self.volatility, self.option.T, self.n)
        averages = np.mean(sims.paths, axis=0)
        f = np.vectorize(lambda x: self.option.payoff(x))
        avg_payoff = f(averages).mean()
        return avg_payoff


def main():
    pass

if __name__ == "__main__":
    main()

