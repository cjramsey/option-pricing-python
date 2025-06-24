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
        self._average = self.get_average()

    def generate_paths(self):
        dt = self.T/self.n
        self.paths = np.zeros((self.N, self.n+1))
        random = np.random.standard_normal((self.N, self.n+1))
        self.paths[:,0] = self.S
        for i in range(1, self.n+1):
            self.paths[:,i] = (self.paths[:,i-1] + self.paths[:,i-1]*self.drift_rate*dt + 
                                self.paths[:,i-1]*self.volatility*random[:,i]*np.sqrt(dt))
        return self.paths
    
    def plot_paths(self, file_name=''):
        fig, ax = plt.subplots()
        ax.plot(self.paths.T)

        if file_name:
            base_dir = os.path.dirname(__file__)
            try:
                os.mkdir('figures')
            except FileExistsError:
                pass
            file_path = os.path.join(base_dir, 'figures', file_name)
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


if __name__ == "__main__":
    y = FastMonteCarloSimualtion(N=10000, S=100, drift_rate=0.1, volatility=0.2, T=1, n=300)
    y.plot_paths()
    print(y.average)

