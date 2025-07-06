import numpy as np

from core.interface import PathGenerator

class GeometricBrownianMotion(PathGenerator):

    def __init__(self, spot, drift_rate, volatility, T, 
                 no_of_paths=10000, no_of_steps=250):
        super().__init__(no_of_paths, no_of_steps)
        self.spot = spot
        self.drift_rate = drift_rate
        self.volatility = volatility
        self.T = T
        self.paths = self.generate_paths()

    def generate_paths(self):
        dt = self.T/self.no_of_steps
        dz = np.random.standard_normal(size=(self.no_of_paths, self.no_of_steps+1))
        dz[:,0] = 0
        dW = np.cumsum(dz*np.sqrt(dt), axis=1)
        t = np.linspace(0, self.T, self.no_of_steps+1)
        paths = self.spot*np.exp((self.drift_rate - (self.volatility**2)/2)*t +self.volatility*dW)
        return paths
    
