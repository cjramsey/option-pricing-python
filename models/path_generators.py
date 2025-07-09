from numbers import Real

import numpy as np
import matplotlib.pyplot as plt

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
    
class FractionalBrownianMotion(PathGenerator):

    VALID_METHODS = {"cholesky", "hosking", "daviesharte"}

    def __init__(self, H, T=1, method="daviesharte", no_of_paths=10000, no_of_steps=250):
        super().__init__(no_of_paths, no_of_steps)
        if not isinstance(H, Real):
            raise TypeError(f"Parameter H must be a real number, got {H} ({type(H)}).")
        if H < 0 or H > 1:
            raise ValueError(f"Parameter H must be between 0 and 1, got {H}.")
        if not isinstance(T, Real):
            raise TypeError(f"Parameter T must be a real number, got {T} ({type(T)}).")
        if T < 0:
            raise ValueError(f"Parameter T must be > 0, got {T}.")
        if method not in self.VALID_METHODS:
            raise ValueError(f"Parameter method must be in {self.VALID_METHODS}, got {method}.")
        
        self.H = H
        self.T = T
        self.methods = {"cholesky": self.cholesky,
                        "hosking": self.hosking,
                        "daviesharte": self.daviesharte}
        self.method = method

    def cholesky(self):
        H = self.H
        n = self.no_of_steps
        t = np.linspace(0, self.T, n+1)

        gamma = lambda k: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H)) if k >= 0 else 0

        C = np.eye(n+1)
        for i in range(n+1):
            for j in range(n+1):
                if i != j:
                    C[i, j] = gamma(np.abs(i - j))

        L = np.linalg.cholesky(C)
        
        Z = np.random.standard_normal((n+1))
        fGn = np.dot(L, Z)[1:]
        fBm = np.concatenate([np.array([0]), np.cumsum(fGn)])
        return t, fBm, fGn

    def hosking(self):
        H = self.H
        n = self.no_of_steps
        t = np.linspace(0, self.T, n+1)

        gamma = lambda k: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H)) if k >= 0 else 0

        X = [np.random.standard_normal()]
        mu = [gamma(1)*X[0]]
        var = [1 - (gamma(1)**2)]
        tau = [gamma(1)**2]
        
        d = np.array([gamma(1)])
        
        for i in range(1, n):
            F = np.rot90(np.identity(i+1))
            c = np.array([gamma(k+1) for k in range(0, i+1)])
                
            s = var[i-1] - ((gamma(i+1) - tau[i-1])**2)/var[i-1]

            phi = (gamma(i+1) - tau[i-1])/var[i-1]
            d = d - phi*d[::-1]
            d = np.append(d, phi)        
            
            x = mu[i-1] - var[i-1] * np.random.standard_normal()
            
            X.append(x)
            var.append(s)
            mu.append(d @ X[::-1])
            tau.append(c @ F @ d)
        
        fGn = X
        fBm = np.concat([[0], np.cumsum(X)])    
        return t, fBm, fGn

    def daviesharte(self):
        H = self.H
        n = self.no_of_steps
        t = np.linspace(0, self.T, n)

        gamma = lambda k: 0.5 * (abs(k + 1)**(2 * H) - 2 * abs(k)**(2 * H) + abs(k - 1)**(2 * H))
        c = np.concatenate(
            [np.array([gamma(k) for k in range(n+1)]), 
             np.array([gamma(k) for k in range(n-1, 0, -1)])]
             )
        
        L = np.fft.fft(c).real
        if not np.allclose(np.fft.fft(c).imag, 0, atol=1e-10):
            raise ValueError("FFT has significant imaginary component, check input vector.")
        
        if np.any(L < 0):
            raise ValueError("Negative eigenvalues encountered, invalid ciruclar embedding.")
        
        M = 2 * n
        Z = np.zeros(M, dtype=np.complex128)
        Z[0] = np.sqrt(L[0]) * np.random.normal()
        Z[n] = np.sqrt(L[n]) * np.random.normal()
        X = np.random.normal(0, 1, n-1)
        Y = np.random.normal(0, 1, n-1)
        for k in range(1, n):
            Z[k] = np.sqrt(L[k] / 2) * (X[k-1] + 1j *Y[k-1])
            Z[M-k] = np.conj(Z[k])

        fGn = np.fft.ifft(Z).real[:n] * (self.T / n) ** H * np.sqrt(M)

        fBm = np.concatenate([np.array([0]), np.cumsum(fGn)])

        return t, fBm, fGn
    
    def generate_paths(self):
        method = self.methods.get(self.method)
        return method()
    
def main():
    pass

if __name__ == "__main__":
    main()


        
