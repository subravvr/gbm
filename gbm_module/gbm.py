import numpy as np

class GeometricBrownianMotion:
    """
    gBM class describing a stochastic process satisfying
    dS(t) = alpha*S(t)dt + sig*S(t)dW(t)
    where
    alpha,sig > 0
    W(t) is a Brownian motion
    """
    def __init__(self,
                 alpha : float,
                 sig: float):
        self.alpha,self.sig = alpha,sig
    
    def integrate(self,
                  iv: float,
                  trange: np.ndarray) -> np.ndarray:
        """
        Evaluates integral of gBM over trange with current
        parameters.
        trange: [t0,...,tN]
        """
        w = np.cumsum(np.random.normal(size=trange.shape))
        return iv*np.exp(self.sig*w + (self.alpha - 0.5 * self.sig * self.sig)*trange)

        
