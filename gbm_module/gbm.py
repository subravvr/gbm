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
        Integrates the gBM over trange with current
        parameters.
        trange: [t0,...,tN]
        """
        St = [iv]
        dt = trange[1]-trange[0]
        sqrt_dt = np.sqrt(dt)
        for t in range(1,len(trange)):
            St.append(St[-1] + self.alpha*St[-1]*dt + self.sig*St[-1]*np.random.normal()*sqrt_dt)
        return np.array(St)
        
