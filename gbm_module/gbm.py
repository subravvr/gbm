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
                 sig: float,
                 iv: float,
                 trange: np.ndarray):
        self.alpha,self.sig = alpha,sig
        self.iv = iv
        self.trange = trange
    
    def integrate(self) -> np.ndarray:
        """
        Evaluates integral of gBM over trange with current
        parameters.
        trange: [t0,...,tN]
        """
        w = np.cumsum(np.random.normal(size=self.trange.shape))
        return self.iv*np.exp(self.sig*w + (self.alpha - 0.5 * self.sig * self.sig)*self.trange)
    
    def ensemble(self,
                 Nmc: int) -> np.ndarray:
        """
        Evaluates ensemble simulations with instantiated parameters.
        Nmc: number of Monte Carlo simulations
        Returns a numpy array of shape (len(trange),Nmc) for easy plotting.
        """
        ensemble_arr = np.zeros(shape=(self.trange.shape[0],Nmc))
        for nmc in range(Nmc):
            ensemble_arr[:,nmc] = self.integrate()
        return ensemble_arr
    
    def update_args(self,
                    newargs: list):
        """
        Updates parameters of the model.
        newargs: [alpha,sigma,iv,trange].
        """
        self.alpha,self.sig,self.iv,self.trange = newargs



        
