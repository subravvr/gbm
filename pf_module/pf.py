import numpy as np
from gbm_module.gbm import GeometricBrownianMotion

class ParticleFilter:
    """
    Particle filter class for gBM calibration.
    """
    def __init__(self,
                 gbm:GeometricBrownianMotion):
        """
        gbm : gBM class to be instantiated.
        static_args : unchanging parameters in gBM
            - initial value
            - time domain
        dynamic_args : paramters to be estimated in gBM
            - drift
            - volatility
        """
        self.gbm = gbm

    def wrap_args(self,
                  sa: list,
                  da: list) -> list:
        """
        Wraps arguments to pass as unpackable list to gBM.
        """
        args  = [da[0],da[1],sa[0],sa[1]]
        return args
    
    def evaluate_args(self,
                      static_args: list,
                      dynamic_args: list,
                      Nevals=1) -> np.ndarray:
        """
        Updates and evaluates the parameters for the gBM.
        static_args: [iv,trange]
        dynamic_args: [alpha,sig]
        Nevals: Number of Monte Carlo evaluations. 
                Passed as Nmc to gbm.ensemble.
        """
        newargs = self.wrap_args(static_args,dynamic_args)
        self.gbm.update_args(newargs)
        return self.gbm.ensemble(Nevals)
    
    def load_data(self,
                  data:np.ndarray):
        """
        Instantiates data stream (pure observations).
        Instantiates a time domain nominally starting from
        t=0 to t=T (T=len(data)-1).
        data: np.ndarray of timeseries observations.
        """
        self.data = data
        self.timedomain = np.arange(len(data))