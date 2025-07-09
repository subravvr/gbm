import numpy as np

class ParticleFilter:
    """
    Particle filter class for gBM calibration.
    """
    def __init__(self,gbm,static_args,dynamic_args):
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
        self.static_args = static_args
        self.dynamic_args = dynamic_args

        def wrap_args(sa,da):
            args  = [da[0],da[1],sa[0],sa[1]]
            return args