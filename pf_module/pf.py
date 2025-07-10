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
    
    def generate_particles(self,
                           nparticles: int,
                           bounds: np.ndarray)->np.ndarray:
        """
        Generates particles indpendently for each dynamic arg
        and wraps them together.
        nparticles: number of particles to be generated
        bounds: 2 x len(dynamic_args) to define the bounds of generation
        """
        particles = np.array(
            [
                [np.random.uniform(low=bounds[0,0],high=bounds[1,0]),
                 np.random.uniform(low=bounds[0,1],high=bounds[1,1]),]
                 for n in range(nparticles)
            ]
        )
        return particles
    
    def perturb_particles(self,
                          particles: np.ndarray,
                          scale_factors: np.ndarray)->np.ndarray:
        """
        Perturbs particles in state space for next timestep inferencing.
        Assumes Brownian motion in state space.
        particles: Nx2 array containing alpha, sigma particles.
        scale_factors; 1x2 array scaling the Gaussian random walks in state space.
        """
        perturbations = np.vstack([
            np.random.normal(loc=0,scale=scale_factors[0],size=particles.shape[0]),
            np.random.normal(loc=0,scale=scale_factors[1],size=particles.shape[0])
        ]).transpose()
        return particles+perturbations
    
    def compute_point_likelihood(self,
                                 model_point: float,
                                 data_point: float,
                                 emf=3)->float:
        """
        Computes the point Gaussian likelihood of a model value vs. data value.
        model_point: value of the model
        data_point: value of the observed data
        """
        return pow(2*np.pi*emf,-0.5)*np.exp(-0.5*pow((model_point-data_point)/emf,2))
    
    def compute_windowed_likelihood(self,
                                    model_res: np.ndarray,
                                    data_res: np.ndarray)->float:
        """
        Computes the averaged point likelihood over a window.
        Assumes that model_res and data_res are already windowed over the desired
        time domain.
        """
        averaged_likelihood = 0
        for mp,dp in zip(model_res,data_res):
            averaged_likelihood += self.compute_point_likelihood(mp,dp)/len(model_res)
        return averaged_likelihood
    
    def filter(self,
               windowsize: int,
               bounds: np.ndarray,
               perturb_scale: np.ndarray,
               Nparts: np.ndarray,
               pct_resample: float)->np.ndarray:
        """
        Main particle filter algorithm.
        """
        if self.data is None:
            print("Must load data. Aborting.")
            return
        # window data
        data_window_idx = [windowsize*i for i in range()]
        windows = None ## DEFINE WINDOWS HERE:
        ## [w0,w1,...]
        ## w0 = [d0,d1,..] where d is data.
        # instantiate particles
        particles = self.generate_particles(Nparts,bounds)

        for w in windows:
            # compute particle evaluation over window
            # have to adjust initial condition!!!

            # compute likelihood of particle evaluations averaged over window

            # weight particles with likelihood

            # filter weights

            # resample particles

            # perturb particles in state space

            # finish loop.
            pass
        