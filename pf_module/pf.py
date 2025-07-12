import numpy as np
from scipy.stats import lognorm
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
        perturbed = particles+perturbations
        while np.any(perturbed<=0):
            perturbations = np.vstack([
            np.random.normal(loc=0,scale=scale_factors[0],size=particles.shape[0]),
            np.random.normal(loc=0,scale=scale_factors[1],size=particles.shape[0])]).transpose()
            perturbed = particles + perturbations
        return perturbed
    
    def compute_tmesh_likelihood(self,
                                 model_evals,
                                 data_res,
                                 Nevals: int,
                                 twindow: np.ndarray,
                                 tmesh: int):
        # computes an ensemble of model evaluations over twindow
        # defaults to evaluating the model at the first and last time points
        pass
    
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
               stride: int,
               bounds: np.ndarray,
               perturb_scale: np.ndarray,
               Nparts: np.ndarray,
               Nevals: int,
               pct_resample: float)->np.ndarray:
        """
        Main particle filter algorithm.
        """
        if self.data is None:
            print("Must load data. Aborting.")
            return
        initial_window = self.data[:windowsize]
        # instantiate particles
        particles = self.generate_particles(Nparts,bounds)
        curr_weights = np.ones(particles.shape[0])/particles.shape[0]
        self.particle_history = [particles]
        twindows = []

        for i in range(windowsize,len(self.data)-windowsize,stride):
            # extract local data, trange
            local_data = self.data[i:i+windowsize]
            local_trange = self.timedomain[i:i+windowsize]
            twindows.append(local_trange)
            iv = local_data[0]

            # Evaluate current particles in over window
            particle_evals = [
                self.evaluate_args([iv,np.arange(windowsize)],particle,Nevals=Nevals) for particle in particles
                ]
            # compute likelihood of particle evaluations averaged over window
            likelihoods = []
            for eval in particle_evals:
                # evaluate the ensemble Neval for each particle
                particle_ensemble_likelihood = [self.compute_windowed_likelihood(peval,local_data) for peval in eval]
                likelihoods.append(np.mean(particle_ensemble_likelihood))
            # recursively weight particles with likelihood and normalize between 0 and 1
            curr_weights = [w*lh for w,lh in zip(curr_weights,likelihoods)]
            curr_weights = curr_weights/np.sum(curr_weights)
            # filter weights
            cutoff = np.quantile(curr_weights,pct_resample)
            particles = particles[np.where(curr_weights>cutoff)]
            # resample particles from surviving population
            filtered_weights = curr_weights[np.where(curr_weights>cutoff)]
            sample_weights = filtered_weights/np.sum(filtered_weights)
            newparticles = []
            while len(newparticles)<Nparts-len(particles):
                u = np.random.uniform()
                try:
                    newparticle = particles[np.where(sample_weights>u)][0]
                    newparticles.append(newparticle)
                except:
                    pass
            particles = np.vstack([particles,np.array(newparticles)])
            # perturb particles in state space
            particles = self.perturb_particles(particles,perturb_scale)
            self.particle_history.append(particles)
            # finish loop.
        self.twindows = np.array(twindows)