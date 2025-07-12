import numpy as np
import matplotlib.pyplot as plt
from gbm_module.gbm import GeometricBrownianMotion
from pf_module.pf import ParticleFilter

np.random.seed(314) # for reproducibility.
trange = np.linspace(0,1000,1001)
S0 = 50
gBM = GeometricBrownianMotion(0.001,0.01,S0,trange)
St = gBM.integrate()
plt.plot(gBM.trange,St)
plt.show()

pf_gbm = ParticleFilter(gBM)
evals = pf_gbm.evaluate_args([25,trange],
                            [0.001,0.01],
                            Nevals=10)
plt.plot(pf_gbm.gbm.trange,evals)
plt.show()

# Loading a gBM sample with known parameters
# alpha=0.001; sigma=0.01
pf_gbm.load_data(St)
bounds = np.array([[0.0005,0.005],
                   [0.0015,0.015]])
perturbation_scale = np.array([1.5e-5,1e-4])
particles = pf_gbm.generate_particles(10,bounds)
plt.scatter(particles[:,0],particles[:,1])
new_particles = pf_gbm.perturb_particles(particles,perturbation_scale)
plt.scatter(new_particles[:,0],new_particles[:,1])
plt.show()


evaluations = np.hstack([
    pf_gbm.evaluate_args([S0,np.linspace(0,250,251)],particle) for particle in particles
])
plt.plot(trange,St,color='b',lw=3)
plt.plot(pf_gbm.gbm.trange,evaluations)
plt.show()

pf_gbm.filter(250,250,bounds,perturbation_scale,100,50,0.5)
particle_history = np.array(pf_gbm.particle_history)
tw_end = np.hstack([[0],[tw[-1] for tw in pf_gbm.twindows]])
plt.hist(particle_history[-1][:,0])
plt.show()
plt.hist(particle_history[-1][:,1])
plt.show()
