import numpy as np
import matplotlib.pyplot as plt
from gbm_module.gbm import GeometricBrownianMotion
from pf_module.pf import ParticleFilter

trange = np.linspace(0,1000,1001)
S0 = 50
gBM = GeometricBrownianMotion(0.001,0.01,S0,trange)
St = gBM.integrate()
plt.plot(gBM.trange,St)
plt.show()

Nmc = 10
gbm_ensemble = gBM.ensemble(Nmc)
plt.plot(gBM.trange,gbm_ensemble)
plt.show()

pf_gbm = ParticleFilter(gBM)
eval = pf_gbm.evaluate_args([25,np.linspace(0,1000,1001)],
                            [0.001,0.01],
                            Nevals=10)
plt.plot(pf_gbm.gbm.trange,eval)
plt.show()