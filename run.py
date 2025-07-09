import numpy as np
import matplotlib.pyplot as plt
from gbm_module.gbm import GeometricBrownianMotion

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