import numpy as np
import matplotlib.pyplot as plt
from gbm_module.gbm import GeometricBrownianMotion

gBM = GeometricBrownianMotion(0.01,0.1)
trange = np.linspace(0,100,101)
S0 = 1
St = gBM.integrate(S0,trange)
plt.plot(trange,St)
plt.show()