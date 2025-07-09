import numpy as np
import matplotlib.pyplot as plt
from gbm_module.gbm import GeometricBrownianMotion

gBM = GeometricBrownianMotion(0.01,0.01)
trange = np.linspace(0,100,101)
S0 = 50
St = gBM.integrate(S0,trange)
plt.plot(trange,St)
plt.show()