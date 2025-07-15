import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, special

s = 7
n = 20
B = 1000

X = stats.beta.rvs(s+1,n-s+1,size=B)
psi = np.log(X/(1-X))
psi_range = np.linspace(psi.min(),psi.max(),100)
h = (special.gamma(n+2)/(special.gamma(s+1)*special.gamma(n-s+1)))*np.exp((s+1)*psi_range)/(1+np.exp(psi_range))**(n+2)

plt.hist(psi, density=True)
plt.plot(psi_range, h)
plt.show()