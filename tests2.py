import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


mu = 50
sigma = 1
n = 1000

X = stats.norm.rvs(loc=mu,scale=sigma,size = n)
X_range = np.linspace(X.min(),X.max(),100)
Y = np.exp(X)
Y_range = np.linspace(Y.min(),Y.max(),100)


fig, ax = plt.subplots(1,2)
ax[0].hist(X, density=True)
ax[0].plot(X_range, stats.norm.pdf(X_range,loc=mu,scale=sigma))
ax[1].hist(Y, density=True)
ax[1].plot(Y_range, stats.norm.pdf(np.log(Y_range),loc=mu,scale=sigma)/Y_range)
plt.show()
