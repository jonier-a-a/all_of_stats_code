import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


mu = 5
sigma = 1
n = 100

X = stats.norm.rvs(loc=mu,scale=sigma,size = n)
mu_hat = X.mean()

mu_post = stats.norm.rvs(loc=mu_hat,scale=sigma/n**0.5,size = 10000)
theta_post = np.exp(mu_post)

mu_range = np.linspace(mu_post.min(), mu_post.max(), 100)
theta_range = np.linspace(theta_post.min(), theta_post.max(), 100)
#theta_pdf = (n/(2*np.pi))**0.5*np.exp(-n*(np.log(theta_range)-mu_hat)**2/2)/theta_range
theta_pdf = stats.norm.pdf(np.log(theta_range),loc=mu_hat,scale=sigma/n**0.5)/theta_range
theta_pdf_delta = stats.norm.pdf(theta_range,loc=np.exp(mu_hat),scale=np.exp(mu_hat)/(n)**0.5)



fig, ax = plt.subplots(1,2)
ax[0].hist(mu_post, density=True,bins=mu_range)
ax[0].plot(mu_range, stats.norm.pdf(mu_range,loc=mu_hat,scale=sigma/n**0.5))
ax[1].hist(theta_post, density=True,bins=theta_range)
ax[1].plot(theta_range, theta_pdf)
ax[1].plot(theta_range, theta_pdf_delta)
plt.show()
