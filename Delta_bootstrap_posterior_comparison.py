import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

#data X is Binom(n,p1) corresponding to placebo Y is Binom(m,p2) corresponding to treatment
n = 50
X = 30
m = 50
Y = 40
p1_hat = X/n
p2_hat = Y/m
tau_hat = p2_hat-p1_hat

#delta method
se_delta = (p1_hat*(1-p1_hat)/n+p2_hat*(1-p2_hat)/m)**0.5
z_05 = stats.Normal().icdf(0.95)
psi_hat = np.log(p1_hat*(1-p2_hat)/(p2_hat*(1-p1_hat)))
se_psi_delta = (1/(n*p1_hat*(1-p1_hat))+1/(m*p2_hat*(1-p2_hat)))**0.5

#bootstrap
B = 10000
X_boot = stats.binom.rvs(n,p1_hat,size=B)
Y_boot = stats.binom.rvs(m,p2_hat,size=B)
tau_boot = Y_boot/m - X_boot/n
tau_boot_mean = tau_boot.mean()
tau_boot_std = tau_boot.std()

#posterior (bayesian)
BB = 10000
p1_post = stats.beta.rvs(X+1, n-X +1,size=BB)
p2_post = stats.beta.rvs(Y+1, m-Y +1,size=BB)
tau_post = p2_post-p1_post
tau_post_mean = tau_post.mean()
a_tau_post = np.quantile(tau_post,0.05)
b_tau_post = np.quantile(tau_post,0.95)
psi_post = np.log(p1_post*(1-p2_post)/(p2_post*(1-p1_post)))
psi_post_mean = psi_post.mean()
a_psi_post = np.quantile(psi_post,0.05)
b_psi_post = np.quantile(psi_post,0.95)


print('############################')
print('Below \'tau\' is the difference between recovery probabilities')
print(f'tau MLE: {tau_hat:.3f}')
print(f'90% confidence interval for tau, delta method ({tau_hat-z_05*se_delta:.3f},{tau_hat+z_05*se_delta:.3f})')
print(f'tau mean bootstrap: {tau_boot_mean:.3f}')
print(f'90% confidence interval for tau, bootstrap ({tau_hat-z_05*tau_boot_std:.3f},{tau_hat+z_05*tau_boot_std:.3f})')
print(f'tau posterior mean (constant prior) {tau_post_mean:.3f}')
print(f'90% posterior interval for tau (constant prior) ({a_tau_post:.3f},{b_tau_post:.3f})')
print('Below \'psi\' is the log of the odds-ratio (success of placebo/success of treatment)')
print(f'psi MLE: {psi_hat:.3f}')
print(f'90% confidence interval for psi, delta method ({psi_hat-z_05*se_psi_delta:.3f},{psi_hat+z_05*se_psi_delta:.3f})')
print(f'psi posterior mean (constant prior) {psi_post_mean:.3f}')
print(f'90% posterior interval for psi (constant prior) ({a_psi_post:.3f},{b_psi_post:.3f})')
print('############################')