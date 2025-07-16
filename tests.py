## everything here is temporary and can be erased

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

alpha = .05
z_alpha_half = stats.Normal().icdf(1-alpha/2)

lam = 1
n=20
exp_num = 10000

reject_null = 0
#reject_null_alt = 0

for i in range(exp_num):
	X = stats.poisson.rvs(lam, size=n)
	lam_hat = np.mean(X)
	if np.abs(lam_hat - lam)>(lam_hat/n)**0.5*z_alpha_half:
		reject_null +=1
	#m = lam + z_alpha_half**2/(2*n**2)
	#r = z_alpha_half*(4*lam + (z_alpha_half/n)**2)**0.5/(2*n)	
	#if np.abs(lam_hat-m) > r:
	#	reject_null_alt +=1

print(reject_null)
#print(reject_null_alt)

print(f'Empirical type I error probability = {reject_null/exp_num:.3f}')

