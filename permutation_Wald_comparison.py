from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#data
'''
X = np.array([0.225, 0.262, 0.217, 0.240, 0.230, 0.229, 0.235, 0.217])
Y = np.array([0.209, 0.205, 0.196, 0.210, 0.202, 0.207, 0.224, 0.223, 0.220, 0.201])
'''
n = 55+33+70+49 
X = np.zeros(n)
X[0:55] = -2
X[55:55+33] = -1 
X[55+33:55+33+70] = 1
X[55+33+70:] = 2
#print(X)
m = 141+145+139+161
Y = np.zeros(m)
Y[0:141] = -2
Y[141:141+145] = -1 
Y[141+145:141+145+139] = 1
Y[141+145+139:] = 2
#print(Y)

#Wald
mean_X = np.mean(X)
print(mean_X)
std_X = np.std(X)
#print(std_X)
mean_Y = np.mean(Y)
print(mean_Y)
std_Y = np.std(Y)
#print(std_Y)
mean_diff = mean_X - mean_Y
se_diff = (std_X**2/len(X) + std_Y**2/len(Y))**0.5
#print(se_diff)
W = mean_diff/se_diff
print(W)
p = 2*stats.Normal().cdf(-np.abs(W))
print(f'Wilson Test p-value: {p:.4f}')
z_025 = stats.Normal().icdf(.975)
print(f'95% Confidence interval for mean difference (asymptotic normal approx): ({mean_diff-z_025*se_diff:.3f},{mean_diff+z_025*se_diff:.3f})')

#Permutation
full_data = np.concatenate((X,Y)) 
B = 1000000
rng = np.random.default_rng()	
count = 0
for i in range(B):
	Z = rng.permuted(full_data)
	new_X = Z[:len(X)]
	new_Y = Z[len(X):]
	new_diff = np.mean(new_X) - np.mean(new_Y)
	if np.abs(new_diff)>np.abs(mean_diff):
		count +=1
p_perm = count/B 
print(f'Permutation test p-value:{p_perm:.4f}')