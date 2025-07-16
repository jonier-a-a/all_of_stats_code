from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#data is a pandas data frame and estimand is a function that should be applied to the data for computing the intended estimate, potentially using *args
#returns a list of boostrap estimates
def bootstrap(data, estimand, num_of_bsamples):
	n = len(data)
	theta_boot = np.empty(num_of_bsamples)
	for i in range(num_of_bsamples):
		tt_b = estimand(data.sample(n=n, replace=True))
		#print(tt_b)
		theta_boot[i] = tt_b
	return theta_boot

#distr is an scipy.stats.rv_continuous object with parameters already included
def parametric_bootstrap(data, distr, estimand, num_of_bsamples):
	n = len(data)
	theta_pboot = np.empty(num_of_bsamples)
	for i in range(num_of_bsamples):
		tt_b = estimand(distr.rvs(size=n))
		#print(tt_b)
		theta_pboot[i] = tt_b
	return theta_pboot

def plot_bootstrap(data, estimand, num_of_bsamples):
	theta_boot = bootstrap(data, estimand, num_of_bsamples)
	print(f'Standard error nonparametric bootstrap: {theta_boot.std():.5f}')
	print(f'.95 Confidence Interval nonparametric bootstrap: ({theta_est - z_025*se_delta:.3f},{theta_est + z_025*se_delta:.3f})')
	plt.hist(theta_boot, bins='auto', density=True)



##############################################################
#data 1: log-normal
##############################################################
n = 50
z_025 = stats.Normal(mu=0,sigma=1).icdf(.975)
fig, ax = plt.subplots()

#actual distribution pdf
theta_real = np.exp(5) 
print(f'Exact value of parameter: {theta_real:.5f}')
distr = stats.lognorm(s=1/n**0.5, scale=theta_real)
x = np.linspace(distr.ppf(0.001), distr.ppf(0.999), 100)
ax.plot(x, distr.pdf(x),label='Exact')

#data
data_array = stats.Normal(mu=5,sigma=1).sample(n)
data_series = pd.Series(data=data_array)
sample_mean = data_array.mean()
sample_std = data_array.std()
theta_est = np.exp(sample_mean)

#delta method
se_delta = theta_est/n**0.5
distr_delta = stats.norm(loc=theta_est, scale=se_delta)
x = np.linspace(distr_delta.ppf(0.001), distr_delta.ppf(0.999), 100)
ax.plot(x, distr_delta.pdf(x),label='Delta')
print(f'Estimate of parameter: {theta_est:.5f}')
print(f'Standard error delta method: {se_delta:.5f}')
print(f'.95 Confidence Interval delta method: ({theta_est - z_025*se_delta:.3f},{theta_est + z_025*se_delta:.3f})')

#nonparametric bootstrap
B = 10000
theta_boot = bootstrap(data_series, lambda data: np.exp(data.mean()), B)
se_boot = theta_boot.std()
print(f'Standard error nonparametric bootstrap: {se_boot:.5f}')
print(f'.95 Confidence Interval nonparametric bootstrap: ({theta_est - z_025*se_boot:.3f},{theta_est + z_025*se_boot:.3f})')
ax.hist(theta_boot, bins='auto', density=True, label='NP Bootstrap')

#parametric bootstrap
theta_pboot = parametric_bootstrap(data_series, stats.norm(loc = sample_mean, scale = sample_std), lambda data: np.exp(data.mean()), B)
se_pboot = theta_pboot.std()
print(f'Standard error parametric bootstrap: {se_pboot:.5f}')
print(f'.95 Confidence Interval parametric bootstrap: ({theta_est - z_025*se_pboot:.3f},{theta_est + z_025*se_pboot:.3f})')
ax.hist(theta_pboot, bins='auto', density=True, label='P Bootstrap')

ax.legend()
plt.show()


'''


##############################################################
#data 2: max of uniform
##############################################################
n = 50
z_025 = stats.Normal(mu=0,sigma=1).icdf(.975)
fig, ax = plt.subplots()

#actual distribution pdf
theta_real = 1 
print(f'Exact value of parameter: {theta_real:.5f}')
x_range = np.linspace(0,theta_real,100)
distr = n*(x_range/theta_real)**(n-1)
ax.plot(x_range, distr,label='Exact')

#data
data_array = stats.Uniform(a=0,b=theta_real).sample(n)
data_series = pd.Series(data=data_array)
theta_est = data_array.max()
se_exact = n*theta_real**2/((n+1)**2*(n+2))
se_est = n*theta_est**2/((n+1)**2*(n+2))
print(f'Estimate of parameter: {theta_est:.5f}')
print(f'Standard error exact: {se_exact:.5f}')
print(f'.95 Confidence Interval exact parameters but normal approx: ({theta_real - z_025*se_exact:.3f},{theta_real + z_025*se_exact:.3f})')
print(f'.95 Confidence Interval estimated parameters and normal approx: ({theta_est - z_025*se_est:.3f},{theta_est + z_025*se_est:.3f})')

#nonparametric bootstrap
B = 10000
theta_boot = bootstrap(data_series, lambda data: data.max(), B)
se_boot = theta_boot.std()
print(f'Standard error nonparametric bootstrap: {se_boot:.5f}')
print(f'.95 Confidence Interval nonparametric bootstrap: ({theta_est - z_025*se_boot:.3f},{theta_est + z_025*se_boot:.3f})')
ax.hist(theta_boot, bins='auto', density=True, label='NP Bootstrap')

#parametric bootstrap
theta_pboot = parametric_bootstrap(data_series, stats.uniform(loc = 0, scale = theta_est), lambda data: data.max(), B)
se_pboot = theta_pboot.std()
print(f'Standard error parametric bootstrap: {se_pboot:.5f}')
print(f'.95 Confidence Interval parametric bootstrap: ({theta_est - z_025*se_pboot:.3f},{theta_est + z_025*se_pboot:.3f})')
ax.hist(theta_pboot, bins='auto', density=True, label='P Bootstrap')

ax.legend()
plt.show()
'''