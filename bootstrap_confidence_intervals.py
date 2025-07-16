from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#data is a pandas data frame and estimand is a function that should be applied to the data for computing the intended estimate, potentially using *args
#returns a list of boostrap estimates
def bootstrap(data, estimand, num_of_bsamples, *args):
	n = len(data)
	theta_boot = []
	for i in range(num_of_bsamples):
		theta_boot += [estimand(data.sample(n=n, replace=True), *args)]
	return theta_boot

#confidence intervals
def confidence_intervals(data, estimand, num_of_bsamples, alpha_confidence, *args):
	theta = estimand(data, *args)
	theta_boot = bootstrap(data, estimand, num_of_bsamples, *args)
	se = np.std(theta_boot)
	#older method: stats.norm.ppf(.975))
	#newer method using random variable Normal()
	z_alpha = stats.Normal().icdf(1-alpha_confidence/2)
	theta_boot_lower_quantile = np.quantile(theta_boot,alpha_confidence/2)
	theta_boot_upper_quantile = np.quantile(theta_boot,1-alpha_confidence/2)

	print(f'estimate = {theta:.3f}')
	print(f'standard error = {se:.3f}')
	print(f'Normal CI: ({theta-z_alpha*se:.3f},{theta+z_alpha*se:.3f})')
	print(f'Pivotal CI: ({2*theta-theta_boot_upper_quantile:.3f},{2*theta-theta_boot_lower_quantile:.3f})')
	print(f'Percentile CI: ({theta_boot_lower_quantile:.3f},{theta_boot_upper_quantile:.3f})')



'''
#data 1: LSAT GPA correlation
LSAT = [576, 635, 558, 578, 666, 580, 555, 661, 651, 605, 653, 575, 545, 572, 594]
GPA = [3.39, 3.30, 2.81, 3.03, 3.44, 3.07, 3.00, 3.43, 3.36, 3.13, 3.12, 2.74, 2.76, 2.88, 2.96]
df = pd.DataFrame(data={'LSAT': LSAT,'GPA':GPA})

confidence_intervals(df, lambda data: data.corr()['GPA']['LSAT'], 1000, 0.05)


#data 2: skewness of log-normal
norm_data = stats.Normal(mu=0,sigma=1).sample(500)
df = pd.DataFrame(data={'log_normal':np.exp(norm_data)})

confidence_intervals(df, lambda data: data['log_normal'].skew(), 10000, 0.05)
'''

#data 3: quantile length of student t
t_data = stats.t.rvs(3,size=25)
df = pd.Series(data=t_data)

confidence_intervals(df,lambda data: (data.quantile(.75)-data.quantile(.25))/1.34,1000, 0.05)

print(f'exact theoretical value of theta = {(stats.t.ppf(.75,3)-stats.t.ppf(.25,3))/1.34:.3f}')