from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

confidence = .95
n = 100
margin = (np.log(2/(1-confidence))/(2*n))**0.5


#using new scipy random variable class (so far only has built in normal and uniform, others can be transformed)
Z = stats.Normal(mu=0, sigma=1)
samples = Z.sample(shape=n)
x_range = np.linspace(samples.min(),samples.max(),num=100)
F = Z.cdf(x_range)
plt.plot(x_range,F)


#built in ecdf function only works to plot cdf and survival function
#sample_ecdf = stats.ecdf(samples)
#sample_ecdf.cdf.plot()

#manually creating ecdf
sorted_samples = np.sort(samples)
sample_ecdf = np.array([i/n for i in range(n)])
plt.step(sorted_samples,sample_ecdf)

#empirical confidence interval from Dvoretzky-Kiefer-Wolfowitz inequality
upper_interval = np.minimum(sample_ecdf + margin, 1)
lower_interval = np.maximum(sample_ecdf - margin, 0)
plt.step(sorted_samples, upper_interval, color='red', label=f'{confidence} CI DKW inequality')
plt.step(sorted_samples, lower_interval, color='red')

#empirical confidence interval from built-in scipy functions
sample_ecdf_obj = stats.ecdf(samples)
#print(type(sample_ecdf_obj.cdf))
ecdf_ci = sample_ecdf_obj.cdf.confidence_interval(confidence_level=confidence)
ecdf_ci.low.plot(color='black',label=f'{confidence} CI built-in')
ecdf_ci.high.plot(color='black')

#using CLT limit and normal approximation (appears to be the same as the built in)
F_hat = sample_ecdf
end_pts_std_norm = stats.norm.interval(confidence)
upper_margin_normal = F_hat + end_pts_std_norm[1]*(F_hat*(1-F_hat)/n)**0.5
lower_margin_normal = F_hat + end_pts_std_norm[0]*(F_hat*(1-F_hat)/n)**0.5
plt.step(sorted_samples,upper_margin_normal,color='purple',label=f'{confidence} CI normal approx')
plt.step(sorted_samples,lower_margin_normal, color='purple')

plt.legend()
plt.show()