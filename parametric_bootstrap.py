from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.array([
    3.23, -2.50,  1.88, -0.68,  4.43, 0.17,
    1.03, -0.07, -0.01,  0.76,  1.76, 3.18,
    0.33, -0.31,  0.30, -0.61,  1.52, 5.43,
    1.54,  2.28,  0.42,  2.33, -1.03, 4.00,
    0.39   
])

n = len(data)

z_95 = stats.Normal().icdf(0.95)
mu = data.mean()
sigma = data.std()

#mle for estimator of .95 quantile
MLE = mu + z_95*sigma

#standard error from delta method
se_delta = sigma*((1 + z_95**2/2)/n)**0.5

#standard error from parametric bootstrap
B = 1000
MLE_boot = np.empty(B)
for i in range(B):
	X = stats.Normal(mu = mu, sigma = sigma)
	boot_sample = X.sample(n)
	MLE_boot[i] = np.mean(boot_sample)+ np.std(boot_sample)*z_95
se_boot = np.std(MLE_boot)

print(f'.95 quantile MLE:{MLE:.5f}')
print(f'.95 quantile plug in:{np.quantile(data,.95):.5f}')
print(f'MLE Standard Error by delta method:{se_delta:.5f}')
print(f'MLE Standard Error by parametric bootstrap:{se_boot:.5f}')