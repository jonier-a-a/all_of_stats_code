from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


#comparing intervals of confidence for the estimator (sample_average(X1,...,Xn)) of a Bernoulli parameter p using the direct interval for binomial, using hoeffding inequality and using a normal approximation
#intervals are of the form (estimator - l_bound, estimator + u_bound)

confidence = 0.9
p = 0.1

upper_margin_binom = []
upper_margin_hoeff = []
upper_margin_normal = []

for n in range(10,1000,30):
	#binomial
	l, u = stats.binom.interval(confidence, n, p)
	end_pts_binom = (l/n,u/n)
	upper_margin_binom += [end_pts_binom[1]-p]

	#hoeffding
	upper_margin_hoeff += [(np.log(2/(1-confidence))/(2*n))**0.5]
	#end_pts_hoeff = (p - upper_margin_hoeff, p + upper_margin_hoeff)

	#normal
	end_pts_std_norm = stats.norm.interval(confidence)
	upper_margin_normal += [end_pts_std_norm[1]*(p*(1-p)/n)**0.5]



plt.scatter(range(10,1000,30),upper_margin_binom, label='binom')
plt.scatter(range(10,1000,30),upper_margin_hoeff, label='hoeffding')
plt.scatter(range(10,1000,30),upper_margin_normal, label='normal')
plt.legend()
plt.title(f"p = {p}")
plt.show()


