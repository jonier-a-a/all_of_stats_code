from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


confidence = .95

full_df = pd.read_csv('fijiquakes.dat',sep='\s+')

x_samples = full_df['mag']
#x_sorted = x_samples.sort_values()
x_range = np.linspace(x_samples.min(),x_samples.max())

n = len(x_samples)

#empirical confidence interval from Dvoretzky-Kiefer-Wolfowitz inequality
#functions to apply to x_samples
epsilon =((1 / (2 * n)) * np.log(2 / (1-confidence)))**0.5
F_n = lambda x : sum(x_samples < x)/n
L_n = lambda x : max(F_n(x) - epsilon,0)
U_n = lambda x : min(F_n(x) + epsilon,1)

df = pd.DataFrame({
    'x': x_range, 
    'F_n': np.array(list(map(F_n, x_range))), 
    'U_n': np.array(list(map(U_n, x_range))), 
    'L_n': np.array(list(map(L_n, x_range)))
})

plt.plot( 'x', 'L_n', data=df, color='red',label='DKW_ineq')
plt.plot( 'x', 'U_n', data=df, color='red',label='')
plt.plot( 'x', 'F_n', data=df, color='purple')
plt.legend()
plt.show()


#percent point function; for new implementation of scipy random variables, replace by icdf
z_95 = stats.norm.ppf(.975)
theta = F_n(4.9) - F_n(4.3)
se = (theta * (1 - theta) / n)**0.5

print(f'95%% confidence interval: ({theta - z_95 * se:.3f}, {theta + z_95 * se:.3f})')


