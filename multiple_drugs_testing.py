from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_dict = {'drug_name': ['placebo', 'Chlorpromazine', 'Dimenhydranate', 'Pentobarbital 100mg', 'Pentobarbital 150mg'], 'num_of_patients' : [80, 75, 85, 67, 85], 'nausea_incidence' : [45, 26, 52, 35, 37]}

data = pd.DataFrame(data = data_dict)


data['nausea_frequency'] = data['nausea_incidence']/data['num_of_patients']
data['variance'] = data['nausea_frequency']*(1-data['nausea_frequency'])/data['num_of_patients']
data['Wald_statistic'] = (data['nausea_frequency'][0] - data['nausea_frequency'])/(data['variance'][0]+data['variance'])**0.5
data['p_values'] = 1 - stats.Normal().cdf(data['Wald_statistic'])
data['odds_ratios'] = (1-data['nausea_frequency'])*data['nausea_frequency'][0]/(data['nausea_frequency']*(1-data['nausea_frequency'][0]))

print(data[['drug_name','odds_ratios', 'Wald_statistic', 'p_values']][data['drug_name']!='placebo'])

alpha = 0.05
#one-sided test
#z_alpha = stats.Normal().icdf(1-alpha)
print(f'alpha = {alpha:.3f}')
print(f'Bonferroni corrected alpha = {alpha/4:.3f}')


Wald_scores = []
p_values = []
odds_ratios = []

	
#find Benjamini-Hochberg threshold. eliminated placebo row, made it into numpy for ease of computation in next step
p_values_ordered = data['p_values'][data['drug_name']!='placebo'].sort_values().to_numpy()
BH_ind = np.argwhere(p_values_ordered-(alpha/4)*np.arange(1,5)<0).max()
alpha_BH = p_values_ordered[BH_ind]
print(f'Benjamini-Hochberg threshold: {alpha_BH:.3f}')



for i in range(1,5):
	print('##### Drug: '+data['drug_name'][i] + ' #####')
	p_value = data['p_values'][i]
	W = data['Wald_statistic'][i]
	odds_ratio =  data['odds_ratios'][i]
	print(f'W score = {W:.3f}. p-value = {p_value:.3f}') 
	print(f'Odds-ratio = {odds_ratio:.3f}')
	if p_value <= alpha:
		print(f'For uncorrected test: H_0 rejected. Drug seems to reduce incidence of nausea.')
	else:
		print(f'For uncorrected test: Failed to reject H_0. Drug not shown to reduce incidence of nausea.')
	if p_value <= alpha/4:
		print(f'For Bonferroni corrected test: H_0 rejected. Drug seems to reduce incidence of nausea.')
	else:
		print(f'For Bonferroni corrected test: Failed to reject H_0. Drug not shown to reduce incidence of nausea.')
	if p_value <= alpha_BH:
		print(f'For Benjamini-Hochberg corrected test: H_0 rejected. Drug seems to reduce incidence of nausea.')
	else:
		print(f'For Benjamini-Hochberg corrected test: Failed to reject H_0. Drug not shown to reduce incidence of nausea.')
