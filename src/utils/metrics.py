import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def transfer_accuracy(domain1, domain2, type1, type2):
	"""
	Label Transfer Accuracy: Metric from UnionCom
	"""
	knn = KNeighborsClassifier()
	knn.fit(domain2, type2)
	type1_predict = knn.predict(domain1)
	count = 0
	for label1, label2 in zip(type1_predict, type1):
		if label1 == label2:
			count += 1
	return count / len(type1)

def calc_frac_idx(x1_mat,x2_mat):
	"""
	Returns fraction closer than true match for each sample (as an array)
	"""
	fracs = []
	x = []
	nsamp = x1_mat.shape[0]
	rank=0
	for row_idx in range(nsamp):
		euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))
		true_nbr = euc_dist[row_idx]
		sort_euc_dist = sorted(euc_dist)
		rank =sort_euc_dist.index(true_nbr)
		frac = float(rank)/(nsamp -1)

		fracs.append(frac)
		x.append(row_idx+1)

	return fracs,x

def calc_domainAveraged_FOSCTTM(data1, data2, links):
	"""
	FOSCTTM from SCOT: Fraction of samples that are matched in both directions
	
	Outputs average FOSCTTM measure (averaged over both domains)
	Get the fraction matched for all data points in both directions
	Averages the fractions in both directions for each data point
	"""
	sorted_data1 = data1[np.transpose(links)[0]]
	sorted_data2 = data2[np.transpose(links)[1]]

	fracs1,xs = calc_frac_idx(sorted_data1, sorted_data2)
	fracs2,xs = calc_frac_idx(sorted_data2, sorted_data1)
	fracs = []
	for i in range(len(fracs1)):
		fracs.append((fracs1[i]+fracs2[i])/2)  
	return np.mean(fracs)