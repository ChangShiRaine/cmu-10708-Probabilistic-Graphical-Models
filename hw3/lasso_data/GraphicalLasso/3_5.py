import numpy as np
from scipy.sparse import kronsum

c = 0.316# 0<c<0.707

complements = np.array([4,13]) -1 #SC
edges = np.array([1,2,3,5,6,7,8,9,10,11,12,14,15,16]) -1 #S

cov_matrix =  np.array([[1,c,c,2*c*c],[c,1,0,c],[c,0,1,c],[2*c*c,c,c,1]])
kron = np.kron(cov_matrix,cov_matrix)
max_value = 0
for e in complements:
	term1 = np.zeros((1,14))
	term2 = np.zeros((14,14))
	for s in range(14):
		term1[0,s] = kron[e,edges[s]]
	for s1 in range(14):
	    for s2 in range(14):
	    	term2[s1,s2] = kron[edges[s1],edges[s2]]
	term2 = np.linalg.inv(term2)
	total = np.sum(np.abs(np.dot(term1,term2)))
	if total > max_value:
		max_value = total
if max_value >= 1:
	condition1 = True
else:
	condition1 = False

gamma =(kronsum(cov_matrix,cov_matrix) / 2).toarray()

max_value = 0
for e in complements:
	term1 = np.zeros((1,14))
	term2 = np.zeros((14,14))
	for s in range(14):
		term1[0,s] = gamma[e,edges[s]]
	for s1 in range(14):
	    for s2 in range(14):
	    	term2[s1,s2] = gamma[edges[s1],edges[s2]]
	term2 = np.linalg.inv(term2)
	total = np.sum(np.abs(np.dot(term1,term2)))
	if total > max_value:
		max_value = total
if max_value < 1:
	condition2 = True
else:
	condition2 = False

if condition1 and condition2:
	print(c, "good")
else:
	print(c, "bad")


