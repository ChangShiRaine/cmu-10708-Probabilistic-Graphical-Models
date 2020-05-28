import numpy as np 
from sklearn.covariance import GraphicalLasso
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

X_train = np.genfromtxt('graph.csv', delimiter=",") #[1000000,4]
alphas = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
y_gt = np.array([[1,1,1,0],[1,1,1,1],[1,1,1,1],[0,1,1,1]])
for alpha in alphas:
    graphicallasso = GraphicalLasso(alpha=alpha,max_iter=1000000)
    graphicallasso.fit(X_train)
    precision = graphicallasso.get_precision()
    precision = np.abs(precision) > 1e-4
    print("alpha: ",alpha)
    #print("precision: ", precision)
    print("precision_score",precision_score(y_gt,precision,average='micro'))
    print("recall_score",recall_score(y_gt,precision,average='micro'))