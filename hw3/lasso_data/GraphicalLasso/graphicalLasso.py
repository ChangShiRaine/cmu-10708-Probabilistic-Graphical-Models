from sklearn import preprocessing
from sklearn.linear_model import Lasso
import numpy as np
import csv

class GraphicalLasso:
    # X is data (n_samples*n_features)
    # A is precision matrix (n_features*n_features)
    # S is covariance matrix (n_features*n_features)
    # rho is regularizer

    # initialization
    def __init__(self,X=None,A=None,S=None,rho=0.1,
                 maxItr=1e+3,tol=1e-2):
        self.X=X
        self.A=A
        self.S=S
        self.rho=rho
        self.maxItr=int(maxItr)
        self.tol=tol
        self.scaler=None
        self.history=[]

    # graphical lasso
    def fit(self,X):
        n_samples,n_features=X.shape[0],X.shape[1]

        self.scaler=preprocessing.StandardScaler().fit(X)
        self.X=self.scaler.transform(X)
        S=self.X.T.dot(self.X)/n_samples
        A=np.linalg.pinv(S)
        A_old=A
        invA=S

        clf=Lasso(alpha=self.rho)
        # block cordinate descent
        # we wanna find l and lmbd
        for i in range(self.maxItr):
            # print 'iteraction', i+1
            for j in range(n_features):
                R,s,sii=self.get(S)
                W=self.get(invA)[0]
                L=self.get(A)[0]

                # find sigma
                sigma=sii+self.rho
                U,D,V=np.linalg.svd(W)
                # print W.shape
                W_half=U.dot(np.diag(np.sqrt(D)).dot(U.T))

                b=np.linalg.pinv(W_half).dot(s)

                # performs lasso
                beta=-clf.fit(W_half,b).coef_

                # find w
                w=W.dot(beta)

                l=-beta/(sigma-beta.T.dot(W).dot(beta))
                lmbd=1/(sigma-beta.T.dot(W).dot(beta))

                A=self.put(L,l,lmbd)
                invA=self.put(W,w,sigma)
                S=self.put(R,s,sii)
                self.history.append(np.linalg.norm(A-A_old,ord=2))

            if np.linalg.norm(A-A_old,ord=2)<self.tol:
                break
            else:
                A_old=A

        self.S=S
        self.A=A


    # delete pth row and column form ndarray X
    def get(self,S):
        end=S.shape[0]-1
        R=S[:-1,:-1]
        s=S[end,:-1]
        sii=S[end][end]

        return [R,s,sii]

    def put(self,R,s,sii):
        n=R.shape[0]+1
        X=np.empty([n,n])
        X[1:,1:]=R
        X[1:,0]=s
        X[0,1:]=s
        X[0][0]=sii

        return X

if __name__ == '__main__':
    X_train = np.genfromtxt('graph.csv', delimiter=",") #[1000000,4]
    rhos = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    for rho in rhos:
        graphicallasso = GraphicalLasso(X=X_train,rho=rho)
        graphicallasso.fit(X_train)
        print("alpha: ",alpha)
        print("precision: ", graphicallasso.A)
        print("covariance: ", graphicallasso.S)