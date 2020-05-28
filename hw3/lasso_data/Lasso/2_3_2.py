import numpy as np
import csv
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

X_train = np.genfromtxt('features_train.csv', delimiter=",")
y_train = np.genfromtxt('labels_train.csv', delimiter=",")
X_test = np.genfromtxt('features_test.csv', delimiter=",")
y_test = np.genfromtxt('labels_test.csv', delimiter=",")

allEntry = list(range(1000))
non_actives = set()
for i in range(1000):
    for j in range(i+1, 1000):
        if (X_train[:, i] == X_train[:, j]).mean() > 0:
            non_actives.add(j)

actives = list(set(allEntry)^set(non_actives))
X_train = X_train[:,actives]
X_test = X_test[:,actives]

#lambdas  = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5])
lambdas = np.linspace(np.log(1e-5), np.log(1e5), 10)
lambdas = np.exp(lambdas)
#lambdas = np.array([0.1])
alphas = lambdas / 2 / 500
mse_trains = []
mse_tests = []
for alpha in alphas:
	lasso = Lasso(alpha=alpha)
	model =lasso.fit(X_train,y_train)
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)
	mse_train=mean_squared_error(y_train, y_train_pred)
	mse_test=mean_squared_error(y_test, y_test_pred)
	mse_trains.append(mse_train)
	mse_tests.append(mse_test)
	print(alpha)
	print("mse_train: ", mse_train)
	print("mse_test: ", mse_test)

#mse_trains = np.log(np.array(mse_trains))
#mse_tests = np.log(np.array(mse_tests))
lambdas = np.log(lambdas)
plt.figure()
plt.plot(lambdas, mse_trains,'r.-', label=r'MSE$_{train}$')
plt.plot(lambdas, mse_tests,'b.-', label=r'MSE$_{test}$')
plt.legend()
plt.title(r'MSE$_{train}$ and MSE$_{test}$ of lasso model(with duplicated feature cleaning) with difference choice of $\lambda$')
plt.xlabel(r'$\lambda$(log)')
plt.ylabel('MSE')
plt.show()


