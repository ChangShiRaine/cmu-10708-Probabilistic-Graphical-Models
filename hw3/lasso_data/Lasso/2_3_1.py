import numpy as np
import csv
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import sklearn

X_train = np.genfromtxt('features_train.csv', delimiter=",")
y_train = np.genfromtxt('labels_train.csv', delimiter=",")
X_test = np.genfromtxt('features_test.csv', delimiter=",")
y_test = np.genfromtxt('labels_test.csv', delimiter=",")

#lambdas  = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5])
lambdas = np.linspace(np.log(1e-5), np.log(1e5), 10)
lambdas = np.exp(lambdas)
#lambdas = np.array([0.1])
alphas = lambdas / 2 / 500
threshold = 0.9
sample_ratio = 0.75
sample_times = 200
mse_trains = []
mse_tests = []
for alpha in alphas:
	lasso = Lasso(alpha=alpha)
	total = np.zeros(1000)
	for i in range(sample_times):
		index = list(np.random.choice(500,int(500*sample_ratio)))
		#print(index)
		X_train_sample = X_train[index]
		#print(X_train_sample.shape)
		y_train_sample = y_train[index]
		model =lasso.fit(X_train_sample,y_train_sample)
		total += (np.abs(model.coef_) > 1e-4)
	total /= sample_times
	#print(total)
	allEntry = list(range(1000))
	non_actives = set()
	for i in range(1000):
		if total[i]<=threshold:
			non_actives.add(i)
	actives = list(set(allEntry)^set(non_actives))

	print("#feature selected:",len(actives))
	X_train_select = X_train[:,actives]
	y_train_select = y_train[:,actives]
	X_test_select = X_test[:,actives]
	y_test_select = y_test[:,actives]
	final_model =lasso.fit(X_train_select,y_train_select)
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
plt.title(r'MSE$_{train}$ and MSE$_{test}$ of radomized lasso model with difference choice of $\lambda$')
plt.xlabel(r'$\lambda$(log)')
plt.ylabel('MSE')
plt.show()
