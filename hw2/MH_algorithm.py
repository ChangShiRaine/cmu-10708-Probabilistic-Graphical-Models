import numpy as np
from scipy.stats import multivariate_normal, poisson, norm
import seaborn as sns
import matplotlib.pyplot as plt 

sigma = 0.05
t = 5
alpha = 0.1
beta = 0.1
tau_0 = 0.0001
tau_1 = 0.0001
samples = np.zeros((10000,45))
homes = np.zeros(10000)
rejections = 0
rejection = False

def metropolis_hastings(y):
    x = np.zeros(45)
    x[43] = 1
    x[44] = 1
    iter = 5000
    index = 0
    print("Burning phase...")
    for i in range(iter):
        x = DrawSample(y, x, sigma)
        #samples[index,:] = x 
        homes[index] = x[0]
        index += 1
        if i % 100 == 0:
            print("iter ", i, "  ", x[0])
    print(x)
    print("MCMC phase...")
    for i in range(t*iter):
        x = DrawSample(y, x, sigma)
        if rejection == True:
            rejections += 1
        if i % t == 0:
            #samples[index,:] = x
            homes[index] = x[0]
            index += 1
            if i % (100*t) == 0:
                print("sample ", i / t, "  ", x[0])
        if i == 5*iter-1: 
            print("final estimation:", x)
            print("rejection ratio:", rejections / (t*iter))
            with open("results.txt","a") as results:
                results.write("sigma" + str(sigma)[2:] + "_t" + str(t) + "\n")
                results.write("final estimation: " + str(x) + "\n")
                results.write("rejection ratio: " + str(rejections / (t*iter)) + "\n")

    return 0

def gamma_pdf(x):
    return x**(alpha-1) * np.exp(-beta*x)

def JointLikelihood(y, x):
    #print("x",x)
    p = 0
    #home = np.log(norm(0, 1/tau_0).pdf(x[0]))
    #mu_att = np.log(norm(0, 1/tau_1).pdf(x[41]))
    #mu_def = np.log(norm(0, 1/tau_1).pdf(x[42]))
    home, mu_att, mu_def = np.log(norm(0, 1/tau_0).pdf([x[0],x[41],x[42]]))
    tau_att = np.log(gamma_pdf(x[43]))
    tau_def = np.log(gamma_pdf(x[44]))
    #tau_att, tau_def = np.log(gamma_pdf([x[43],x[44]]))
    #print("mu_att",mu_att)
    #print("mu_def",mu_def)
    #print("tau_att",tau_att)
    #print("tau_def",tau_def)
    #print("home",home)
    p += mu_att + mu_def + tau_att + tau_def + home
    att_t = np.zeros(20)
    def_t =  np.zeros(20)
    for i in range(20):
        att_t[i] = np.log(norm(x[41], 1/x[43]).pdf(x[i+1]))
        def_t[i] = np.log(norm(x[42], 1/x[44]).pdf(x[i+21]))
        #print("att_t[i]",att_t[i])
        #print("def_t[i]",def_t[i])
        p += att_t[i] + def_t[i]
    theta = np.zeros((380,2))
    goal = np.zeros((380,2))
    for g in range(380):
        for j in range(2):
            if j == 0:
                theta[g][j] = np.exp(x[0] + att_t[int(y[g,2])] - def_t[int(y[g][3])])
            else:
                theta[g][j] = np.exp(att_t[int(y[g,3])] - def_t[int(y[g,2])])
            goal[g][j] = np.log(poisson(theta[g][j]).pmf(int(y[g,j])))
            #print("goal[g][j]",goal[g][j])
            p += goal[g][j]
    #print("p",p)
    return p

def DrawSample(y, x,sigma): # x = [theta, eta]
    x_prime = multivariate_normal.rvs(x, sigma*np.eye(45))
    while not (x_prime[43]>0 and x_prime[44]>0):
        x_prime = multivariate_normal.rvs(x, sigma*np.eye(45))
    x_prime[1]=0
    x_prime[21]=0
    #print("x_prime",x_prime)
    u = np.random.uniform(0,1,1)
    #print("JointLikelihood",JointLikelihood(y, x))
    a = np.minimum(1,np.exp(JointLikelihood(y, x_prime) - JointLikelihood(y, x)))
    #print(a)
    #print("JointLikelihood_new",JointLikelihood(y, x_prime))
    if u < a:
        x_new = x_prime
        rejection = False
    else:
        x_new = x
        rejection = True
    return x_new

if __name__ == '__main__':
    y = np.loadtxt("premier_league_2013_2014.csv",delimiter=",")
    metropolis_hastings(y)
    #sns.jointplot(samples[:, 0], samples[:, 1])
    x=np.arange(5000)
    burning = homes[:5000]
    mcmc = homes[-5000:]

    with open("results.txt","a") as results:
        results.write("burning homes: " + str(burning) + "\n")
        results.write("mcmc homes: " + str(mcmc) + "\n")

    plt.figure()
    plt.plot(x,burning,'b--')
    plt.title('Trace plot of the burning phase for latent variable home')
    plt.xlabel('Iteration')
    plt.ylabel('Home')
    #plt.savefig('burning_sigma05_t5.jpg')
    plt.savefig('burning_sigma' + str(sigma)[2:] + '_t'+ str(t)+'.jpg')

    plt.figure()
    plt.plot(x,mcmc,'b--')
    plt.title('Trace plot of the MCMC samples for latent variable home')
    plt.xlabel('Iteration')
    plt.ylabel('Home')
    #plt.savefig('mcmc_sigma05_t5.jpg')
    plt.savefig('mcmc_sigma' + str(sigma)[2:] + '_t'+ str(t)+'.jpg')








   

