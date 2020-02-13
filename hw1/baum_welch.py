import numpy as np
from scipy.stats import multivariate_normal

def e_step(X, pi, A, mu, sigma2):
    """E-step: forward-backward message passing"""
    # Messages and sufficient statistics
    N, T, K = X.shape
    M = A.shape[0]
    alpha = np.zeros([N,T,M])  # [N,T,M]
    alpha_sum = np.zeros([N,T])  # [N,T], normalizer for alpha
    beta = np.zeros([N,T,M])  # [N,T,M]
    gamma = np.zeros([N,T,M])  # [N,T,M]
    xi = np.zeros([N,T-1,M,M])  # [N,T-1,M,M]

    # Forward messages
    # TODO ...   

    emission_function = np.zeros([N,T,M])
    for i in range(M):
        emission_function[:,:,i] = multivariate_normal.pdf(X, mu[i,:], sigma2[i]*np.eye(K))

    for n in range(N):
        p_xz = emission_function[n, 0]
        alpha[n,0,:] =  pi * p_xz

    alpha_sum[:,0] = np.sum(alpha[:,0,:], axis=-1)
    alpha[:,0,:] /= alpha_sum[:,0].reshape(-1,1)

    for t in range(1, T):
        for n in range(N):
            p_xz = emission_function[n, t]
            alpha[n,t,:] = p_xz * np.sum(alpha[n,t-1,:].reshape((-1,1)) * A, axis=0)
        alpha_sum[:, t] = np.sum(alpha[:,t,:], axis=-1)
        alpha[:,t,:] /= alpha_sum[:,t].reshape(-1,1)

    # Backward messages
    # TODO ...
    beta[:, -1, :] = 1

    for t in range(T-1,0,-1):
        for n in range(N):
            p_xz = emission_function[n, t]
            beta[n,t-1,:] = 1 / alpha_sum[n,t] * np.dot(A, beta[n,t,:] * p_xz)

    # Sufficient statistics
    # TODO ...
    gamma = alpha * beta

    for t in range(1, T):
        for n in range(N):
            p_xz = emission_function[n, t].reshape((-1,1))
            xi[n,t-1,:,:] = 1 / alpha_sum[n,t] * np.dot(alpha[n,t-1,:].reshape((-1,1)), (p_xz * beta[n,t,:].reshape((-1,1))).T ) * A

    # Although some of them will not be used in the M-step, please still
    # return everything as they will be used in test cases
    return alpha, alpha_sum, beta, gamma, xi


def m_step(X, gamma, xi):
    #M-step: MLE
    # gamma = np.zeros([N,T,M])  # [N,T,M]
    # xi = np.zeros([N,T-1,M,M])  # [N,T-1,M,M]
    N, T, K = X.shape
    M = gamma.shape[2]

    pi = np.average(gamma[:,0,:], axis=0)
    pi /= np.sum(pi)
    A = np.average(np.sum(xi, axis=1), axis=0)
    A /= np.sum(A,axis=1)[:,None]

    mu_num =  np.zeros((N, T, M, K))
    for i in range(N):
        for j in range(T):
            mu_num[i, j] = np.outer(gamma[i, j], X[i, j])
    mu_num = np.average(np.sum(mu_num, axis=1), axis=0)
    mu_denom = np.average(np.sum(gamma, axis=1), axis=0).reshape((-1,1))
    mu =  mu_num / mu_denom

    sigma2_num = np.zeros((N, T, M))
    for n in range(N):
        for t in range(T):
            for m in range(M):
                    sigma2_num[n, t, m] = gamma[n, t, m] * np.linalg.norm(X[n, t] - mu[m])**2
    sigma2_num = np.average(np.sum(sigma2_num, axis=1), axis=0)
    print(sigma2_num.shape)
    sigma2_denom =  np.average(np.sum(gamma, axis=1), axis=0)* K
    print(sigma2_denom.shape)
    sigma2 = sigma2_num / sigma2_denom

    return pi, A, mu, sigma2


def hmm_train(X, pi, A, mu, sigma2, em_step=20):
    """Run Baum-Welch algorithm."""
    for step in range(em_step):
        _, alpha_sum, _, gamma, xi = e_step(X, pi, A, mu, sigma2)
        pi, A, mu, sigma2 = m_step(X, gamma, xi)
        print(f"step: {step}  ln p(x): {np.einsum('nt->', np.log(alpha_sum))}")
    return pi, A, mu, sigma2


def hmm_generate_samples(N, T, pi, A, mu, sigma2):
    """Given pi, A, mu, sigma2, generate [N,T,K] samples."""
    M, K = mu.shape
    Y = np.zeros([N,T], dtype=int) 
    X = np.zeros([N,T,K], dtype=float)
    for n in range(N):
        Y[n,0] = np.random.choice(M, p=pi)  # [1,]
        X[n,0,:] = multivariate_normal.rvs(mu[Y[n,0],:], sigma2[Y[n,0]] * np.eye(K))  # [K,]
    for t in range(T - 1):
        for n in range(N):
            Y[n,t+1] = np.random.choice(M, p=A[Y[n,t],:])  # [1,]
            X[n,t+1,:] = multivariate_normal.rvs(mu[Y[n,t+1],:], sigma2[Y[n,t+1]] * np.eye(K))  # [K,]
    return X


def main():
    """Run Baum-Welch on a simulated toy problem."""
    # Generate a toy problem
    np.random.seed(12345)  # for reproducibility
    N, T, M, K = 10, 100, 4, 2
    pi = np.array([.0, .0, .0, 1.])  # [M,]
    A = np.array([[.7, .1, .1, .1],
                  [.1, .7, .1, .1],
                  [.1, .1, .7, .1],
                  [.1, .1, .1, .7]])  # [M,M]
    mu = np.array([[2., 2.],
                   [-2., 2.],
                   [-2., -2.],
                   [2., -2.]])  # [M,K]
    sigma2 = np.array([.2, .4, .6, .8])  # [M,]
    X = hmm_generate_samples(N, T, pi, A, mu, sigma2)

    # Run on the toy problem
    pi_init = np.random.rand(M)
    pi_init = pi_init / pi_init.sum()
    A_init = np.random.rand(M, M)
    A_init = A_init / A_init.sum(axis=-1, keepdims=True)
    mu_init = 2 * np.random.rand(M, K) - 1
    sigma2_init = np.ones(M)
    pi, A, mu, sigma2 = hmm_train(X, pi_init, A_init, mu_init, sigma2_init, em_step=20)
    print(pi)
    print(A)
    print(mu)
    print(sigma2)


if __name__ == '__main__':
    main()