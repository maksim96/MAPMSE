# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.special import factorial
import sys
from scipy.optimize import minimize, LinearConstraint
import itertools

# In[2]:


def UKData():
    suspects = np.array([[1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                         [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
                         [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                         [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0]]).transpose()

    counts = np.array([54, 463, 907, 695, 316, 57, 15, 19, 3, 56, 19, 1, 3, 69, 10, 31, 8, 6, 1, 1, 1, 4, 3, 1, 1])
    N = 6
    suspect_count = np.sum(counts)
    return suspects, counts, N, suspect_count


# In[3]:

np.random.seed(0)
# simulate data
N = 30
suspect_count = 10000

pool = np.ones(2 * N - 1)
pool[:N-1] = 0

# suspects = np.random.choice(pool,replace=False,size=N)


#suspects = np.zeros((suspect_count, N),dtype=bool)

probs = np.exp(-np.arange(N))
probs = np.append(np.ones(N-1),probs)
probs /= np.sum(probs)
for s in range(suspect_count):
    suspects[s] = np.random.choice(pool, replace=False, p=probs,size=N)
suspects, counts = np.unique(suspects, return_counts=True, axis=0)
print(counts)
print(counts.shape)
suspects, counts, N, suspect_count = UKData()


def compute_log_lambdas(mu,alphas,betas,lists):
    alphas = np.repeat([alphas], suspects.shape[0], axis=0)
    # print(alphas)
    # print(lists)
    log_lambdas = mu + np.sum(lists * alphas, axis=1)  # + \
    # np.sum(lists * betas, axis=1)

    betas_summed = np.zeros(lists.shape[0])
    for i in range(lists.shape[0])
        betas_summed[i] = np.sum(betas[suspects[i] > 0][:, suspects[i] > 0]) / 2
        # print(i/lists.shape[0])

    log_lambdas += betas_summed

    return log_lambdas

def compute_log_lambdas(x,A):
    return np.sum(A * x[:, np.newaxis], axis=0)



# In[76]:

#I forget on almost all funcutions here to use only one half of the betas matrix!!!!!


def neg_likelihood_vectorized(x, A, counts):
    log_lambdas = compute_log_lambdas(x,A)

    lambdas = np.exp(log_lambdas)
    # print(log_lambdas,lambdas)
    return -np.sum(counts * log_lambdas - lambdas)# + np.dot(x,x)

def neg_likelihood_alphas_only(x,lists,counts):
    n = lists.shape[1]
    alphas = np.repeat([x[1:n+1]], lists.shape[0], axis=0)
    # print(alphas)
    # print(lists)
    log_lambdas = x[0] + np.sum(lists * alphas, axis=1)  # + \

    exp_params = np.exp(x)

    return -(np.sum(counts * log_lambdas) - exp_params[0] * np.prod(exp_params[1:] + 1))


def neg_derivative_vectorized(x, A, counts):
    return -(np.sum(A*counts - A*np.exp(compute_log_lambdas(x,A)), axis = 1))# + 2*x

def neg_derivative_alphas_only(x,lists,counts):
    exp_params = np.exp(x)
    helper = np.prod(exp_params[1:] + 1)
    del_mu = np.sum(counts) - exp_params[0]*helper
    del_alphas = np.sum(lists * counts[:, np.newaxis], axis = 0) - exp_params[0]*helper*exp_params[1:]/(exp_params[1:]+1)

    return -np.append(del_mu,del_alphas)


def callbackF(xk):
    print(xk[0], neg_likelihood_vectorized(xk,A,counts))

def callbackF_alphas_only(xk):
    print(xk[0], neg_likelihood_alphas_only(xk,suspects,counts))

#returns parameter_matrix and count of betas
def construct_parameter_matrix(lists):
    list_count = lists.shape[1]
    intersection_count = lists.shape[0]
    A = np.zeros((1 + list_count, intersection_count))
    A[0] = 1 #mu
    A[1:list_count + 1] = lists.T #alphas
    #betas

    betas = []

    for i in range(list_count - 1):
        for j in range(i + 1, list_count - 1):
            lists_with_beta = (lists[:, i] > 0) & (lists[:, j] > 0)
            if np.any(lists[lists_with_beta]):
                v = np.zeros(intersection_count)
                v[lists_with_beta] = 1
                betas.append(v)

    betas = np.array(betas)

    A = np.vstack((A,betas))

    return A, betas.shape[0]


A,beta_count = construct_parameter_matrix(suspects)

mu = np.random.rand()
alphas = np.zeros(N)#-10*np.ones(N)#np.random.rand(N)
betas = np.zeros(beta_count)#np.random.rand(beta_count)

x = np.append(np.append(mu, alphas),betas)

#bnds = [(0,None)]
#bnds += ([(None,None)]*(x.shape[0] - 1))

sol = minimize(neg_likelihood_vectorized, x, (A,counts),method='L-BFGS-B',jac=neg_derivative_vectorized,callback=callbackF)#, bounds=bnds)

print(sol)
print(np.exp(sol.x[0]))
x = np.append(mu,alphas)

sol = minimize(neg_likelihood_alphas_only, x, (suspects,counts),method='L-BFGS-B', jac=neg_derivative_alphas_only,callback=callbackF_alphas_only)

print(sol)
print(np.exp(sol.x[0]))

