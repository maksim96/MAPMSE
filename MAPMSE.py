# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.special import factorial
import sys
from scipy.optimize import minimize, LinearConstraint
from sympy.utilities.iterables import multiset_permutations
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

    '''
    c = np.zeros((25,8)) 
    c[:,2:] = np.fliplr(suspects.T)
    c =c.astype(np.bool) 
    d = np.zeros(64) 
    d[np.packbits(c,axis=1).flatten()] = counts     #only works for maximum 8bith 
    d = d.astype(np.int) #how to get count_all
    count_all=d[1:] #this is probably needed
    '''
    counts_all =np.array([ 54, 463,  15, 907,  19,  56,   1, 695,   3,  19,   1,  69,   0,
         4,   1, 316,   0,   1,   0,  10,   0,   0,   0,   8,   0,   0,
         0,   0,   0,   0,   0,  57,   0,   3,   0,  31,   0,   3,   0,
         6,   0,   0,   0,   1,   0,   0,   0,   1,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0])

    N = 6
    suspect_count = np.sum(counts)
    return suspects, counts, N, suspect_count, counts_all


# In[3]:
'''
np.random.seed(0)
# simulate data
N = 30
suspect_count = 10000

pool = np.ones(2 * N - 1)
pool[:N-1] = 0

# suspects = np.random.choice(pool,replace=False,size=N)


suspects = np.zeros((suspect_count, N),dtype=bool)

probs = np.exp(-np.arange(N))
probs = np.append(np.ones(N-1),probs)
probs /= np.sum(probs)
for s in range(suspect_count):
    suspects[s] = np.random.choice(pool, replace=False, p=probs,size=N)
suspects, counts = np.unique(suspects, return_counts=True, axis=0)
print(counts)
print(counts.shape)
'''
suspects, counts, N, suspect_count,counts_exp = UKData()
counts_copy = counts

def get_index_matrix_alphas_only(N):
    A = np.array(list(itertools.product([0, 1], repeat=N))).T
    A = np.append([np.ones(2**N)], A, axis=0)
    A[0,0] = 0
    return A[:,1:]

def get_index_matrix(N):
    A = get_index_matrix_alphas_only(N)
    B = A[1:]
    combinations = np.array(list(itertools.combinations(range(B.shape[0]), 2)))
    B = np.prod(B[combinations],axis=1) #all possible row pairs
    B = np.append(A, B, axis=0)
    return B

def get_lambdas_matrix_alphas_only(N, withoutLambdaEmptySet=True):
    alpha_indexer = np.array(list(itertools.product([0, 1], repeat=N)))
    alpha_indexer = np.append(np.ones(alpha_indexer.shape[0])[:, np.newaxis], alpha_indexer, axis=1)
    if withoutLambdaEmptySet:
        return alpha_indexer[1:]
    else:
        return alpha_indexer

#order of betas is given by multiset_permutations
def get_lambda_matrix(N, withoutLambdaEmptySet=True):
    alpha_indexer = get_lambdas_matrix_alphas_only(N, withoutLambdaEmptySet)
    alphas_to_betas = np.fliplr(np.array(list(multiset_permutations([0]*(N-2)+[1,1]))).T)
    beta_indexer = np.floor(np.dot(alpha_indexer[:,1:], alphas_to_betas)/2)

    return np.append(alpha_indexer,beta_indexer, axis=1)


def compute_log_lambdas(mu,alphas,betas,lists):
    alphas = np.repeat([alphas], suspects.shape[0], axis=0)
    # print(alphas)
    # print(lists)
    log_lambdas = mu + np.sum(lists * alphas, axis=1)  # + \
    # np.sum(lists * betas, axis=1)

    betas_summed = np.zeros(lists.shape[0])
    for i in range(lists.shape[0]):
        betas_summed[i] = np.sum(betas[suspects[i] > 0][:, suspects[i] > 0]) / 2
        # print(i/lists.shape[0])

    log_lambdas += betas_summed

    return log_lambdas

def compute_log_lambdas(x,A):
    return np.sum(A * x[:, np.newaxis], axis=0)





# In[76]:

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
    log_lambdas = x[0] + np.sum(lists * alphas, axis=1)  #only those lambdas where the list count is nonzero.

    exp_params = np.exp(x)

    return -(np.sum(counts * log_lambdas) - exp_params[0] * np.prod(exp_params[1:] + 1) + exp_params[0])


def neg_derivative_vectorized(x, A, counts):
    return -(np.sum(A*counts - A*np.exp(compute_log_lambdas(x,A)), axis = 1))# + 2*x

def neg_derivative_alphas_only(x,lists,counts):
    exp_params = np.exp(x)
    helper =  np.prod(exp_params[1:] + 1)
    del_mu = np.sum(counts) - exp_params[0]*helper +exp_params[0]
    #this computation is indeed correct
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



#expects exponential/full (also empty intersection) representation
def neg_likelihood_naive(x, parameter_indexer, lambda_indexer, counts, enforce_zero):
    log_lambdas = np.dot(lambda_indexer, x)
    lambdas = np.exp(log_lambdas)
    print(-(np.sum(counts*log_lambdas - lambdas)))
    return -(np.sum(counts*log_lambdas - lambdas))

#expects exponential/full (also empty intersection) representation
def neg_derrivative_naive(x, parameter_indexer, lambda_indexer, counts, enforce_zero):
    lambdas = np.exp(np.dot(lambda_indexer, x))
    derr = -np.dot(parameter_indexer, counts - lambdas)
    derr[enforce_zero] = 0
    return derr
def optimize_naive(counts,N,x,enforce_zero=None):
    if enforce_zero is None:
        enforce_zero = [False]*x.size
    parameter_indexer = get_index_matrix(N)
    lambda_indexer = get_lambda_matrix(N)
    sol = minimize(neg_likelihood_naive, x, (parameter_indexer, lambda_indexer, counts, enforce_zero), method='L-BFGS-B', jac=neg_derrivative_naive)
    print(sol)
    print(np.exp(sol.x[0]))
A,beta_count = construct_parameter_matrix(suspects)

mu = np.random.rand()
alphas = np.zeros(N)#-10*np.ones(N)#np.random.rand(N)
betas = np.zeros(beta_count)#np.random.rand(beta_count)

x = np.append(np.append(mu, alphas),betas)

#bnds = [(0,None)]
#bnds += ([(None,None)]*(x.shape[0] - 1))

#sol = minimize(neg_likelihood_vectorized, x, (A,counts),method='L-BFGS-B',jac=neg_derivative_vectorized,callback=callbackF)#, bounds=bnds)

#print(sol)
#print(np.exp(sol.x[0]))

index_matrix = get_index_matrix(3)
print(index_matrix)

x = np.append(mu,alphas)

sol = minimize(neg_likelihood_alphas_only, x, (suspects,counts), method='L-BFGS-B', jac=neg_derivative_alphas_only, callback=callbackF_alphas_only)

print(sol)
print(np.exp(sol.x[0]))

print("===========================")
print("===========================")
print("===========================")
#np.seterr(all='raise')
x = np.random.randn(7+15)
enforce_zeros = np.zeros(x.size).astype(bool)
enforce_zeros[15:] = True
x[enforce_zeros] = 0
optimize_naive(counts_exp, 6, x, enforce_zeros)
