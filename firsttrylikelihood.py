import numpy as np
import datagenerator
from scipy.optimize import minimize

def get_pruned_lambda_matrix(counts,intersections,N):
    keep_betas = []
    for i in range(N):
        for j in range(i+1,N):
            if np.sum(counts[intersections[:,i]*intersections[:,j] > 0]) > 0:
                keep_betas.append([i,j])

    return keep_betas

def compute_log_lambdas(x,A):
    return np.sum(A * x[:, np.newaxis], axis=0)

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

def old_style_UK_fitting():
    suspects, counts, N, suspect_count,counts_exp = datagenerator.UKData()

    A,beta_count = construct_parameter_matrix(suspects)

    mu = np.random.rand()
    alphas = np.zeros(N)#-10*np.ones(N)#np.random.rand(N)
    betas = np.zeros(beta_count)#np.random.rand(beta_count)

    x = np.append(np.append(mu, alphas),betas)

    sol_all = minimize(neg_likelihood_vectorized, x, (A,counts),method='L-BFGS-B',jac=neg_derivative_vectorized)


    x = np.append(mu,alphas)

    sol_alphas_only = minimize(neg_likelihood_alphas_only, x, (suspects,counts), method='L-BFGS-B', jac=neg_derivative_alphas_only)

    return sol_all, sol_alphas_only
