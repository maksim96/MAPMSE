import numpy as np
import itertools
from scipy.optimize import minimize
from sympy.utilities.iterables import multiset_permutations


#expects exponential/full (also empty intersection) representation

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

def neg_likelihood_naive(x, parameter_indexer, lambda_indexer, counts, enforce_zero):
    #as lambda_indexer is just an index matrix we can safely set 0*np.inf = 0 here.
    with np.errstate(invalid='ignore'):
        log_lambdas = lambda_indexer*x
        log_lambdas[np.isnan(log_lambdas)] = 0
        log_lambdas = np.sum(log_lambdas, axis=1)
        lambdas = np.exp(log_lambdas)

        counts_times_log_lambdas_nan_safe = counts*log_lambdas
        counts_times_log_lambdas_nan_safe[np.isnan(counts_times_log_lambdas_nan_safe)] = 0
        return -(np.sum(counts_times_log_lambdas_nan_safe - lambdas))


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
    return sol, parameter_indexer, lambda_indexer, enforce_zero