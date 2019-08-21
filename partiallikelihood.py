import math
import numpy as np
import itertools
from scipy.optimize import minimize
from sympy.utilities.iterables import multiset_permutations
from numba import jit

#expects exponential/full (also empty intersection) representation
import singlebetalikelihood

'''
Make it sparse!!!!!!!!!!!!!!!!!!!!!!!!
'''

def get_lambdas_matrix_alphas_only(N, t):
    alpha_indexer = np.eye(N)
    for i in range(2, t+1):
        temp = np.zeros((math.factorial(N)//(math.factorial(i)*math.factorial(N-i)), N))
        for j, idx in enumerate(itertools.combinations(range(N), i)):
            temp[j, list(idx)] = 1

        alpha_indexer = np.append(alpha_indexer, temp, axis=0)


    alpha_indexer = np.append(np.ones(alpha_indexer.shape[0])[:, np.newaxis], alpha_indexer, axis=1)
    return alpha_indexer

#order of betas is given by multiset_permutations
def get_lambda_matrix(N, t):
    alpha_indexer = get_lambdas_matrix_alphas_only(N, t)
    alphas_to_betas = np.fliplr(np.array(list(multiset_permutations([0]*(N-2)+[1,1]))).T)
    beta_indexer = np.floor(np.dot(alpha_indexer[:,1:], alphas_to_betas)/2)

    return np.append(alpha_indexer,beta_indexer, axis=1)

@jit(nopython=True)
def get_log_lambdas_naive(x, K, lists, t):
    mu = x[0]
    alphas = x[1:K + 1]
    betas = x[K + 1:]
    beta_summed = np.zeros(lists.shape[0])
    for i in range(lists.shape[0]):
        #if np.sum(lists[i]) > t:
        #    continue
        index_shift = 0
        for j in range(K-1):

            for k in range(j+1,K):
                if lists[i,j] == 1 and lists[i,k] == 1:
                    beta_summed[i] +=  betas[index_shift+ (k - j - 1)]#betas[j] * K - betas[j]* (betas[j]+ 1) // 2 + (betas[k] - betas[j]) - 1
            index_shift += (K - 1 - j)
    log_lambdas = mu + np.dot(lists, alphas) + beta_summed
    return log_lambdas


def neg_likelihood(x, K, lambda_indexer, lists, counts, t, der):
    #as lambda_indexer is just an index matrix we can safely set 0*np.inf = 0 here.
    with np.errstate(invalid='ignore'):
        log_lambdas = get_log_lambdas_naive(x, K, lists, t)

        #nansafe dot product
        log_lambdas_partial = lambda_indexer * x
        log_lambdas_partial[np.isnan(log_lambdas_partial)] = 0
        log_lambdas_partial = np.sum(log_lambdas_partial, axis=1)
        lambdas_partial = np.exp(log_lambdas_partial)

        summands_with_nonzero_counts = (counts*log_lambdas)
        #counts_times_log_lambdas_nan_safe[np.isnan(counts_times_log_lambdas_nan_safe)] = 0
        return -(np.sum(summands_with_nonzero_counts) - np.sum(lambdas_partial))


@jit(nopython=True)
#probably add possibility to filter data term also by |A| <= t
def neg_derivative(x, K, lambda_indexer, lists, counts, t, der):
    lambdas_partial = np.exp(np.dot(lambda_indexer, x))
    der[0] = np.sum(counts) - np.sum(lambdas_partial)

    for i in range(K):
        relevant_rows_parameter = np.where(lambda_indexer[:, i+1] == 1)
        relevant_rows_data = np.where(lists[:, i] == 1)
        der[i+1] = np.sum(counts[relevant_rows_data]) - np.sum(lambdas_partial[relevant_rows_parameter])
    index_shift = 0
    for i in range(K-1):
        for j in range(i+1,K):
            relevant_rows_parameter = np.where((lambda_indexer[:, i + 1] == 1) & (lambda_indexer[:, j+1] == 1))
            relevant_rows_data = np.where((lists[:, i] == 1) & (lists[:, j] == 1))
            der[K+1 + (index_shift + j - i - 1)] = np.sum(counts[relevant_rows_data]) - np.sum(lambdas_partial[relevant_rows_parameter])
        index_shift += (K - 1 - i)

    return -der

def optimize(x, K, t , lists, counts):
    #parameter_indexer = get_index_matrix(N)
    lambda_indexer = get_lambda_matrix(K, t)
    der = np.zeros(x.size)
    sol = minimize(neg_likelihood, x, (K, lambda_indexer, lists, counts, t, der), method='L-BFGS-B', jac=neg_derivative)
    return sol


if __name__ == "__main__":
    print(get_lambda_matrix(5,5))