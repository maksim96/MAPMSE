import numpy as np
import itertools
from scipy.optimize import minimize
from sympy.utilities.iterables import multiset_permutations
import datagenerator
import naivelikelihood
import queue


# in only puts those lamda_A's into the indexing matrix, which are really needed.
# This could help with the exponential sum problem
# expects compact representation (e.g. original UKData representation)


def get_index_matrix_alphas_only(counts, intersection_indexer, N):
    total_counts_in_each_list = np.dot(counts, intersection_indexer)
    intersection_indexer = intersection_indexer[:, total_counts_in_each_list > 0].T
    which_alphas = np.arange(N)
    which_alphas = which_alphas[total_counts_in_each_list > 0]
    which_alphas = dict(zip([(i,) for i in which_alphas], which_alphas))
    A = np.append([np.ones(counts.size)], intersection_indexer, axis=0)
    return A, which_alphas


def get_index_matrix(counts, intersection_indexer, N):
    A, which_alphas = get_index_matrix_alphas_only(counts, intersection_indexer, N)
    B = A[1:]
    combinations = np.array(list(itertools.combinations(range(B.shape[0]), 2)))
    B = np.prod(B[combinations], axis=1)  # all possible row pairs
    total_counts_in_each_pair = np.dot(B, counts)
    B = B[total_counts_in_each_pair > 0]
    B = np.append(A, B, axis=0)
    # first but betas into dict (keys are of the type [i,j] and the values are the row index in the intersection_indexer matrix)
    which_parameter = np.array(list(itertools.combinations(sorted(list(which_alphas.values())), 2)))
    which_parameter = which_parameter[total_counts_in_each_pair > 0]
    which_parameter = dict(zip(list(map(tuple, which_parameter)),np.arange(A.shape[0], B.shape[0])))
    # now put alphas into dict
    which_parameter.update(which_alphas)

    return B, which_parameter


def get_lambdas_matrix_alphas_only(N, withoutLambdaEmptySet=True):
    alpha_indexer = np.array(list(itertools.product([0, 1], repeat=N)))
    alpha_indexer = np.append(np.ones(alpha_indexer.shape[0])[:, np.newaxis], alpha_indexer, axis=1)
    if withoutLambdaEmptySet:
        return alpha_indexer[1:]
    else:
        return alpha_indexer


# order of betas is given by multiset_permutations
def get_lambda_matrix(counts, parameter_matrix, which_parameter, N):
    rows = []
    q = queue.Queue()
    [q.put((i,)) for i in range(N)]
    while not q.empty():
        parameters = q.get()  # [i] is alpha_i, [i,j] is beta_{i,j}, ...
        # this is the crucial new part, which is not done in naive. We only take intersections, where we know lambda_A is not directly 0 (by beta_ij = -inf)
        # basically pruned BFS
        if len(parameters) > 2 or (parameters in which_parameter and np.dot(parameter_matrix[ which_parameter[parameters]], counts) > 0):
            row = np.zeros(1 + N + int(N * (N - 1) / 2))
            row[0] = 1
            row[list(parameters)] = 1
            rows.append(row)
            [q.put(parameters+(i,)) for i in range(max(parameters)+1,N)]

    return np.vstack(tuple(rows))


def neg_likelihood_naive(x, parameter_indexer, lambda_indexer, counts, enforce_zero):
    # as lambda_indexer is just an index matrix we can safely set 0*np.inf = 0 here.
    with np.errstate(invalid='ignore'):
        log_lambdas = lambda_indexer * x
        log_lambdas[np.isnan(log_lambdas)] = 0
        log_lambdas = np.sum(log_lambdas, axis=1)
        lambdas = np.exp(log_lambdas)

        counts_times_log_lambdas_nan_safe = counts * log_lambdas
        counts_times_log_lambdas_nan_safe[np.isnan(counts_times_log_lambdas_nan_safe)] = 0
        return -(np.sum(counts_times_log_lambdas_nan_safe - lambdas))


def neg_derrivative_naive(x, parameter_indexer, lambda_indexer, counts, enforce_zero):
    lambdas = np.exp(np.dot(lambda_indexer, x))
    derr = -np.dot(parameter_indexer, counts - lambdas)
    derr[enforce_zero] = 0
    return derr


def optimize_naive(counts, N, x, enforce_zero=None):
    if enforce_zero is None:
        enforce_zero = [False] * x.size
    parameter_indexer = get_index_matrix(N)
    lambda_indexer = get_lambda_matrix(N)
    sol = minimize(neg_likelihood_naive, x, (parameter_indexer, lambda_indexer, counts, enforce_zero),
                   method='L-BFGS-B', jac=neg_derrivative_naive)
    return sol, parameter_indexer, lambda_indexer, enforce_zero


intersection_indexer, counts, list_count, _, _ = datagenerator.UKData()
parameter_matrix, which_parameter = get_index_matrix(counts, intersection_indexer, list_count)

get_lambda_matrix(counts, parameter_matrix, which_parameter, list_count)