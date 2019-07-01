import itertools

import numpy as np
from numba import jit
import singlebetalikelihood
import math
from scipy.special import gamma

def neg_likelihood(x, K, lists, counts):
    '''
    This is not the correct likelihood. It tries to heuristically approximate the correct likelihood,
    with the dynamic programming approach
    Expects to contain only those lists, where the count is > 0
    :param x: the current parameters [mu, alpha_1,...alpha_K,beta]
    :param lists: a 2d array consisting of index vector rows
    :param counts: array with all the counts for each index vector
    :return:
    '''
    # print(alphas)
    # print(lists)

    mu = x[0]
    alphas = x[1:K + 1]
    betas = x[K + 1:]

    ids = np.arange(K)

    log_lambdas = np.zeros(lists.shape[0])

    for i in range(lists.shape[0]):
        beta_ids = np.array(list(itertools.combinations(ids[lists[i].astype(np.bool)], 2)))
        if beta_ids.size > 0:
            beta_ids = beta_ids[:, 0] * K - beta_ids[:, 0] * (beta_ids[:, 0] + 1) // 2 + \
                       (beta_ids[:, 1] - beta_ids[:, 0]) - 1
        else:
            beta_ids = []
        log_lambdas[i] = mu + np.dot(lists[i], alphas) + np.sum(betas[beta_ids])  # only those lambdas with N_A>0

    data_term = np.dot(counts, log_lambdas)
    parameter_term = compute_lambda_some_betas(x, K)

    likelihood = data_term - parameter_term

    # print(likelihood)

    return -likelihood

@jit(nopython=True)
def factorial(x):
    return np.prod(np.arange(int(x+1)))

@jit(nopython=True)
def compute_lambda_some_betas(parameter, K, A=None):
    '''
    Compute sum A' lambda_A' over all \emptyset \neq A' \subseteq A in a dynamic programming fashion in time O(|A|^2) instead of naive O(2^|A|)
    :param parameter: vector containing mu, alphas and the betas (has to be a array to work with numba)
    :param A: the index vector A \subseteq {1,...,K} (has to be a array to work with numba)
    :return:  sum A' lambda_A' over all \emptyset \neq A' \subseteq A with the funny approximation
    '''

    # scale down to reduce overflow issues: https://scicomp.stackexchange.com/questions/1122/how-to-add-large-exponential-terms-reliably-without-overflow-errors
    # max_value = np.sum(np.abs(parameter + parameter[-1]*(parameter.size - 2))) #estimate for the maximum sum in the exponent
    # parameter = parameter-max_value

    exp_mu = np.exp(parameter[0])
    if A is None:
        exp_alphas = np.exp(parameter[1:K + 1])
    else:
        exp_alphas = np.exp(parameter[1 + A])

    betas = parameter[K + 1:]  # interprete as beta_12,beta_13,...beta_23,beta_24,...beta_K-1K

    summed_lambda_A = 0

    dp = np.copy(exp_alphas)

    summed_lambda_A += np.sum(exp_alphas)

    exp_betas = np.exp(betas)

    # misuse this y to compute the approximating beta terms
    y = np.zeros(1 + betas.size + 1)
    y[1:-1] = exp_betas

    for i in range(K - 1):

        # dp[:-i] = alphas[:-i]*dp[1:K-(i-1)]

        # log_lambda_A += np.power(beta, (i*(i-1))//2)*np.sum(dp[:-i])

        prev_sum = np.zeros(K)

        for j in range(K - 1, i, -1):
            prev_sum[j] = dp[j]


            alpha_idx = j - 1 - i
            beta_start_idx = alpha_idx * K - alpha_idx * (alpha_idx + 1) // 2
            beta_end_idx = beta_start_idx + K - alpha_idx - 1
            if j == K - 1:
                # use just beta_(j-1+i, j+i),... beta_(j-1+i,K-1) for indices 0,...,K-1
                beta_factor = np.exp(np.sum(betas[beta_start_idx:beta_end_idx]))
                dp[j] = exp_alphas[j - 1 - i] * np.sum(prev_sum[j:])
                dp[j] *= beta_factor
                summed_lambda_A += dp[j]
            elif i == 0:
                # use just average(beta_(j-1+i, j+i), ...,beta_(j-1+i,K-1)) for indices 0,...,K-1
                dp[j] = exp_alphas[j - 1 - i] * np.sum(prev_sum[j:] * exp_betas[beta_start_idx:beta_end_idx])
                summed_lambda_A += dp[j]
            else:
                averaging_factor = math.gamma(1 + beta_end_idx - beta_start_idx)/(math.gamma(1 + beta_end_idx - beta_start_idx - (i + 2))*math.gamma( 1+ i + 2))
                beta_factor = np.exp(singlebetalikelihood.compute_lambda_single_beta(y,
                                                np.arange(beta_start_idx, beta_end_idx), 0, (i+2)) / averaging_factor)
                dp[j] = exp_alphas[j - 1 - i] * np.sum(prev_sum[j:])
                dp[j] *= beta_factor
                summed_lambda_A += dp[j]


    summed_lambda_A *= exp_mu
    # summed_lambda_A *= np.exp(max_value)
    return summed_lambda_A
