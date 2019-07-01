import numpy as np
from numba import jit
#import numba

def neg_likelihood(x, lists, counts):
    '''
    This is not the correct likelihood. All betas are assumed to be equal
    Expects to contain only those lists, where the count is > 0
    :param x: the current parameters [mu, alpha_1,...alpha_K,beta]
    :param lists: a 2d array consisting of index vector rows
    :param counts: array with all the counts for each index vector
    :return:
    '''
    K = (len(x) -1)//2
    # print(alphas)
    # print(lists)

    mu = x[0]
    alphas = x[1:K+1]
    betas = x[K+1:]

    intersection_sizes = np.sum(lists, axis=1)
    log_lambdas = np.zeros(lists.shape[0])

    for i in range(lists.shape[0]):
        log_lambdas[i] = mu + np.dot(lists[i], alphas) + np.sum(np.arange(intersection_sizes[i]-1, -1, -1)*betas[lists[i].astype(np.bool)]) #only those lambdas with N_A>0

    data_term = np.dot(counts, log_lambdas)
    parameter_term = compute_lambda_linear_betas(x, K, np.arange(K))

    likelihood = data_term - parameter_term

    #print(likelihood)

    return -likelihood

def neg_derivative(x, lists, counts):
    '''
    see neg_likelihood
    :param x:
    :param lists:
    :param counts:
    :return:
    '''

    K = len(x) - 2
    grad = np.zeros_like(x)
    all_lambdas_summed = compute_lambda_single_beta(x)
    grad[0] = np.sum(counts) - all_lambdas_summed

    all_over_two_sized_with_correct_count_lambdas = compute_lambda_single_beta_for_grad(x)

    lambda_i = np.exp(x[0] + x[1:-1]) #lambda_{1}, lambda_{2},...

    all_lambda_i_summed = np.sum(lambda_i)


    log_lambdas = x[0] + np.dot(lists, x[1:-1]) + np.sum(lists, axis=1)*(np.sum(lists, axis=1)-1)/2 * x[-1]

    for i in range(K):
        grad[i+1] = np.dot(counts, lists[:,i]) - (lambda_i[i]+np.exp(x[i+1])*compute_lambda_single_beta(x, np.delete(np.arange(K), i), 1))

    #grad[1:-1] =

    two_or_more = np.sum(lists, axis=1) >= 2
    beta_count = np.sum(lists[two_or_more], axis=1)
    beta_count = beta_count * (beta_count - 1) / 2

    grad[-1] = np.dot(counts[two_or_more], beta_count) - all_over_two_sized_with_correct_count_lambdas
    return -grad

@jit(nopython=True)
def compute_lambda_linear_betas(parameter, K, A=None):
    '''
    Compute sum A' lambda_A' over all \emptyset \neq A' \subseteq A in a dynamic programming fashion in time O(|A|^2) instead of naive O(2^|A|)
    :param parameter: vector containing mu, alphas and the single_beta (has to be a array to work with numba)
    :param A: the index vector A \subseteq {1,...,K} (has to be a array to work with numba)
    :return:  sum A' lambda_A' over all \emptyset \neq A' \subseteq A
    '''

    #scale down to reduce overflow issues: https://scicomp.stackexchange.com/questions/1122/how-to-add-large-exponential-terms-reliably-without-overflow-errors
    #max_value = np.sum(np.abs(parameter + parameter[-1]*(parameter.size - 2))) #estimate for the maximum sum in the exponent
    #parameter = parameter-max_value

    exp_mu = np.exp(parameter[0])
    if A is None:
        exp_alphas = np.exp(parameter[1:1+K])
    else:
        exp_alphas = np.exp(parameter[1 + A])

    betas = parameter[1+K:]

    K = exp_alphas.size

    summed_lambda_A = 0

    dp = np.copy(exp_alphas)

    summed_lambda_A += np.sum(exp_alphas)

    exp_betas = np.exp(betas)

    for i in range(K - 1):

        # dp[:-i] = alphas[:-i]*dp[1:K-(i-1)]

        # log_lambda_A += np.power(beta, (i*(i-1))//2)*np.sum(dp[:-i])

        prev_sum = 0

        for j in range(K - 1, i, -1):
            prev_sum += dp[j]
            dp[j] = np.exp(betas[j-1-i]*(i+1))*exp_alphas[j - 1 - i] * prev_sum
            summed_lambda_A += dp[j]

    summed_lambda_A *= exp_mu
    #summed_lambda_A *= np.exp(max_value)
    return summed_lambda_A
