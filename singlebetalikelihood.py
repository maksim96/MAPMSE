import numpy as np
import datagenerator


def neg_likelihood(x, lists, counts):
    '''
    This is not the correct likelihood. All betas are assumed to be equal
    Expects to contain only those lists, where the count is > 0
    :param x: the current parameters [mu, alpha_1,...alpha_K,beta]
    :param lists: a 2d array consisting of index vector rows
    :param counts: array with all the counts for each index vector
    :return:
    '''
    K = len(x) - 2
    # print(alphas)
    # print(lists)

    mu = x[0]
    alphas = x[1:-1]
    beta = x[-1]

    beta_count = np.sum(lists, axis=1)
    beta_count = beta_count * (beta_count - 1) / 2

    log_lambdas = mu + np.dot(lists, alphas) + beta_count * beta #only those lambdas with N_A>0

    data_term = np.sum(counts * log_lambdas)
    parameter_term = datagenerator.compute_lambda_single_beta(x, np.arange(K))

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
    all_lambdas_summed = datagenerator.compute_lambda_single_beta(x, np.arange(K))
    grad[0] = np.sum(counts) - all_lambdas_summed

    all_over_two_sized_lambdas = datagenerator.compute_lambda_single_beta_for_grad(x) - np.exp(x[0])

    grad[1:-1] = np.dot(counts, lists) - (all_over_two_sized_lambdas + np.exp(x[0] + x[1:-1]))

    two_or_more = np.sum(lists, axis=1) >= 2
    beta_count = np.sum(lists[two_or_more], axis=1)
    beta_count = beta_count * (beta_count - 1) / 2

    grad[-1] = np.sum(counts[two_or_more] * beta_count) - all_over_two_sized_lambdas
    return -grad


if __name__ == "__main__":
    lists = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0],
                      [1, 0, 1],
                      [0, 1, 1],
                      [1, 1, 1]])

    counts = np.arange(1, 7)

    neg_likelihood(np.array([np.log(10), 0, 0, 0, np.log(0.5)]), lists, counts)
