import numpy as np

#this is the correct likelihood function, written in a linear way instead of the full exponentially sized sum
def neg_likelihood_alphas_only(x,lists,counts):
    n = lists.shape[1]
    alphas = np.repeat([x[1:n+1]], lists.shape[0], axis=0)
    # print(alphas)
    # print(lists)
    log_lambdas = x[0] + np.sum(lists * alphas, axis=1)  #only those lambdas where the list count is nonzero.

    exp_params = np.exp(x)

    return -(np.sum(counts * log_lambdas) - exp_params[0] * np.prod(exp_params[1:] + 1) + exp_params[0])


def neg_derivative_alphas_only(x,lists,counts):
    exp_params = np.exp(x)
    helper =  np.prod(exp_params[1:] + 1)
    del_mu = np.sum(counts) - exp_params[0]*helper +exp_params[0]
    #this computation is indeed correct
    del_alphas = np.sum(lists * counts[:, np.newaxis], axis = 0) - exp_params[0]*helper*exp_params[1:]/(exp_params[1:]+1)

    return -np.append(del_mu,del_alphas)