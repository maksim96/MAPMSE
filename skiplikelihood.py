import numpy as np

def compute_log_lambdas(x,A):
    return np.sum(A * x[:, np.newaxis], axis=0)

#this likelihood is merely an approximation of the real likelihood, as we drop all summands -lambda_A if N_A is 0.
def neg_likelihood_vectorized(x, A, counts):
    log_lambdas = compute_log_lambdas(x,A)

    lambdas = np.exp(log_lambdas)
    # print(log_lambdas,lambdas)
    return -np.sum(counts * log_lambdas - lambdas)# + np.dot(x,x)

def neg_derivative_vectorized(x, A, counts):
    return -(np.sum(A*counts - A*np.exp(compute_log_lambdas(x,A)), axis = 1))# + 2*x

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

