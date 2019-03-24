# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.special import factorial
import sys
from scipy.optimize import minimize, LinearConstraint
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
    N = 6
    suspect_count = np.sum(counts)
    return suspects, counts, N, suspect_count


# In[3]:

np.random.seed(0)
# simulate data
N = 30
suspect_count = 10000

pool = np.ones(2 * N - 1)
pool[:N-1] = 0

# suspects = np.random.choice(pool,replace=False,size=N)


#suspects = np.zeros((suspect_count, N),dtype=bool)

probs = np.exp(-np.arange(N))
probs = np.append(np.ones(N-1),probs)
probs /= np.sum(probs)
for s in range(suspect_count):
    suspects[s] = np.random.choice(pool, replace=False, p=probs,size=N)
suspects, counts = np.unique(suspects, return_counts=True, axis=0)
print(counts)
print(counts.shape)
suspects, counts, N, suspect_count = UKData()


def compute_log_lambdas(mu,alphas,betas,lists):
    alphas = np.repeat([alphas], suspects.shape[0], axis=0)
    # print(alphas)
    # print(lists)
    log_lambdas = mu + np.sum(lists * alphas, axis=1)  # + \
    # np.sum(lists * betas, axis=1)

    betas_summed = np.zeros(lists.shape[0])
    for i in range(lists.shape[0])
        betas_summed[i] = np.sum(betas[suspects[i] > 0][:, suspects[i] > 0]) / 2
        # print(i/lists.shape[0])

    log_lambdas += betas_summed

    return log_lambdas

def compute_log_lambdas(x,A):
    return np.sum(A * x[:, np.newaxis], axis=0)



# In[76]:

#I forget on almost all funcutions here to use only one half of the betas matrix!!!!!

def likelihood(lists, counts, mu, alphas, betas):
    log_lambdas = compute_log_lambdas(mu,alphas,betas,lists)

    lambdas = np.exp(log_lambdas)
    # print(log_lambdas,lambdas)
    return np.sum(counts * log_lambdas - lambdas), lambdas


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
    log_lambdas = x[0] + np.sum(lists * alphas, axis=1)  # + \

    exp_params = np.exp(x)

    return -(np.sum(counts * log_lambdas) - exp_params[0] * np.prod(exp_params[1:] + 1))


# log exponential prior with parameter = 1. pdf = e^-x for x >= 0, 0 o/w
def prior_exp(mu, alphas, betas):
    all_params = np.append(np.append(mu, alphas), betas.flatten())
    if np.any(all_params < 0):
        return -np.inf
    return np.sum(-all_params)


# log_prior
def prior_almost_uniform(mu, alphas, betas):
    w = 20
    eps = 0.0001
    height = 1 / (w - 2 / 3 * eps)
    all_params = np.append(np.append(mu, alphas), betas.flatten())
    result = np.zeros(all_params.shape[0])

    x = all_params[(all_params > 0) & (all_params < eps)]
    result[(all_params > 0) & (all_params < eps)] = -height / (eps * eps) * x ** 2 + 2 * height / eps * x

    x = all_params[(all_params >= eps) & (all_params < w - eps)]
    result[(all_params >= eps) & (all_params < w - eps)] = height

    x = all_params[(all_params >= w - eps) & (all_params < w)]
    result[(all_params >= w - eps) & (all_params < w)] = -height / (eps * eps) * (w - x) ** 2 + 2 * height / eps * (
                w - x)

    return np.sum(np.log(result))


def hyperbole_barrier(mu, alphas, betas):
    lower = np.tril_indices(betas.shape[0], -1)
    all_params = np.append(np.append(mu, alphas), betas[lower].flatten())
    if np.any(all_params < 0):
        return -np.inf
    return -np.sum(10 / (all_params))


def derivative(lists, counts, mu, alphas, betas, lambdas):
    del_mu = np.sum(counts - lambdas)

    del_alphas = np.sum(lists * counts[:, np.newaxis] - lists * lambdas[:, np.newaxis], axis=0)

    A = lists * counts[:, np.newaxis]

    del_betas = np.zeros(betas.shape)

    for i in range(alphas.shape[0]):
        for j in range(i + 1, alphas.shape[0]):
            lists_with_beta = (A[:, i] > 0) & (A[:, j] > 0)
            if A[lists_with_beta].size != 0:
                del_betas[i, j] = np.sum(A[lists_with_beta,i] - lambdas[lists_with_beta])

    del_betas += del_betas.transpose()



    return del_mu, del_alphas, del_betas
    # print(del_betas)
    # print(del_alphas)


def neg_derivative_vectorized(x, A, counts):
    return -(np.sum(A*counts - A*np.exp(compute_log_lambdas(x,A)), axis = 1))# + 2*x

def neg_derivative_alphas_only(x,lists,counts):
    exp_params = np.exp(x)
    helper = np.prod(exp_params[1:] + 1)
    del_mu = np.sum(counts) - exp_params[0]*helper
    del_alphas = np.sum(lists * counts[:, np.newaxis], axis = 0) - exp_params[0]*helper*exp_params[1:]/(exp_params[1:]+1)

    return -np.append(del_mu,del_alphas)

def der_exp_prior(mu, alphas, betas):
    all_params = np.append(np.append(mu, alphas), betas.flatten())
    # if np.any(all_params < 0):
    #    return np.log(-1/np.sum(all_params[all_params < 0])
    return -1, -np.ones(alphas.shape), -np.ones(betas.shape)


def der_uniform_prior(mu, alphas, betas):
    w = 20
    eps = 0.0001
    height = 1 / (w - 2 / 3 * eps)
    all_params = np.append(np.append(mu, alphas), betas.flatten())
    result = np.zeros(all_params.shape[0])

    x = all_params[(all_params > 0) & (all_params < eps)]
    result[(all_params > 0) & (all_params < eps)] = -2 * height / (eps * eps) * x + 2 * height / eps
    result[(all_params > 0) & (all_params < eps)] /= -height / (eps * eps) * x ** 2 + 2 * height / eps * x

    x = all_params[(all_params >= w - eps) & (all_params < w)]
    result[(all_params >= w - eps) & (all_params < w)] = 2 * height / (eps * eps) * (w - x) - 2 * height / eps
    result[(all_params >= w - eps) & (all_params < w)] /= -height / (eps * eps) * (w - x) ** 2 + 2 * height / eps * (
                w - x)

    return result[0], result[1:alphas.shape[0] + 1], result[alphas.shape[0] + 1:].reshape(betas.shape)


def der_hyperbole_barrier(mu, alphas, betas):
    lower = np.tril_indices(betas.shape[0], -1)
    all_params = np.append(np.append(mu, alphas), betas[lower].flatten())
    if np.any(all_params < 0):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!ALERT!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    result = 10 / (all_params ** 2)
    der_betas = np.zeros(betas.shape)
    der_betas[lower] = result[alphas.shape[0] + 1:]
    der_betas += der_betas.transpose()
    return result[0], result[1:alphas.shape[0] + 1], der_betas


# In[ ]:

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


def do_own_gradient_descnet():

    max_p = -np.inf
    best_mu = 0
    for z in range(1):
        mu = 11#np.random.rand()*10
        alphas = np.array([-5.08597335, -2.96526883, -2.29015778, -2.56056138, -3.34577209, -5.04601935])#np.random.rand(N)
        betas = np.array([[ 0.,          1.49560395,  1.06822664, -0.01815479,  0.2219303,   0.44158877],
        [ 1.49560395,  0.,          0.16951484, -0.50182992, -2.79659005,  0.21746344],
        [ 1.06822664,  0.16951484,  0.,         -0.00505412, -1.16342458, 1.64563318],
        [-0.01815479, -0.50182992, -0.00505412,  0.,         -1.116265,    0.03034782],
        [ 0.2219303, -2.79659005, -1.16342458, -1.116265,    0.,         -0.71584414],
        [ 0.44158877,  0.21746344,  1.64563318,  0.03034782, -0.71584414,  0.        ]]) #np.random.rand(N, N)
        #betas = np.triu(betas, 1)
        #betas += betas.transpose()
        # print(betas,alphas,mu)
        p = 0

        print("mu:", mu)

        test_nus = 0.5 ** np.arange(-5, 30)  # 0.5,0.25,...

        for i in range(10000):
            p, lambdas = likelihood(suspects, counts, mu, alphas)

            #p += hyperbole_barrier(mu, alphas, betas)
            print(p)
            print("============")
            del_mu, del_alphas, del_betas = derivative(suspects, counts, mu, alphas, betas, lambdas)
            a, b, c = 0,0,0#der_hyperbole_barrier(mu, alphas, betas)
            del_mu += a
            del_alphas += b
            del_betas += c

            current_best_p = p
            best_nu = 0.00000001
            # stupid line search
            for nu in test_nus:
                temp_mu = mu + nu * del_mu
                temp_alphas = alphas + nu * del_alphas
                temp_betas = betas + nu * del_betas
                temp_p, _ = likelihood(suspects, counts, temp_mu, temp_alphas, temp_betas)
                #temp_p += hyperbole_barrier(temp_mu, temp_alphas, temp_betas)
                # print("    ", nu, temp_p)
                if temp_p > current_best_p:
                    current_best_p = temp_p
                    best_nu = nu
            mu += best_nu * del_mu
            alphas += best_nu * del_alphas
            betas += best_nu * del_betas

            # if i %100 == 0 or i < 10:
            print(best_nu, mu, np.exp(mu))
            #print(alphas)

            sys.stdout.flush()
            # print(alphas)
            # print(betas)
            if p > max_p:
                max_p = p
                best_mu = mu

        print("=============================================================")
        print("=============================================================")
        print("=============================================================")


    print(max_p, best_mu, np.exp(best_mu))

np.random.seed

A,beta_count = construct_parameter_matrix(suspects)

mu = np.random.rand()
alphas = np.zeros(N)#-10*np.ones(N)#np.random.rand(N)
betas = np.zeros(beta_count)#np.random.rand(beta_count)

x = np.append(np.append(mu, alphas),betas)

#bnds = [(0,None)]
#bnds += ([(None,None)]*(x.shape[0] - 1))

sol = minimize(neg_likelihood_vectorized, x, (A,counts),method='L-BFGS-B',jac=neg_derivative_vectorized,callback=callbackF)#, bounds=bnds)

print(sol)
print(np.exp(sol.x[0]))
x = np.append(mu,alphas)

sol = minimize(neg_likelihood_alphas_only, x, (suspects,counts),method='L-BFGS-B', jac=neg_derivative_alphas_only,callback=callbackF_alphas_only)

print(sol)
print(np.exp(sol.x[0]))

