# coding: utf-8

# In[1]:


import numpy as np
import datagenerator

# In[2]:
import firsttrylikelihood
import naivelikelihood
from naivelikelihood import neg_likelihood_naive, neg_derrivative_naive



def synthetic_data_experiments():
    correct = 0
    experiment_count = 100
    for i in range(experiment_count):
        N = 15
        param_count = 1 + N + N * (N - 1) // 2
        true_params, counts_exp = datagenerator.sample_poisson_counts_naive(N)
        # np.seterr(all='raise')
        x = np.random.randn(param_count)
        enforce_zeros = np.zeros(x.size).astype(bool)
        # enforce_zeros[15:] = True
        # x[enforce_zeros] = 0
        # x[0] = 9.39
        # print(true_params)
        # print(counts_exp)
        fitted_exp_mu = naivelikelihood.optimize_naive(counts_exp, N, x, enforce_zeros)
        if np.abs(np.exp(true_params[0]) - fitted_exp_mu) / np.exp(true_params[0]) < 0.05:
            correct += 1
        print(np.exp(true_params[0]), fitted_exp_mu,
              np.abs(np.exp(true_params[0]) - fitted_exp_mu) / np.exp(true_params[0]),
              np.abs(np.exp(true_params[0]) - fitted_exp_mu) / np.exp(true_params[0]) < 0.05)

    print(correct / experiment_count)

def exponential_representation_UKData_experiments():
    intersection_indexer, counts, list_count, total_suspect_count, counts_exponential_representation = datagenerator.UKData()
    x = np.random.randn(1 + 6 + 15)  # one mu, 6 alphas, 15 betas
    sol, parameter_indexer, lambda_indexer, enforce_zero = naivelikelihood.optimize_naive(counts_exponential_representation, list_count, x)

    print(sol)
    print(sol.x)

    y = np.copy(sol.x)

    useles_parameters = np.where(np.dot(parameter_indexer, counts_exponential_representation) == 0)[0]

    y[useles_parameters] = -np.inf
    print(y)
    print(neg_likelihood_naive(y, parameter_indexer, lambda_indexer, counts_exponential_representation, enforce_zero) - sol.fun)

def first_try_UKData():
    sol, sol_alhpas_only = firsttrylikelihood.old_style_UK_fitting()
    print(sol.fun, sol.x)
    print(sol_alhpas_only.fun, sol_alhpas_only.x)

if __name__ == '__main__':
    first_try_UKData()

