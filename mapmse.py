import numpy as np

import alphasonlylinearlikelihood
import datagenerator
from scipy.optimize import minimize

import linearbetalikelihood
import partiallikelihood
import skiplikelihood
import naivelikelihood
import somebetaslikelihood
from naivelikelihood import neg_likelihood_naive, neg_derrivative_naive
import matplotlib.pyplot as plt
import scipy.stats as stats
import singlebetalikelihood

plt.style.use('seaborn')


def synthetic_data_experiments():
    correct_naive = 0
    correct_approx = 0
    experiment_count = 100

    error = 0.05

    accuracies_naive = []
    accuracies_approx = []


    for i in range(experiment_count):
        N = 10
        param_count = 1 + N + N * (N - 1) // 2
        true_params, counts_exp = datagenerator.sample_poisson_counts_naive(N)
        counts = counts_exp[counts_exp > 0]
        #np.seterr(all='raise')
        x = np.random.randn(param_count)
        enforce_zeros = np.zeros(x.size).astype(bool)
        # enforce_zeros[15:] = True
        # x[enforce_zeros] = 0
        # x[0] = 9.39
        # print(true_params)
        # print(counts_exp)
        fitted_exp_mu,_,_,_ = naivelikelihood.optimize_naive(counts_exp, N, x, enforce_zeros)
        fitted_exp_mu = np.exp(fitted_exp_mu.x[0])

        accuracy_naive = np.abs(np.exp(true_params[0]) - fitted_exp_mu) / np.exp(true_params[0])
        accuracies_naive.append(accuracy_naive)
        if accuracy_naive < error:
            correct_naive += 1

        ##skip version heuristic
        counts_indexer = naivelikelihood.get_index_matrix_alphas_only(N).T[:,1:]
        counts_indexer = counts_indexer[counts_exp > 0]
        A, beta_count = skiplikelihood.construct_parameter_matrix(counts_indexer)

        mu = np.random.rand()
        alphas = np.zeros(N)  # -10*np.ones(N)#np.random.rand(N)
        betas = np.zeros(beta_count)  # np.random.rand(beta_count)

        x = np.append(np.append(mu, alphas), betas)

        approx_exp_mu = minimize(skiplikelihood.neg_likelihood_vectorized, x, (A, counts), method='L-BFGS-B',
                           jac=skiplikelihood.neg_derivative_vectorized).x[0]
        approx_exp_mu = np.exp(approx_exp_mu)
        accuracy_approx = np.abs(np.exp(true_params[0]) - approx_exp_mu) / np.exp(true_params[0])
        accuracies_approx.append(accuracy_approx)
        if accuracy_approx < error:
            correct_approx += 1

        print("%.2f; %.2f %.2f %r; %.2f %.2f %r" % (np.exp(true_params[0]), \
                    fitted_exp_mu, np.abs(np.exp(true_params[0]) - fitted_exp_mu) / np.exp(true_params[0]), accuracy_naive < error,\
                    approx_exp_mu, np.abs(np.exp(true_params[0]) - approx_exp_mu) / np.exp(true_params[0]), accuracy_approx < error))



    print(correct_naive / experiment_count)
    print(correct_approx / experiment_count)

    fig, ax = plt.subplots()


    stats.probplot(accuracies_naive, dist=stats.uniform, plot=plt)
    stats.probplot(accuracies_approx, dist=stats.uniform, plot=plt)

    # Remove the regression lines
    ax.get_lines()[1].remove()
    ax.get_lines()[2].remove()
    # Change colour of scatter
    ax.get_lines()[0].set_markerfacecolor('C0')
    ax.get_lines()[1].set_markerfacecolor('C1')
    ax.get_lines()[0].set_label("Full Optimization")
    ax.get_lines()[1].set_label("Heuristic")
    ax.get_lines()[0].set_markersize(5.0)
    ax.get_lines()[1].set_markersize(5.0)
    #ax.axvline(x=correct_naive / experiment_count, color='C0')
    #ax.axvline(x=correct_approx / experiment_count, color='C1')
    plt.yscale('log')
    plt.ylabel("Relative Error")
    plt.xlabel("Quantile")
    ax.legend()
    plt.savefig("test.png")

def printx(args):
    print(args.fun)

def comparison():
    np.random.seed(5)
    correct_naive = 0
    correct_approx = 0
    correct_alphas_only = 0
    experiment_count = 100

    error = 0.05

    accuracies_naive = []
    accuracies_approx = []
    accuracies_alphas_only = []
    accuracies_single_beta = []
    accuracies_linear_beta = []
    accuracies_some_beta = []

    for i in range(experiment_count):
        N = 4
        param_count = 1 + N + N * (N - 1) // 2
        true_params, counts_exp = datagenerator.sample_poisson_counts_naive(N)
        counts = counts_exp[counts_exp > 0]
        #np.seterr(all='raise')
        x = np.random.randn(param_count)
        enforce_zeros = np.zeros(x.size).astype(bool)
        # enforce_zeros[15:] = True
        # x[enforce_zeros] = 0
        # x[0] = 9.39
        # print(true_params)
        # print(counts_exp)

        #full mle
        fitted_exp_mu,_,_,_ = naivelikelihood.optimize_naive(counts_exp, N, x, enforce_zeros)
        fitted_exp_mu = np.exp(fitted_exp_mu.x[0])

        accuracy_naive = np.abs(np.exp(true_params[0]) - fitted_exp_mu) / np.exp(true_params[0])
        accuracies_naive.append(accuracy_naive)
        if accuracy_naive < error:
            correct_naive += 1

        ##skip version heuristic
        counts_indexer = naivelikelihood.get_index_matrix_alphas_only(N).T[:,1:]
        counts_indexer = counts_indexer[counts_exp > 0]
        A, beta_count = skiplikelihood.construct_parameter_matrix(counts_indexer)

        mu = np.random.rand()
        alphas = np.zeros(N)  # -10*np.ones(N)#np.random.rand(N)
        betas = np.zeros(beta_count)  # np.random.rand(beta_count)

        x = np.append(np.append(mu, alphas), betas)

        approx_exp_mu = minimize(skiplikelihood.neg_likelihood_vectorized, x, (A, counts), method='L-BFGS-B', jac=skiplikelihood.neg_derivative_vectorized).x[0]
        approx_exp_mu = np.exp(approx_exp_mu)
        accuracy_approx = np.abs(np.exp(true_params[0]) - approx_exp_mu) / np.exp(true_params[0])
        accuracies_approx.append(accuracy_approx)
        if accuracy_approx < error:
            correct_approx += 1

        #alphas_only
        x = np.append(mu, alphas)
        alphas_exp_mu = minimize(alphasonlylinearlikelihood.neg_likelihood_alphas_only, x, (counts_indexer, counts), method='L-BFGS-B', jac=alphasonlylinearlikelihood.neg_derivative_alphas_only).x[0]

        alphas_exp_mu = np.exp(alphas_exp_mu)
        accuracy_alphas= np.abs(np.exp(true_params[0]) - alphas_exp_mu) / np.exp(true_params[0])
        accuracies_alphas_only.append(accuracy_alphas)
        if accuracy_alphas < error:
            correct_alphas_only += 1


        # single beta
        counts_indexer = counts_indexer.astype(np.int)
        x = np.append(np.append(mu, np.random.randn(N)), np.random.randn())
        #x[-1] = 0
        single_betas_exp_mu = minimize(singlebetalikelihood.neg_likelihood, x, (counts_indexer, counts), method='L-BFGS-B', jac=singlebetalikelihood.neg_derivative).x[0]
        single_betas_exp_mu = np.exp(single_betas_exp_mu)
        accuracy_single_beta = np.abs(np.exp(true_params[0]) - single_betas_exp_mu) / np.exp(true_params[0])
        accuracies_single_beta.append(accuracy_single_beta)

        # linear betas
        x = np.append(np.append(mu, np.random.randn(N)), np.random.randn(N))
        # x[-1] = 0
        linear_betas_exp_mu = minimize(linearbetalikelihood.neg_likelihood, x, (counts_indexer, counts)).x[0]
        linear_betas_exp_mu = np.exp(linear_betas_exp_mu)
        accuracy_linear_beta = np.abs(np.exp(true_params[0]) - linear_betas_exp_mu) / np.exp(true_params[0])
        accuracies_linear_beta.append(accuracy_linear_beta)

        # some betas
        x = np.append(np.append(mu, np.random.randn(N)), np.zeros(N*(N-1)//2))#np.random.randn(N*(N-1)//2))
        # x[-1] = 0
        some_betas_exp_mu = minimize(somebetaslikelihood.neg_likelihood, x, (N, counts_indexer, counts), options={'disp':True}).x
        some_betas_exp_mu = np.exp(some_betas_exp_mu[0])
        accuracy_some_beta = np.abs(np.exp(true_params[0]) - some_betas_exp_mu) / np.exp(true_params[0])
        accuracies_some_beta.append(accuracy_some_beta)

        print("%.2f;     %.2f %.2f %r;       %.2f %.2f %r;      %.2f %.2f %r;      %.2f %.2f %r" % (np.exp(true_params[0]),\
                    fitted_exp_mu      , np.abs(np.exp(true_params[0]) - fitted_exp_mu) / np.exp(true_params[0]), accuracy_naive < error,\
                    approx_exp_mu      , np.abs(np.exp(true_params[0]) - approx_exp_mu) / np.exp(true_params[0]), accuracy_approx < error,\
                    alphas_exp_mu      , np.abs(np.exp(true_params[0]) - alphas_exp_mu) / np.exp(true_params[0]), accuracy_alphas < error,
                   single_betas_exp_mu, np.abs(np.exp(true_params[0]) - single_betas_exp_mu) / np.exp(true_params[0]), accuracy_single_beta < error))

        print(i)

    print(correct_naive / experiment_count)
    print(correct_approx / experiment_count)
    print(correct_alphas_only / experiment_count)

    print(accuracies_some_beta)

    fig, ax = plt.subplots()


    stats.probplot([num for num in accuracies_naive if num <= 100], dist=stats.uniform, plot=plt)
    stats.probplot([num for num in accuracies_approx if num <= 100], dist=stats.uniform, plot=plt)
    stats.probplot([num for num in accuracies_alphas_only if num <= 100], dist=stats.uniform, plot=plt)
    stats.probplot([num for num in accuracies_single_beta if num <= 100], dist=stats.uniform, plot=plt)
    stats.probplot([num for num in accuracies_linear_beta if num <= 100], dist=stats.uniform, plot=plt)
    stats.probplot([num for num in accuracies_some_beta if num <= 100], dist=stats.uniform, plot=plt)

    # Remove the regression lines
    ax.get_lines()[1].remove()
    ax.get_lines()[2].remove()
    ax.get_lines()[3].remove()
    ax.get_lines()[4].remove()
    ax.get_lines()[5].remove()
    ax.get_lines()[6].remove()
    # Change colour of scatter
    ax.get_lines()[0].set_markerfacecolor('C0')
    ax.get_lines()[1].set_markerfacecolor('C1')
    ax.get_lines()[2].set_markerfacecolor('C2')
    ax.get_lines()[3].set_markerfacecolor('C3')
    ax.get_lines()[4].set_markerfacecolor('C4')
    ax.get_lines()[5].set_markerfacecolor('C5')
    ax.get_lines()[0].set_label("Full Optimization")
    ax.get_lines()[1].set_label("Heuristic")
    ax.get_lines()[2].set_label("Alphas Only")
    ax.get_lines()[3].set_label("Single Beta")
    ax.get_lines()[4].set_label("Linear Betas")
    ax.get_lines()[5].set_label("Some Betas")
    ax.get_lines()[0].set_markersize(3.0)
    ax.get_lines()[1].set_markersize(3.0)
    ax.get_lines()[2].set_markersize(3.0)
    ax.get_lines()[3].set_markersize(3.0)
    ax.get_lines()[4].set_markersize(3.0)
    ax.get_lines()[5].set_markersize(3.0)

    #ax.axvline(x=correct_naive / experiment_count, color='C0')
    #ax.axvline(x=correct_approx / experiment_count, color='C1')
    plt.yscale('log')
    plt.ylabel("Relative Error")
    plt.xlabel("Quantile")
    ax.legend()
    plt.savefig("test new.png")

def comparison_partial():
    np.random.seed(5)
    correct_naive = 0
    correct_approx = 0
    correct_alphas_only = 0
    experiment_count = 50
    N = 13
    param_count = 1 + N + N * (N - 1) // 2

    error = 0.05

    accuracies_naive = []
    accuracies_approx = []
    accuracies_single_beta = []
    accuracies_alphas_only = []
    accuracies_partial = [[] for _ in range(N+1)]

    for i in range(experiment_count):
        true_params, counts_exp = datagenerator.sample_poisson_counts_naive(N)
        counts = counts_exp[counts_exp > 0]
        #np.seterr(all='raise')
        x = np.random.randn(param_count)
        enforce_zeros = np.zeros(x.size).astype(bool)
        # enforce_zeros[15:] = True
        # x[enforce_zeros] = 0
        # x[0] = 9.39
        # print(true_params)
        # print(counts_exp)


        counts_indexer = naivelikelihood.get_index_matrix_alphas_only(N).T[:,1:]
        counts_indexer = counts_indexer[counts_exp > 0]

        for t in range(1,11):
            fitted_exp_mu = partiallikelihood.optimize(x, N, t, counts_indexer, counts)
            fitted_exp_mu = np.exp(fitted_exp_mu.x[0])

            accuracy_partial = np.abs(np.exp(true_params[0]) - fitted_exp_mu) / np.exp(true_params[0])
            accuracies_partial[t-1].append(accuracy_partial)

        '''
        ##skip version heuristic
        A, beta_count = skiplikelihood.construct_parameter_matrix(counts_indexer)

        mu = np.random.rand()
        alphas = np.zeros(N)  # -10*np.ones(N)#np.random.rand(N)
        betas = np.zeros(beta_count)  # np.random.rand(beta_count)

        x = np.append(np.append(mu, alphas), betas)

        approx_exp_mu = minimize(skiplikelihood.neg_likelihood_vectorized, x, (A, counts), method='L-BFGS-B', jac=skiplikelihood.neg_derivative_vectorized).x[0]
        approx_exp_mu = np.exp(approx_exp_mu)
        accuracy_approx = np.abs(np.exp(true_params[0]) - approx_exp_mu) / np.exp(true_params[0])
        accuracies_approx.append(accuracy_approx)
        if accuracy_approx < error:
            correct_approx += 1



        #alphas_only
        x = np.append(mu, alphas)
        alphas_exp_mu = minimize(alphasonlylinearlikelihood.neg_likelihood_alphas_only, x, (counts_indexer, counts), method='L-BFGS-B', jac=alphasonlylinearlikelihood.neg_derivative_alphas_only).x[0]

        alphas_exp_mu = np.exp(alphas_exp_mu)
        accuracy_alphas= np.abs(np.exp(true_params[0]) - alphas_exp_mu) / np.exp(true_params[0])
        accuracies_alphas_only.append(accuracy_alphas)
        if accuracy_alphas < error:
            correct_alphas_only += 1

        # single beta
        counts_indexer = counts_indexer.astype(np.int)
        x = np.append(np.append(mu, np.random.randn(N)), np.random.randn())
        #x[-1] = 0
        single_betas_exp_mu = minimize(singlebetalikelihood.neg_likelihood, x, (counts_indexer, counts), method='L-BFGS-B', jac=singlebetalikelihood.neg_derivative).x[0]
        single_betas_exp_mu = np.exp(single_betas_exp_mu)
        accuracy_single_beta = np.abs(np.exp(true_params[0]) - single_betas_exp_mu) / np.exp(true_params[0])
        accuracies_single_beta.append(accuracy_single_beta)

        '''
        #full mle
        x = np.random.randn(param_count)
        enforce_zeros = np.zeros(x.size).astype(bool)
        fitted_exp_mu, _, _, _ = naivelikelihood.optimize_naive(counts_exp, N, x, enforce_zeros)
        fitted_exp_mu = np.exp(fitted_exp_mu.x[0])

        accuracy_naive = np.abs(np.exp(true_params[0]) - fitted_exp_mu) / np.exp(true_params[0])
        accuracies_naive.append(accuracy_naive)

        print(i)

    print(accuracies_partial)

    fig, ax = plt.subplots()

    for t in range(1,11):
        stats.probplot([min(i ,100) for i in accuracies_partial[t-1]], dist=stats.uniform, plot=plt)

    stats.probplot([i for i in accuracies_naive if i <= 100], dist=stats.uniform, plot=plt)
    #stats.probplot([i for i in accuracies_approx if i <= 100], dist=stats.uniform, plot=plt)
    #stats.probplot([i for i in accuracies_alphas_only if i <= 100], dist=stats.uniform, plot=plt)


    # Remove the regression lines
    ax.get_lines()[1].remove()
    ax.get_lines()[2].remove()
    ax.get_lines()[3].remove()
    ax.get_lines()[4].remove()
    ax.get_lines()[5].remove()
    ax.get_lines()[6].remove()
    ax.get_lines()[7].remove()
    ax.get_lines()[8].remove()
    ax.get_lines()[9].remove()
    ax.get_lines()[10].remove()
    ax.get_lines()[11].remove()

    # Change colour of scatter
    ax.get_lines()[0].set_markerfacecolor('red')
    ax.get_lines()[1].set_markerfacecolor('green')
    ax.get_lines()[2].set_markerfacecolor('blue')
    ax.get_lines()[3].set_markerfacecolor('orange')
    ax.get_lines()[4].set_markerfacecolor('black')
    ax.get_lines()[5].set_markerfacecolor('yellow')
    ax.get_lines()[6].set_markerfacecolor('white')
    ax.get_lines()[7].set_markerfacecolor('brown')
    ax.get_lines()[8].set_markerfacecolor('purple')
    ax.get_lines()[9].set_markerfacecolor('cyan')
    ax.get_lines()[9].set_markerfacecolor('grey')
    ax.get_lines()[0].set_label("Partial 1/7")
    ax.get_lines()[1].set_label("Partial 2/7")
    ax.get_lines()[2].set_label("Partial 3/7")
    ax.get_lines()[3].set_label("Partial 4/7")
    ax.get_lines()[4].set_label("Partial 5/7")
    ax.get_lines()[5].set_label("Partial 6/7")
    ax.get_lines()[6].set_label("Partial 7/7")
    ax.get_lines()[7].set_label("Partial 8/7")
    ax.get_lines()[8].set_label("Partial 9/7")
    ax.get_lines()[9].set_label("Partial 10/7")
    ax.get_lines()[10].set_label("Naive")
    #ax.get_lines()[8].set_label("Approx")
    #ax.get_lines()[9].set_label("Alphas Only")
    ax.get_lines()[0].set_markersize(4.0)
    ax.get_lines()[1].set_markersize(4.0)
    ax.get_lines()[2].set_markersize(11.0)
    ax.get_lines()[3].set_markersize(10.0)
    ax.get_lines()[4].set_markersize(9.0)
    ax.get_lines()[5].set_markersize(8.0)
    ax.get_lines()[6].set_markersize(7.0)
    ax.get_lines()[7].set_markersize(6.0)
    ax.get_lines()[8].set_markersize(5.0)
    ax.get_lines()[9].set_markersize(4.0)
    ax.get_lines()[10].set_markersize(3.0)

    #ax.axvline(x=correct_naive / experiment_count, color='C0')
    #ax.axvline(x=correct_approx / experiment_count, color='C1')
    plt.yscale('log')
    plt.ylabel("Relative Error")
    plt.xlabel("Quantile")
    ax.legend()
    plt.savefig("test der.png")

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
    suspects, counts, N, suspect_count,counts_exp = datagenerator.UKData()

    A,beta_count = skiplikelihood.construct_parameter_matrix(suspects)

    mu = np.random.rand()
    alphas = np.zeros(N)#-10*np.ones(N)#np.random.rand(N)
    betas = np.zeros(beta_count)#np.random.rand(beta_count)

    x = np.append(np.append(mu, alphas),betas)

    sol_all = minimize(skiplikelihood.neg_likelihood_vectorized, x, (A, counts), method='L-BFGS-B', jac=skiplikelihood.neg_derivative_vectorized)


    x = np.append(mu,alphas)

    sol_alphas_only = minimize(alphasonlylinearlikelihood.neg_likelihood_alphas_only, x, (suspects, counts), method='L-BFGS-B', jac=alphasonlylinearlikelihood.neg_derivative_alphas_only)

    return sol_all, sol_alphas_only


if __name__ == '__main__':
    comparison_partial()

