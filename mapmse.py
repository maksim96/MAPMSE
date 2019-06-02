import numpy as np

import alphasonlylinearlikelihood
import datagenerator
from scipy.optimize import minimize
import skiplikelihood
import naivelikelihood
from naivelikelihood import neg_likelihood_naive, neg_derrivative_naive
import matplotlib.pyplot as plt
import scipy.stats as stats

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
            # np.seterr(all='raise')
            x = np.random.randn(param_count)
            enforce_zeros = np.zeros(x.size).astype(bool)
            # enforce_zeros[15:] = True
            # x[enforce_zeros] = 0
            # x[0] = 9.39
            # print(true_params)
            # print(counts_exp)
            fitted_exp_mu, _, _, _ = naivelikelihood.optimize_naive(counts_exp, N, x, enforce_zeros)
            fitted_exp_mu = np.exp(fitted_exp_mu.x[0])

            accuracy_naive = np.abs(np.exp(true_params[0]) - fitted_exp_mu) / np.exp(true_params[0])
            accuracies_naive.append(accuracy_naive)
            if accuracy_naive < error:
                correct_naive += 1

            ##skip version heuristic
            counts_indexer = naivelikelihood.get_index_matrix_alphas_only(N).T[:, 1:]
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
                                                        fitted_exp_mu,
                                                        np.abs(np.exp(true_params[0]) - fitted_exp_mu) / np.exp(
                                                            true_params[0]), accuracy_naive < error, \
                                                        approx_exp_mu,
                                                        np.abs(np.exp(true_params[0]) - approx_exp_mu) / np.exp(
                                                            true_params[0]), accuracy_approx < error))

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
        # ax.axvline(x=correct_naive / experiment_count, color='C0')
        # ax.axvline(x=correct_approx / experiment_count, color='C1')
        plt.yscale('log')
        plt.ylabel("Relative Error")
        plt.xlabel("Quantile")
        ax.legend()
        plt.savefig("test.png")


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
    first_try_UKData()
    synthetic_data_experiments()

