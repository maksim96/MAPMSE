import numpy as np
import mapmse
import naivelikelihood
from scipy.spatial.distance import pdist

def sample_poisson_counts_naive(list_count, parameter=None, parameter_to_lambdas=None):
    if parameter is None:
        parameter = np.zeros(1 + list_count + list_count*(list_count-1)//2)
        parameter[0] = 10 + 2*np.random.randn()
        parameter[1:list_count+1] = -1 + np.random.randn(list_count)
        parameter[list_count+1:] = -5 +2*np.random.randn(list_count*(list_count-1)//2)
        #parameter = np.clip(parameter, -3,3)
        #parameter /= np.abs(np.max(parameter))/10
    if parameter_to_lambdas is None:
        parameter_to_lambdas = naivelikelihood.get_lambda_matrix(list_count)
    lambdas = np.exp(np.dot(parameter_to_lambdas, parameter))
    return parameter, np.random.poisson(lambdas, lambdas.size)

def compute_lambda_single_beta(parameter, A):
    '''
    Compute sum A' lambda_A' over all A' \superseteq A in a dynamic programming fashion in time O(|A|^2) instead of naive O(2^|A|)
    :param parameter: vector containing mu, alphas and the single_beta
    :param A: the index set A \subseteq {1,...,K}
    :return: lambda_A
    '''

    exp_mu = np.exp(parameter[0])
    exp_alphas = np.exp(parameter[A])
    exp_beta = np.exp(parameter[-1])



    K = exp_alphas.size

    summed_lambda_A = 0

    dp = np.copy(exp_alphas)  # alpha_1+...alpha_i, alpha_2+...+alpha_i ,..., alpha_i

    summed_lambda_A += 1 + np.sum(dp)

    for i in range(exp_alphas.size-1):

        #dp[:-i] = alphas[:-i]*dp[1:K-(i-1)]

        #log_lambda_A += np.power(beta, (i*(i-1))//2)*np.sum(dp[:-i])

        prev_sum = 0

        for j in range(K-1,i,-1):
            prev_sum += dp[j]
            dp[j] = exp_alphas[j-1-i]*prev_sum
            summed_lambda_A += np.power(exp_beta, (i*(i-1))//2)*dp[j]

    summed_lambda_A *= exp_mu
    return summed_lambda_A

def __sample_poisson_recursive__(parameter, K, current_population_size=0, inside=set(), outside=set(), lists=[], counts=[]):
    candidates = list(set(range(K))-inside-outside)
    if (len(candidates)== K):
        #first poisson draw
        lambda_all = compute_lambda_single_beta(parameter, range(K))
        current_population_size = np.random.poisson(lambda_all)
        lists = [[1] * K]
        counts = [current_population_size]

    lambdas = []

    for c,i in enumerate(candidates):
        lambdas.append(compute_lambda_single_beta(parameter, candidates[i+1:]))

    subset_populations = np.random.multinomial(current_population_size, lambdas/np.sum(lambdas), size=1).flatten()

    for c, i in enumerate(candidates):
        if subset_populations[i]>0:
            intersection_index_vector = np.zeros(K).astype(np.bool)
            intersection_index_vector[c] = 1
            intersection_index_vector[list(inside)] = 1

            lists.append(list(intersection_index_vector))
            counts.append(subset_populations[i])

            __sample_poisson_recursive__(parameter, K, current_population_size, inside|set([c]), set(candidates[:i]),lists,counts)





def sample_poisson_counts_single_beta(list_count, parameter=None, parameter_to_lambdas=None):
    '''
    Here we force all beta_ij to be equal to one single beta
    :param list_count:
    :param parameter:
    :param parameter_to_lambdas:
    :return:
    '''
    if parameter is None:
        parameter = np.zeros(1 + list_count + 1)
        parameter[0] = 10 + 2*np.random.randn() #mu
        parameter[1:-1] = -1 + np.random.randn(list_count) #alphas
        parameter[-1] = -5 +2*np.random.randn() # beta


    __sample_poisson_recursive__(parameter, list_count)



#betas decrease with distance
def sample_poisson_counts_map_style(list_count):
    coordinates = np.random.rand(list_count, 2)*1000
    pairwise_distances = pdist(coordinates, 'sqeuclidean')
    mu = np.abs(20 * np.random.randn())
    alphas = np.abs(np.random.randn(list_count))
    betas = np.log(1/pairwise_distances)

    parameter = np.append(np.append(mu, alphas), betas)

    _, counts_exponential_representation = sample_poisson_counts_naive(list_count, parameter)




def sample_suspect_to_list_naive(list_count=50, suspect_count=100000):
    # In[3]:

    np.random.seed(0)
    # simulate data

    pool = np.ones(2 * list_count - 1)
    pool[:list_count - 1] = 0

    suspects = np.zeros((suspect_count, list_count), dtype=bool)

    probs = np.exp(-np.arange(list_count))
    probs = np.append(np.ones(list_count - 1), probs)
    probs /= np.sum(probs)
    for s in range(suspect_count):
        suspects[s] = np.random.choice(pool, replace=False, p=probs, size=list_count)
    suspects, counts = np.unique(suspects, return_counts=True, axis=0)

    return suspects, counts

def UKData():
    intersection_indexer = np.array([[1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                         [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
                         [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                         [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0]]).transpose()

    counts = np.array([54, 463, 907, 695, 316, 57, 15, 19, 3, 56, 19, 1, 3, 69, 10, 31, 8, 6, 1, 1, 1, 4, 3, 1, 1])

    '''
    c = np.zeros((25,8)) 
    c[:,2:] = np.fliplr(suspects.T)
    c =c.astype(np.bool) 
    d = np.zeros(64) 
    d[np.packbits(c,axis=1).flatten()] = counts     #only works for maximum 8bith 
    d = d.astype(np.int) #how to get count_all
    count_all=d[1:] #this is probably needed
    '''
    counts_exponential_representation =np.array([ 54, 463,  15, 907,  19,  56,   1, 695,   3,  19,   1,  69,   0,
         4,   1, 316,   0,   1,   0,  10,   0,   0,   0,   8,   0,   0,
         0,   0,   0,   0,   0,  57,   0,   3,   0,  31,   0,   3,   0,
         6,   0,   0,   0,   1,   0,   0,   0,   1,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0])

    list_count = 6
    total_suspect_count = np.sum(counts)
    return intersection_indexer, counts, list_count, total_suspect_count, counts_exponential_representation


compute_lambda_single_beta(np.array([2,1,2,3,4,1]), np.array([1,2,3,4]))

sample_poisson_counts_single_beta(6)