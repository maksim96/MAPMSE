import numpy as np
import mapmse

def sample_poisson_counts_naive(list_count, parameter=None, parameter_to_lambdas=None):
    if parameter is None:
        parameter = np.zeros(1 + list_count + list_count*(list_count-1)//2)
        parameter[0] = 20 + 2*np.random.randn()
        parameter[1:list_count+1] = 1*np.random.randn(list_count)
        parameter[list_count+1:] = 0.1*np.random.randn(list_count*(list_count-1)//2)
        #parameter = np.clip(parameter, -3,3)
        #parameter /= np.abs(np.max(parameter))/10
    if parameter_to_lambdas is None:
        parameter_to_lambdas = mapmse.get_lambda_matrix(list_count)
    lambdas = np.exp(np.dot(parameter_to_lambdas, parameter))
    return parameter, np.random.poisson(lambdas, lambdas.size)


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