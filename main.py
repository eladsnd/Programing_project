from sklearn.datasets import make_blobs
import argparse
import numpy as np
import math
import mykmeanssp as km
import matplotlib.pyplot as plt

"""
the constants  
params: max_iter - maximum iterations for k-means++
        max_capacity_n - the maximum capacity of data points our program can run in 5 minutes
        max_capacity_n - the maximum capacity of clusters (for the max_capacity_n) our program can run in 5 minutes
"""
max_iter = 300
max_capacity_n = 540
max_capacity_k = 20

"""
create the Weighted Adjacency Matrix
params: original_data- the observations from make_blobs, 
        n- the number of the observations
return: W- the weighted adjacency matrix
"""


def create_w(original_data, n):
    W = np.zeros(shape=(n, n))
    for i in range(n):
        W[i] = np.exp(np.linalg.norm(original_data - original_data[i], axis=1) / (-2))
    np.fill_diagonal(W, 0)
    return W


"""
create the Diagonal Degree Matrix
params: w- the weighted adjacency matrix from create_w function
return: diag_as_matrix- the diagonal degree matrix
"""


def create_d(w):
    diag_w = np.einsum('ij->i', w)
    diag_as_matrix = np.diag(diag_w)
    return diag_as_matrix


"""
create the Normalized Graph Laplacian
params: w_matrix- the weighted adjacency matrix
        d_matrix- the diagonal degree matrix
        n- the number of the observations
return: l_out= I - (d_matrix^(-1/2) @ w_matrix @ d_matrix^(-1/2))
"""


def create_lnorm(d_matrix, w_matrix, n):
    i = np.identity(n)
    #for diagonal matrix inv = 1/diag
    d_inv_sqrt = np.linalg.inv(np.sqrt(d_matrix))
    l_out = i - ((d_inv_sqrt.dot(w_matrix)).dot(d_inv_sqrt))
    return l_out


"""
The Modified Gram-Schmidt Algorithm
params: a- matrix as np-array
        n- the number of the observations
return: q - the orthogonal matrix
        r - the upper triangular matrix
"""


def mgs(a, n):
    u = a.copy()
    r = np.zeros((n, n))
    q = np.zeros((n, n))
    for i in range(n-1):
        r[i][i] = np.linalg.norm(u[:, i])
        #if r[i][i]=0 there will be a zero column in the i column of Q
        if r[i][i] != 0:
            q[:, i] = np.divide(u[:, i], r[i][i])
        #the area of the row that needs changing
        to_change = q[:, i]
        r[i, i + 1:] = np.einsum('i,ij->j', to_change, u[:, i+1:])
        u[:, i + 1:] -= np.einsum('i,j->ji', r[i, i + 1:], to_change)
    return q, r


"""
The QR Iteration Algorithm
params: a- matrix as np-array
        n- the number of the observations
return: eigenvalues- whose diagonal elements approach the eigenvalues of a.
        eigenvectors- whose columns approach the eigenvectors of a.
"""


def qr_iteration(a, n):
    # a'
    eigenvalues = a.copy()
    # Q'
    eigenvectors = np.identity(n)
    for i in range(n):
        q, r = mgs(eigenvalues, n)
        # |Q'|
        eigenvalues = np.dot(r, q)
        eigenvectors_abs = np.abs(eigenvectors)
        # |Q'Q|
        target = np.dot(eigenvectors, q)
        target_q_abs = np.abs(target)
        # |Q'|-|Q'Q|
        ans = np.subtract(eigenvectors_abs, target_q_abs)
        ans = abs(ans)
        if np.all(ans <= 0.0001):
            return eigenvalues, eigenvectors
        eigenvectors = target
    return eigenvalues, eigenvectors


"""
calculate k
params: diagonal_eigenvalues- whose diagonal elements approach the eigenvalues of a.
        n- the number of the observations
return: return_val- the value of k that was found by the Eigen-gap Heuristic
"""


def find_k(diagonal_eigenvalues, n):
    diagonal_eigenvalues_as_array = np.diag(diagonal_eigenvalues)
    sorted_array_eigenvalues = np.sort(diagonal_eigenvalues_as_array)
    eigen1 = sorted_array_eigenvalues[: -1]
    eigen2 = sorted_array_eigenvalues[1:]
    eigen_gaps = np.abs(eigen1 - eigen2)
    relevant_eigen_gaps = eigen_gaps[:int(np.ceil(n / 2))]
    return_val = np.argmax(relevant_eigen_gaps)+1
    return return_val


"""
calculate U
params: eigenvalues- whose diagonal elements approach the eigenvalues of a.
        eigenvectors- whose columns approach the eigenvectors of a.
        k- the number of clusters
return: U- the matrix U whose columns approach the eigenvectors of l_norm
"""


def eigen(eigenvectors, k, eigenvalues):
    sort_index = np.argsort(eigenvalues).T
    eigenvectors = eigenvectors[:, sort_index]
    u = (eigenvectors[:, :k]).copy()
    return u


"""
re-normalizing the matrix U
params: U- the matrix U
return: t- the matrix U after re-normalizing it
"""


def create_t(u):
    x_norm = np.einsum('ij,ij->i', u, u)
    if x_norm.all() == 0:
        for i in range(x_norm.shape[0]):
            x_norm[i] = 1
    t = u / np.sqrt(x_norm[:, np.newaxis])
    return t


"""
generating t from data. 
If random, find k.
params: data- np-array with the samples made by make_blobs
        k- the number of clusters given by the user
        n- the number of the observations
        r- representing Random's boolean value.
return: t- the training matrix for spectral clustering
        k- the number of clusters       
"""


def training_mat(data, k, n, r):
    w_matrix = create_w(data, n)
    d_matrix = create_d(w_matrix)
    l_norm = create_lnorm(d_matrix, w_matrix, n)
    eigenvalues, eigenvectors = qr_iteration(l_norm, n)
    if r:
        k = find_k(eigenvalues, n)
    eigenvalues = np.diag(eigenvalues)
    u = eigen(eigenvectors, k, eigenvalues)
    t = create_t(u)
    return t, k


"""
creating the data and clusters with k and n parameters that was given by he user
params: data_t- the training matrix for spectral clustering
        k- the number of clusters 
        n- the number of the observations
return: clusters- the clusters that was made
        cluster_txt- the clusters file
"""


def spectral_operation(data_t, k, n):
    clusters = kmeans_operation(data_t, k, k, n)
    cluster_txt = clusters_txt(k, n, clusters)
    return clusters, cluster_txt


"""
creating the data and clusters with k and n parameters that was given by he user
params: data- the observations 
        ind_vec- working vector size k
        initial_centroids- working matrix size k*d
        n- the number of the observations
        k- the number of clusters         
return: initial_centroids- representing the initial centroids for kmeans++ algorithm
"""


def possibilities(data, initial_centroids, n, k):
    np.random.seed(0)
    f = np.random.choice(n, 1)
    initial_centroids[0] = data[f]
    distance_vec = np.zeros(n, dtype=np.float64)
    min_distance_vec = np.zeros(n, dtype=np.float64)
    prob = np.zeros(n, dtype=np.float64)
    if k > 1:
        min_distance_vec = np.sum((data - initial_centroids[0]) ** 2, axis=1)
        sum_m = np.sum(min_distance_vec)
        prob = min_distance_vec / sum_m
        f = np.random.choice(n, 1, p=prob)
        initial_centroids[1] = data[f]
        for cent in range(2, k):
            distance_vec = np.sum((data - initial_centroids[cent - 1]) ** 2, axis=1)
            min_distance_vec = np.minimum(min_distance_vec, distance_vec)
            sum_m = np.sum(min_distance_vec)
            prob = min_distance_vec / sum_m
            f = np.random.choice(n, 1, p=prob)
            initial_centroids[cent] = data[f]
    return initial_centroids


"""
creating the data and clusters with k and n parameters that was given by he user
params: data- the observations 
        k- the number of clusters 
        d- the dimension of each observation
        n- the number of the observations
return: clusters- cluster[i] represent the location of  each observation in data, 
                  that belong to that cluster
"""


def kmeans_operation(data, k, d, n):
    initial_centroids = np.zeros((k, d), dtype=float)
    working_data = data.copy()
    ind = possibilities(working_data, initial_centroids, n, k)
    points = working_data.tolist()
    centroid = ind.tolist()
    clusters = km.kmeans_pp(k, n, d, max_iter, points, centroid)
    return clusters


"""
creating the data and clusters with k and n parameters that was given by he user or with random k and n parameters
params: n- the number of the observations
        k- the number of clusters
        d- the dimension of each observation
        r- representing Random's boolean value.
return: data_matrix- the data that was generated from make_blobs
        k- the number of clusters was generated by make_blobs
        n- the number of the observations was generated by make_blobs
        blob_clusters- the clusters that was generated from make_blobs
"""


def create_blobs(n, k, d, r):
    if r:
        k = np.random.randint(low=math.ceil(max_capacity_k / 2), high=max_capacity_k)
        n = np.random.randint(low=math.ceil(max_capacity_n / 2), high=max_capacity_n)
        if k >= n:
            k = n-1
    else:
        if k < 1 or n < 1 or k >= n:
            print("The arguments you have given are invalid. Please make sure that: " + '\n')
            print("1. k < n." + '\n')
            print("2. both k and n are at least 1, and both of them are natural numbers only." + '\n')
            exit(0)
    data_matrix, blob_clusters = make_blobs(n_samples=n, centers=k, n_features=d)
    return data_matrix, k, n, blob_clusters


"""
creating the clusters for txt file
params: k- the number of clusters 
        n- the number of the observations
        clusters- the clusters that were generated
return: clusters_data- the clusters for txt file
"""


def clusters_txt(k, n, clusters):
    clusters_data = [[] for j in range(k)]
    for i in range(n):
        num_of_cluster = clusters[i]
        clusters_data[num_of_cluster].append(i)
    return clusters_data


"""
creating the txt file
params: data_to_file- the observations 
        blob_clusters- clusters that were generated by make_blobs
        spectral_clusters- clusters that were generated by nsc
        kmeans_clusters- clusters that were generated by Kmeans++
        n- the number of the observations
        k- the number of clusters 
        d- the number of dimensions 
"""


def write_to_file(data_to_file, blob_clusters, spectral_clusters, kmeans_clusters, n, k, d):
    #data file
    file = open("data.txt", 'w')
    for i in range(n):
        my_string = ','.join([str(data_to_file[i][j]) for j in range(d)]) + ',' + str(blob_clusters[i])
        file.write(my_string + '\n')
    file.close()
    #clusters file
    f = open("clusters.txt", 'w')
    f.write(str(k) + '\n')
    for i in spectral_clusters:
        my_string = ','.join([str(x) for x in i])
        f.write(my_string + '\n')
    for i in kmeans_clusters:
        my_string = ','.join([str(x) for x in i])
        f.write(my_string + '\n')
    f.close()


"""
calculate the Jaccard Measure
params: made_by_us_clusters- the clusters made by the Kmeans++ Algorithm
        make_blobs_clusters - the clusters made by make_blobs
return:the Jaccard Measure for arr1, arr2
"""


def jaccard(make_blobs_clusters, made_by_us_clusters):
    numerator = 0
    denominator = 0
    length = len(make_blobs_clusters)
    for i in range(length):
        for j in range(i+1, length):
            if (make_blobs_clusters[i] == make_blobs_clusters[j] and
                    made_by_us_clusters[i] == made_by_us_clusters[j]):
                numerator += 1
            if (make_blobs_clusters[i] == make_blobs_clusters[j] or
                    made_by_us_clusters[i] == made_by_us_clusters[j]):
                denominator += 1
    if denominator == 0:
        return 1.0
    return numerator / denominator


"""
create the visualization of the algorithm's results
params: k- the number of clusters was found by the mgs algorithm
        og_k- the number of clusters was given by the user
        dim- the dimension of each observation
        spectral_from_kmeans- the clusters made by the Kmeans++ Algorithm and the Eigen-gap Heuristic
        kmeans_from_kmeans- the clusters made by the Kmeans++ Algorithm and the observations made by make_blobs
        data- np-array with the samples made by make_blobs
        n- the number of the observations
        spectral_jaccard- the Jaccard Measure for clusters made by mgs
        kmeans_jaccard- the Jaccard Measure for clusters made by kmeans++
        labels- the clusters made by make_blobs
"""


def create_pdf_plot(k, og_k, dim, kmeans_from_kmeans, spectral_from_kmeans, data, n, spectral_jaccard, kmeans_jaccard):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 8)
    if dim == 2:
        km_plt = fig.add_subplot(gs[0:2, 4:7])
        km_plt.set_title("k-means")
        nsc_plt = fig.add_subplot(gs[0:2, 0:3])
        nsc_plt.set_title("Normalized Spectral Clustering")
        km_plt.scatter(data[:, 0], data[:, 1], c=kmeans_from_kmeans, s=50, cmap='plasma')
        nsc_plt.scatter(data[:, 0], data[:, 1], c=spectral_from_kmeans, s=50, cmap='plasma')
    if dim == 3:
        km_plt = fig.add_subplot(gs[0:2, 4:7], projection='3d')
        km_plt.set_title("k-means")
        nsc_plt = fig.add_subplot(gs[0:2, 0:3], projection='3d')
        nsc_plt.set_title("Normalized Spectral Clustering")
        km_plt.scatter(data[:, 0], data[:, 1], data[:, 2], c=kmeans_from_kmeans, s=50, cmap='plasma')
        nsc_plt.scatter(data[:, 0], data[:, 1], data[:, 2], c=spectral_from_kmeans, s=50, cmap='plasma')
    fig.suptitle(f'Data was generated from the values: \n '
                 f'n = {n} k = {og_k}\nThe k that was used for both algorithms was {k}\n'
                 f'The Jaccard measure for Spectral Clustering: {spectral_jaccard}\n'
                 f'The Jaccard measure for K-means Clustering: {kmeans_jaccard} ', x=0.5, y=0.25)
    fig.savefig('my_figure.pdf')


'''
main function
'''


def main():
    print("max values are: n=" + str(max_capacity_n) + "k=" + str(max_capacity_k))
    parser = argparse.ArgumentParser()
    parser.add_argument("K", type=int, help="K centroids")
    parser.add_argument("N", type=int, help="N data")
    parser.add_argument('--Random', dest='Random', action='store_true')
    parser.add_argument('--no-Random', dest='Random', action='store_false')
    parser.set_defaults(Random=True)
    args = parser.parse_args()
    K = args.K
    N = args.N
    og_k = K
    Random = args.Random
    d = np.random.choice([2, 3])
    #create blobs
    data, K, N, blob_clusters = create_blobs(N, K, d, Random)
    if not Random:
        og_k = K
    #create training matrix
    t, K = training_mat(data, K, N, Random)
    #k-means++ and generate data in right format
    kmeans_clusters_from_kmeans = kmeans_operation(data, K, d, N)
    kmeans_clusters_txt = clusters_txt(K, N, kmeans_clusters_from_kmeans)
    spectral_clusters_from_kmeans, spectral_cluster_txt = spectral_operation(t, K, N)
    #calculate jaccard
    kmeans_jaccard_14 = jaccard(blob_clusters, kmeans_clusters_from_kmeans)
    kmeans_jaccard = float("{:.2f}".format(kmeans_jaccard_14))
    spectral_jaccard_14 = jaccard(blob_clusters, spectral_clusters_from_kmeans)
    spectral_jaccard = float("{:.2f}".format(spectral_jaccard_14))
    #create output
    write_to_file(data, blob_clusters, spectral_cluster_txt, kmeans_clusters_txt, N, K, d)
    create_pdf_plot(K, og_k, d, kmeans_clusters_from_kmeans, spectral_clusters_from_kmeans, data, N, spectral_jaccard,
                    kmeans_jaccard)


if __name__ == '__main__':
    main()
