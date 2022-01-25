import sklearn.cluster as skcl
import numpy as np
import matplotlib.pyplot as plt
import clustering
from sklearn.metrics import silhouette_score
from scipy import sparse


def plot_clustering(data,labels):
    """Get 2d scatter plot of clustering with colors indicating cluster membership.
    Used for plotting toy data set clusterings (e.g. figure 1.1).
    """
    clusters = np.unique(labels)
    for c in clusters:
        plt.scatter(data[labels == c, 0], data[labels == c, 1], label = (f"cluster {c}" if c >= 0 else "noise"))
    plt.legend()
    plt.show()

def plot_elbow_k(data):
    """Plot elbow graph to determine k in k-means (section 2.4.1).
    """
    krange = range(1,11)
    inertia = []
    for k in krange:
        c = skcl.KMeans(n_clusters=k).fit(data).inertia_
        inertia.append(c)
    plt.scatter(krange,inertia)
    plt.xlabel("k")
    plt.xticks(krange)
    plt.ylabel(" f' ")
    plt.show()

def plot_sil_k(data):
    """Scatter silhouette score for a range of k values to determine k in k-means
    (section 2.4.2).
    """
    krange = range(2,11)
    silhouettes = []
    for k in krange:
        c = skcl.KMeans(n_clusters=k).fit(data)
        silhouettes.append(silhouette_score(data,c.labels_))
    plt.scatter(krange,silhouettes)
    plt.xlabel("k")
    plt.xticks(krange)
    plt.ylabel(" Sil' ")
    plt.show()

def plot_persistence(data):
    """Scatter sum of cluster persistence (f_D) for a range of c_min values
    (section 4.2.4 / 4.3.2 / 5.4)
    """
    crange = [10*x for x in range(2,21,2)]
    stab = np.zeros(len(crange))
    hdbscanParam = {'min_cluster_size': 0, 'metric': 'cosine'}
    for i, cmin in enumerate(crange):
            hdbscanParam['min_cluster_size'] = cmin
            hdb = clustering.HDBSCAN(data,hdbscanParam)
            pers = sum(hdb.cluster_persistence_)
            stab[i] = pers
    plt.scatter(list(crange),stab)
    ax = plt.gca()
    ax.set_ylim([0, 0.5])
    plt.xticks(crange)
    plt.xlabel("c_min")
    plt.ylabel(" f_D ")
    plt.show()

def plot_percentageclustered(data):
    """Scatter percentage of points not clustered as noise for a range of c_min values
    (section 4.2.4)
    """
    crange = [10*x for x in range(2,21,2)]
    perc = np.zeros(len(crange))
    hdbscanParam = {'min_cluster_size': 0, 'metric': 'cosine'}
    for i, cmin in enumerate(crange):
            hdbscanParam['min_cluster_size'] = cmin
            hdb = clustering.HDBSCAN(data,hdbscanParam)
            clustered = [l >= 0 for l in hdb.labels_]
            percclustered = (100*sum(clustered)) / len(clustered) if clustered else 0
            perc[i] = percclustered
    plt.scatter(list(crange),perc)
    ax = plt.gca()
    ax.set_ylim([0, 100])
    plt.xticks(crange)
    plt.xlabel("c_min")
    plt.ylabel(f"% clustered")
    plt.show()

def plot_sil_n_neighbors(data,affinity):
    """Scatter silhouette score for a range of n_neighbors and k values,
    given some affinity applicable to spectral clustering 
    (see clustering.py, appendix C).
    """
    kValues = range(2,11)
    n_neighbors = range(10,110,5)
    labels = [f"k = {k}" for k in kValues]
    spectralParam = {'n_clusters': 0, 'affinity': affinity, 'n_neighbors': 0}
    silhouettes = np.zeros(shape = (len(kValues),len(n_neighbors)))
    for i,k in enumerate(kValues):
        spectralParam['n_clusters'] = k
        for j,n in enumerate(n_neighbors):
            spectralParam['affinity'] = affinity
            spectralParam['n_neighbors'] = n
            spec = clustering.Spectral(data,10,1e-05,spectralParam)
            sil = silhouette_score(data, spec.labels_, metric = affinity)
            silhouettes[i,j] = sil
    for line,lab in zip(silhouettes,labels):
        plt.scatter(list(n_neighbors), line,label = lab)
    ax = plt.gca()
    ax.set_ylim([0, 0.5])
    plt.xlabel("n_neighbors")
    plt.ylabel("mean silhouette score")
    plt.legend()
    plt.show()

def plot_eigengaps(data,affinity):
    """Plot the first 10 eigengaps found when performing spectral clustering
    given some affinity (section 5.3).
    """
    kmax = 10
    n_neighbors = range(20,110,20)
    spectralParam = {'n_clusters': 2, 'affinity' : affinity, 'n_neighbors': 0}
    eigengaps = np.zeros(shape = (len(n_neighbors), kmax-1))
    for i,n in enumerate(n_neighbors):
        spectralParam['affinity'] = affinity
        spectralParam['n_neighbors'] = n
        affin = clustering.Spectral(data, 10, 1e-05, spectralParam).affinity_matrix_
        laplacian = sparse.csgraph.laplacian(affin, normed = True)
        eigs = np.flip(sparse.linalg.eigsh(laplacian,kmax,which="SA", maxiter = 1000, return_eigenvectors = False))
        eigengaps[i] = np.diff(eigs)
    labels = [f"n_neighbors = {n}" for n in n_neighbors]
    lambdas = range(1,10)
    differ = [f"|λ_{i+1}-λ_{i}|" for i in range(1,10)]

    for line, lab in zip(eigengaps, labels):
        plt.scatter(lambdas,line,label = lab)
    ax = plt.gca()
    ax.set_ylim([0, 0.5])
    plt.xticks(lambdas, labels = differ, rotation='vertical')
    plt.ylabel("eigengap")
    plt.legend()
    plt.show()

