def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sklearn.neighbors
import sklearn.cluster as skcl
import sklearn.metrics
import hdbscan

"""Implementation of Spectral clustering and HDBSCAN.

Fixed parameters in both algorithms
--------
data : ndarray
    The data to cluster.
"""


def Spectral(data,n_init,gamma,spectralparam):
    """Performs Spectral clustering.

    Fixed parameters
    --------
    n_init : int
        Number of times the final k-means algorihm restarts with new centroids.
    gamma :  int
        Optimization parameter for RBF kernel.

    Adjustable parameters
    --------
    affinity : string
        Metric used to construct the (nearest_neighbor) affinity matrix W.
        Options are:
            * all kernels defined by sklearn.metrics.pairwise.distance_metrics
                (such as ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan'])
    n_clusters : int
        Number of clusters k to find.
    """
    pwmetric = spectralparam["affinity"]
    pw_distance = sklearn.metrics.pairwise_distances(data, metric = pwmetric)
    spectralparam["affinity"] = "precomputed_nearest_neighbors"
    spec = skcl.SpectralClustering(n_init=n_init,gamma = gamma,**spectralparam).fit(pw_distance)
    spectralparam["affinity"] = pwmetric
    return spec


def HDBSCAN(data,hdbscanParam):
    """Performs HDBSCAN clustering.

    Fixed parameters
    --------

    Adjustable parameters
    --------
    min_cluster_size : int
        Minimum size of a cluster before split up or considered noise. See chapter on HDBSCAN (
        where we call it c_min)
    metric : str
        Metric used for measuring distance between points, hence estimating local density.
        Options are:
            * 'cosine': cosine similarity.
            * all kernels defined by hdbscan.dist_metrics.METRIC_MAPPING

    """
    metric = hdbscanParam['metric']
    if metric in ['cosine']:
        pwmetric = hdbscanParam['metric']
        pw_distance = sklearn.metrics.pairwise_distances(data, metric = pwmetric)
        hdbscanParam['metric'] = 'precomputed'
        hdbc = hdbscan.HDBSCAN(**hdbscanParam).fit(pw_distance)
    else:
        hdbc = hdbscan.HDBSCAN(**hdbscanParam).fit(data)
    return hdbc