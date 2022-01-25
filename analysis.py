def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict
from math import ceil
from sklearn.metrics import silhouette_score


class ClusterInfo:
    """Computing relevant info given a clustering.

    Adjustable parameters
    --------
    threshold : int
        Any z-score higher than threshold (in absolute terms) is considered an `extreme deviation'.
        Used to compute the number of extreme deviations per individual (see "numExtremeDeviations").
    trim : int
        Determines how trimmed the trimmed mean of top deviations should be (see "meanTopDeviations").
        e.g. trim = 0.1 means taking a trimmed mean of 10% 
    N : int
        The top N deviations for every subject that will be returned in "topNDeviations".
    """
    def __init__(self,ASDData,ADOSscores,clusterLabels,predZScores,clinicalLabels,clustering,threshold,trim,N):

        self.ASDData = ASDData
        self.ADOSscores = ADOSscores
        self.clusterLabels = clusterLabels
        self.predZScores = predZScores
        self.clinicalLabels = clinicalLabels
        self.clustering = clustering
        self.threshold = threshold
        self.trim = trim
        self.N = N
        
        self.clusters = np.unique(self.clusterLabels)
        self.numExtDev = self.numExtremeDeviations()
        self.meanTopDev = self.meanTopDeviations()
        self.nTopDev = self.nTopDeviations()

    def getNames(self):
        """Returns a list with cluster names (i.e. `Cluster' plus the label it got from the algorithm).
        """
        return [f"Cluster {c}" for c in self.clusters]

    def getRes(self,resnames, to_string = True):
        """Returns datalist with specified results given the functions in resnames.

        Adjustable parameters
        --------
        resnames : [str]
            resnames contains the wanted results in the analysis for every cluster.
            All strings should be the name of a callable function in this class,
            which takes only the cluster label `c' as input and only returns the 
            value of the computed result (see for example "mean_iq(c)")
        to_string : bool
            Whether to convert the integer results to strings before returning.
            Useful for exporting to .xlsx files.
        """
        datalist = np.zeros(shape = (len(resnames), len(self.clusters)))
        for j, c in enumerate(self.clusters):
            for i,res in enumerate(resnames):
                value = eval(f"self.{res}({c})")
                datalist[i,j] = str(value) if (to_string) else value
        return datalist        

    def getStats(self,statnames, to_string = True):
        statlist = []
        for stat in statnames:
            value = eval(f"self.{stat}()")
            value = str(value) if (to_string) else value
            statlist.append(value)
        return statlist

    def clusternTopDeviations(self,c):
        clusterZScores = self.clusterZScores(c)
        clinicalLabels = self.clinicalLabels
        N = self.N

        devIndices = np.zeros(shape=(clusterZScores.shape[0], N))
        for subind in range(clusterZScores.shape[0]):
            abssub = np.array([abs(x) for x in clusterZScores[subind]])
            devIndices[subind] = self.indnLargest(abssub,N)
        
        devIndices = devIndices.astype(int)
        toplist = defaultdict(int)
        for topdev in devIndices:
            for top in topdev:
                toplist[clinicalLabels[top]] += 1
        toplist = dict(sorted(toplist.items(), key=lambda pair: pair[1], reverse = True)[:N])
        return [f"{key} : {toplist[key]}" for key in toplist]

    def nTopDeviations(self):
        return [self.clusternTopDeviations(c) for c in self.clusters]

    def nLargest(self,list,n):
        return list[self.indnLargest(list,n)]
    def indnLargest(self,list,n):
        return np.argpartition(list,-n)[-n:]

    def meanTopDeviations(self):
        ZScores = self.predZScores
        trim = self.trim
        dev = np.zeros(ZScores.shape[0])
        for subind in range(ZScores.shape[0]):
            sub = ZScores[subind]
            abssub = np.array([abs(x) for x in sub])
            toptrim = ceil(trim * len(sub))
            trimsub = self.nLargest(abssub,toptrim)
            dev[subind] = np.mean(trimsub)
        return dev

    def numExtremeDeviations(self):
        ZScores = self.predZScores
        threshold = self.threshold
        dev = np.zeros(ZScores.shape[0])
        for subind in range(ZScores.shape[0]):
            dev[subind] = sum([abs(x) >= threshold for x in ZScores[subind]])
        return dev

    def clusterZScores(self, c):
        return self.predZScores[self.clusterLabels == c]

    def clusterASDData(self, c):
        return self.ASDData[self.clusterLabels == c]

    def clusterADOSData(self, c):
        return self.ADOSscores[self.clusterLabels == c]

    def clusterADOSavailable(self, c):
        return self.ADOSscores[(self.clusterLabels == c) & np.isfinite(self.ADOSscores)]

    def clusternumExtDev(self, c):
        return self.numExtDev[(self.clusterLabels == c) & np.isfinite(self.ADOSscores)]

    def clustermeanExtDev(self,c):
        return self.meanTopDev[(self.clusterLabels == c) & np.isfinite(self.ADOSscores)]


    # Here follow the functions that can be called in getRes()

    def clustersizes(self,c):
        return len(self.clusterASDData(c))

    def mean_age(self, c):
        return np.mean(self.clusterASDData(c)[:,3],axis=0)
    
    def median_age(self,c):
        return np.median(self.clusterASDData(c)[:,3],axis=0)

    def sd_age(self, c):
        return np.std(self.clusterASDData(c)[:,3],axis=0)

    def mean_iq(self, c):
        return np.mean(self.clusterASDData(c)[:,11],axis=0)

    def sd_iq(self, c):
        return np.std(self.clusterASDData(c)[:,11],axis=0)

    def N_female(self, c):
        c_ASD = self.clusterASDData(c)
        return np.sum(c_ASD[c_ASD[:,4] == 2][:,4])
    def perc_female(self, c):
        c_ASD = self.clusterASDData(c)
        return len(c_ASD[c_ASD[:,4] == 2])/len(c_ASD[:,4])

    def perc_ADOS(self, c):
        c_ADOS = self.clusterADOSData(c)
        c_ADOSavailable = self.clusterADOSavailable(c)
        return len(c_ADOSavailable) / len(c_ADOS)

    def mean_ADOS(self, c):
        return np.mean(self.clusterADOSavailable(c),axis=0)
    def sd_ADOS(self,c):
        return np.std(self.clusterADOSavailable(c),axis =0)
    
    def mean_NSD(self, c):
        return np.mean(self.clusternumExtDev(c), axis = 0)
    
    def mean_MTD(self,c):
        return np.mean(self.clustermeanExtDev(c),axis = 0)

    def median_NSD(self,c):
        return np.median(self.clusternumExtDev(c),axis = 0)

    def median_MTD(self,c):
        return np.median(self.clustermeanExtDev(c),axis = 0)

    def corr_numextdev(self, c):
        return pearsonr(self.clusternumExtDev(c), self.clusterADOSavailable(c))[0]
    
    def p_numextdev(self, c):
        return pearsonr(self.clusternumExtDev(c), self.clusterADOSavailable(c))[1]

    def corr_meantopdev(self, c):
        return pearsonr(self.clustermeanExtDev(c), self.clusterADOSavailable(c))[0]

    def p_meantopdev(self, c):
        return pearsonr(self.clustermeanExtDev(c), self.clusterADOSavailable(c))[1]

    # Here follow the functions that get called in getStats()

    def silhouette(self):
        return silhouette_score(self.predZScores,self.clusterLabels, metric = "euclidean")

    def perc_clustered(self):
        clusteredpoints = [l >= 0 for l in self.clusterLabels]
        percClustered = (100*sum(clusteredpoints)) / len(clusteredpoints) if clusteredpoints else 0
        return percClustered

    def cluster_persistence(self):
        return self.clustering.cluster_persistence_
        




    