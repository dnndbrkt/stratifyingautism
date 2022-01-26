import prepareData
import NPM
import clustering
import analysis
import pandas as pd
from datetime import datetime
import plots

if __name__ == "__main__":
    """Running export.py as a main script exports results to excel sheet.
    The relevant parameters to tweak are contained in dictionaries
    ending in 'Param'. Explanation of the parameters can be found in the 
    relevant module (e.g. dataParam --> prepareData.py).
    """

    # importing ENIGMA data (clinical + covariates), plus extra covariate file including ADOS scores.

    PATH = "./DATA/ENIGMA-ASD.xlsx"
    PATHADOS = "./DATA/Covariates_allsites.xlsx"

    # creating usable dataframe from .xlsx files (see 'prepareData.py')

    stripped = True

    covariateNames = ['Age','Sex (1=male, 2=female)','IQ', 'Site set']
    includeICV = False
    includeTotalSurf = False

    dataParam = {'covariateNames': covariateNames, 
                'includeICV': includeICV,
                'includeTotalSurf': includeTotalSurf}

    d = prepareData.StructuredData(PATH, PATHADOS, stripped, **dataParam)
    d.printDemographics()

    # creating a normative probability map (NPM) using gaussian process regression (see 'npm.py')

    Xtrain = d.covariateNTData # matrix X from D_NT
    YtrainN = d.clinicalNTData # matrix Y from D_NT
    Xtest = d.covariateASDData # matrix X' from D_ASD
    realYtest = d.clinicalASDData # matrix Y' from D_ASD
    centralize_data = True
    n_restarts_optimizer = 5

    kernel = "rbf"  
    npmParam = {'kernel': kernel}

    npm = NPM.NPM(Xtrain, YtrainN, Xtest, realYtest, centralize_data, n_restarts_optimizer, **npmParam)

    clusterData = npm.predZScores # dataset Z_ASD mentioned in experiment section

    # performing spectral clustering on the determined Z-scores (see 'clustering.py')
    n_init = 10
    gamma = 1e-5

    n_clusters = 2
    affinity = 'cosine'
    n_neighbors = 40

    spectralParam = {'n_clusters': n_clusters, 'affinity': affinity, 'n_neighbors': n_neighbors}

    spec = clustering.Spectral(clusterData,n_init,gamma,spectralParam)

    # performing HDBSCAN on the determined Z-scores (see 'clustering.py')

    min_cluster_size = 80
    metric = 'cosine'

    hdbscanParam = {'min_cluster_size': min_cluster_size, 'metric': metric}

    hdbc = clustering.HDBSCAN(clusterData,hdbscanParam)

    # determine correlations between (extreme) Z-score deviations and ADOSscores and get statistics per cluster

    threshold = 1.96
    trim = 0.01
    N = 15

    corrParam = {'threshold': threshold,'trim': trim, 'N': N}

    specClusInfo = analysis.ClusterInfo(d.ASDData,d.ADOSscores,spec.labels_,npm.predZScores,d.clinicalLabels,spec,**corrParam)
    hdbClusInfo = analysis.ClusterInfo(d.ASDData,d.ADOSscores,hdbc.labels_,npm.predZScores,d.clinicalLabels,hdbc,**corrParam)

    resnames = ['clustersizes', 'mean_age','median_age', 
                'sd_age', 'mean_iq','sd_iq','mean_ADOS',
                'sd_ADOS','N_female','perc_female',
                'perc_ADOS','mean_NSD','mean_MTD',
                'median_NSD',
                'median_MTD','corr_numextdev','p_numextdev',
                'corr_meantopdev','p_meantopdev'
                ]

    specRes, specNames = specClusInfo.getRes(resnames), specClusInfo.getNames()
    hdbRes, hdbNames = hdbClusInfo.getRes(resnames), hdbClusInfo.getNames()

    specstatnames = ['silhouette']
    hdbstatnames = ['perc_clustered', 'cluster_persistence']
    allstatnames = [statname for statnames in [specstatnames,hdbstatnames] for statname in statnames]

    specstats = specClusInfo.getStats(specstatnames)
    hdbstats = hdbClusInfo.getStats(hdbstatnames)

    # TOP deviations per cluster (see analysis.py)

    specntopdev = specClusInfo.nTopDev
    hdbntopdev = hdbClusInfo.nTopDev

    # Creating dataframes suitable for Excel export

    specDF = pd.DataFrame(specRes,index = resnames,columns=specNames)
    hdbDF = pd.DataFrame(hdbRes,index = resnames,columns=hdbNames)

    spectopDF = pd.DataFrame(specntopdev).transpose()
    hdbtopDF = pd.DataFrame(hdbntopdev).transpose()
    spectopDF.columns, hdbtopDF.columns = specNames,hdbNames

    specStatlist = [["spectral", stat] for stat in specstats]
    hdbStatlist = [["hdbscan", stat] for stat in hdbstats]
    allStatlist =  [pair for statlist in [specStatlist,hdbStatlist] for pair in statlist]
    statDF = pd.DataFrame(allStatlist, index = allstatnames, columns = ['Algorithm', 'Value'])


    keyParams = [key for param in [dataParam,npmParam,spectralParam,hdbscanParam] for key in param]
    dataParamlist = [["dataParam",dataParam[key]] for key in dataParam]
    npmParamlist = [["npmParam",npmParam[key]] for key in npmParam]
    spectralParamlist = [["spectralParam",spectralParam[key]] for key in spectralParam]
    hdbscanParamlist = [["hdbscanParam",hdbscanParam[key]] for key in hdbscanParam]
    allParamlist = [pair for Paramlist in [dataParamlist,npmParamlist,spectralParamlist,hdbscanParamlist] for pair in Paramlist] #all parameters used
    paramDF = pd.DataFrame(allParamlist, index = keyParams, columns=['Paramlist', 'Value'])


    # Exporting everything
    
    dfs = {"algo parameters": paramDF,"cluster statistics": statDF, "spectral": specDF, "hdbscan": hdbDF, "topdevspec": spectopDF,"topdevhdb":hdbtopDF}

    OUTPUT =  f"Result {datetime.now()}.xlsx"

    writer = pd.ExcelWriter(OUTPUT, engine='xlsxwriter')
    for sheetname, df in dfs.items(): 
        df.to_excel(writer, sheet_name=sheetname)  
        worksheet = writer.sheets[sheetname] 
    writer.save()








