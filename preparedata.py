import numpy as np
import pandas as pd
import itertools

#todo: add basic cohort info (after stripping)



class StructuredData:
    """Extract arrays usable in Python from .xlsx data

    Fixed parameters
    ---------
    stripped : bool
        Whether to perform exclusion as described in experiment section. Set to True.
    PATH: str
        Path name of ENIGMA data set (excl. ADOS scores).
    PATHADOS: str
        Path name of data set including column with available ADOS scores.

    Adjustable parameters
    ---------
    covariateNames : list[str]
        List of covariates to include. Names indicate literal column names in ENIGMA set.
        Possible options are:
            * 'Age'
            * 'Sex (1=male, 2=female)'
            * 'IQ'
            * 'Site set'
            * 'Medication current  (2=yes, 1=no)'
            * 'Site name'
    includeICV : bool
        Whether to include total intracranial volumes as a clinical measure.
    includeTotalSurf : bool
        Wheter to include total surface area as a clinical measure.
    """


    def __init__(self, PATH, PATHADOS, stripped, covariateNames,includeICV, includeTotalSurf):
        self.fullDataFrame = pd.read_excel(PATH) #pandas dataframe extracted from excel
        self.ADOSDataFrame = pd.read_excel(PATHADOS, sheet_name = 'Covariats_allsites')
        
        self.allLabels = list(self.fullDataFrame) #used labels
        self.covariateLabels = covariateNames
        self.covariates = [self.allLabels.index(covariate) for covariate in self.covariateLabels]

        self.clinicalLB = 15 if (includeICV) else 16 
        self.clinicalUB = len(self.allLabels) if (includeTotalSurf) else -1
        self.clinicalLabels = self.allLabels[self.clinicalLB:self.clinicalUB]

        self.allData = np.array(self.fullDataFrame) #np array, only data values, no labels
        
        self.totalNumOfSubjects = self.allData.shape[0]
        self.useData = self.restrictSubjects(self.allData,self.completeList) if (stripped) else self.allData
        self.usedNumOfSubjects = self.useData.shape[0]

        self.covariateData = self.restrictCovariates(self.useData)
        self.clinicalData = self.restrictClinical(self.useData)

        self.ASDData = self.restrictSubjects(self.useData,self.ASDList)
        self.covariateASDData = self.restrictCovariates(self.ASDData)
        self.clinicalASDData = self.restrictClinical(self.ASDData)

        self.ADOSarray = np.array(self.ADOSDataFrame)
        self.ADOSscores = self.getADOS(self.ADOSarray)

        self.NTData = self.restrictSubjects(self.useData,self.NTList)
        self.covariateNTData = self.restrictCovariates(self.NTData)
        self.clinicalNTData = self.restrictClinical(self.NTData)

        self.NPMdata = self.covariateNTData,self.clinicalNTData,self.covariateASDData,self.clinicalASDData

    def restrictSubjects(self, Data, funcList):
        return Data[funcList(Data),:]
    def restrictCovariates(self, Data):
        return Data[:,self.covariates]
    def restrictClinical(self, Data):
        return Data[:,self.clinicalLB:self.clinicalUB]


    def completeList(self, allData): #list of indices with complete data
        completeList = []
        for i in range(allData.shape[0]):
            complete = True
            for j in self.covariates:
                if self.isNull(i,j): complete = False
            for j in range(15,len(allData[i])):
                if self.isNull(i,j): complete = False
                elif allData[i,j] == 0: complete = False
            if complete: completeList.append(i)
        return completeList
    def ASDList(self, Data): #list of indices from certain cohort
        return [i for i in range(Data.shape[0]) if Data[i,2] == 3]
    def NTList(self, Data): #list of indices from certain cohort
        return [i for i in range(Data.shape[0]) if Data[i,2] == 0]

    def getADOS(self,ADOSarray):
        ASDsubj = self.ASDData[:,0]
        ADOSsubj = np.array([ADOSarray[i,0] for i in range(ADOSarray.shape[0])])
        indicesOfASDsubj = np.searchsorted(ADOSsubj,ASDsubj)
        ADOSlist = np.array([ADOSarray[i,9] for i in indicesOfASDsubj])
        return ADOSlist

    def ADOSinADOSData(self, ADOSData):
        return [(ADOSData[i,0],ADOSData[i,9]) for i in range(ADOSData.shape[0])]
    
    def isNull(self,i,j):
        allDataij = self.allData[i,j]
        return (allDataij == "NA" or allDataij != allDataij)

    def printDemographics(self):
        adosavail = self.ADOSscores[np.isfinite(self.ADOSscores)]
        func = ['mean', 'std']
        ind = [3,11]
        sets = ['self.ASDData', 'self.NTData']
        for f,i,s in itertools.product(func,ind,sets):
            val = eval(f"np.{f}({s}[:,{i}], axis = 0)")
            print(f"{f},{i},{s}: {val}")
        print(f"ados,mean {np.mean(adosavail,axis=0)}")
        print(f"ados,sd {np.std(adosavail,axis=0)}")
        sexASD = self.ASDData[:,4]
        fema = sexASD[sexASD == 2]
        sexNT = self.NTData[:,4]
        femn = sexNT[sexNT == 2]
        print (f"female N (asd) = {fema.shape[0]}, %  = {fema.shape[0]/sexASD.shape[0]}")
        print (f"female N (nt) = {femn.shape[0]}, % = {femn.shape[0]/sexNT.shape[0]}")
        print(self.ASDData.shape[0])
        print(self.NTData.shape[0])






    

