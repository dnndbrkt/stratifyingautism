from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
import numpy as np

#todo: make hardcoded covariates etc. variable.

class NPM:
    """Constructing a normative probability map (NPM) using Gaussian Process Regression (GPR).
    
    Fixed parameters
    ---------
    centralize_data : bool
        Whether to centralize Ytrain to have zero mean. The ENIGMA data set is not centralized,
        whence it is fixed to True.
    n_restarts_optimizer : int
        Optimization parameter required by sklearn. Fixed (conservatively) at 5.

    Adjustable parameters
    ---------
    kernel : string
        Kernel function to use when constructing NPM. See experiment section for explanation.
        Options are:
            * 'rbf': Radial Basis Function
            * 'dot': DotProduct
            * 'rbfdotadd': sum of RBF and DotProduct kernel
            * 'rbfdotmult': multiplication of RBF and DotProduct
    
    """
    def __init__(self, 
                Xtrain, YtrainN, Xtest, realYtest,
                centralize_data, n_restarts_optimizer,kernel):

        self.Xtrain, self.Xtest = Xtrain, Xtest
        self.YtrainN = YtrainN #NT brain volumes, uncentralized

        if (centralize_data):
            self.Ytrain, self.YtrainMean = self.centralize(self.YtrainN)
        else:
            self.Ytrain, self.YtrainMean = self.YtrainN, np.mean(self.YtrainN,axis = 0)
        if (kernel == 'rbf'): self.kernel = RBF(length_scale=1*np.ones(Xtrain.shape[1]), length_scale_bounds=(1e-23, 1e10))
        if (kernel == 'dot'): self.kernel = DotProduct()
        if (kernel == 'rbfdotadd'):
            self.kernel = RBF(length_scale=1*np.ones(Xtrain.shape[1]), length_scale_bounds=(1e-23, 1e10)) + DotProduct()
        if (kernel == 'rbfdotmult'):
            self.kernel = RBF(length_scale=1*np.ones(Xtrain.shape[1]), length_scale_bounds=(1e-23, 1e10))*DotProduct()
            
        self.realYtest = realYtest #actual ASD brainvolumes

        self.gpr = GaussianProcessRegressor(kernel = self.kernel,
                                            alpha=1e-8,
                                            random_state=0,
                                            n_restarts_optimizer=n_restarts_optimizer).fit(self.Xtrain,self.Ytrain)

        self.predYtestMean,self.predYtestSTD = self.gpr.predict(self.Xtest,return_std = True)

        self.predZScores = self.zScores(self.YtrainMean,self.realYtest,self.predYtestMean,self.predYtestSTD)
        self.ABSpredZScores = np.absolute(self.predZScores)

    def centralize(self, data, axis = 0):
        mean = np.mean(data, axis = axis)
        return data - mean, mean

    def zScores(self, YtrainMean, realYtest, predYtestMean, predYtestSTD):
        nominator =  realYtest - ( predYtestMean + YtrainMean )
        realvar = np.var(realYtest,axis=0)

        sigij = predYtestSTD.reshape(predYtestSTD.shape[0],1)
        signj = realvar.reshape(1,realvar.shape[0])
        denominator = (sigij**2 + signj)**(1/2)
        return nominator/denominator







        


