import pickle
import numpy as np
import sklearn.datasets as sets

moons, _ = sets.make_moons(n_samples=50, noise=0.01)
moons2, _ = sets.make_moons(n_samples=50, noise=0.03)
blobs, _ = sets.make_blobs(n_samples=50, centers=[(-0.75,2.25), 
(1.25, 2.0)], cluster_std=0.5)
blobs2, _ = sets.make_blobs(n_samples=50, centers=[(-0.75,2.25), 
(1.25, 2.0)], cluster_std=0.5)
data = np.vstack([moons,blobs, moons2, blobs2])

pickle.dump(data,open("toydata.p", "wb"))