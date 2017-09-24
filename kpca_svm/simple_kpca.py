import numpy as np
import matplotlib.pyplot as plt

m = 50
n = 200
X = np.random.randn(2, m)/10

for idx in range(n):
    t = np.random.randn(2, 2)
    tmp = t[:, 0:1]/np.linalg.norm(t[:, 0]) + t[:, 1]/np.linalg.norm(t[:, 1:2])/10
    X = np.hstack((X, tmp))
    
plt.figure()
plt.scatter(X[0, m+1:m+n], X[1, m+1:m+n])
plt.scatter(X[0, 1:m], X[1, 1:m], c='r')

plt.show()

#%% kPCA
def custom_sdist(X):
    """
	Function that given a matrix X returns the squared pairwise distances 
	of the column vectors in matrix form
	"""
    XX = np.dot(X.T, X)
    pdists = np.outer(np.diag(XX), np.ones(XX.shape[1]).T) + np.outer(np.ones(XX.shape[0]), np.diag(XX).T) - 2*XX
    return pdists

def K_gram(X):
    # Gaussian kernel works best
    sigma = 0.15
    X_dist = custom_sdist(X)
    K = np.zeros(X.shape)
    K = np.exp(- X_dist / (2*sigma**2))
    return K

k = 2
K = K_gram(X)
#centering
H = np.eye(X.shape[1]) - np.ones((X.shape[1],X.shape[1]))/X.shape[1]
K_centered = H.dot(K).dot(H)
w,V = np.linalg.eig(K_centered)
Vt = V.T
w = 1./np.sqrt(w)
Sigma_inv = np.diag(w)
S = Sigma_inv[:k,:k].dot(Vt[:k,:]).dot(H).dot(K)


#%% visualize
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#plt.scatter(Y[0, :], Y[1,:])
plt.scatter(S[0, m+1:m+n],np.zeros(199), c='b')
plt.scatter(S[0, 1:m],np.zeros(49), c='r')
plt.show()