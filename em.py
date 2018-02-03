import numpy as np
import matplotlib.pyplot as plt

# SIGMAS

sig1 = np.array([[2, 1], [1, 3]])
sig2 = np.array([[6, 1], [1, 7]])

#w

w1 = 0.3
w2 = 0.7

#mu

mu1 = np.array([1, 1])
mu2 = np.array([5, 5])

nb_points = 50

def genGauss(N, mu, sig):
    data = np.random.randn(2, N)
    A = np.linalg.cholesky(sig)
    data = A @ data
    k = 0
    for i in data:
        for j in i:
            j += mu[k]
        k += 1
    return data

def plotGauss(data, par):
    plt.plot(data[0], data[1], par)


def genGMM(N, mu1, mu2, sig1, sig2, w1, w2):
    f_data = genGauss(nb_points, mu1, sig1)
    s_data = genGauss(nb_points,mu2, sig2 )
    plotGauss(f_data, 'ro')
    plotGauss(s_data, 'bo')
    t = np.random.rand(N)
    data = np.copy(f_data)
    for i in range(N):
        if t[i] > w1:
            data[0][i] = s_data[0][i]
            data[1][i] = s_data[1][i]
    return data

data = genGMM(nb_points, mu1, mu2, sig1, sig2, w1, w2) #Will plot the 2 gaussians
plt.axis([-10, 10, -10, 10])
plt.show()
plotGauss(data, 'go')                                                           #Plot the resulting GMM
plt.axis([-10, 10, -10, 10])
plt.show()