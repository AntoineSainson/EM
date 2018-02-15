import numpy as np
import matplotlib.pyplot as plt

# SIGMAS (Matrice de covariance)

nb_points = 5000

"""
Distribution normale multidimensionnelle
"""
def normal_multi(d, mu, sig, x):
    dens = 1 / (np.power(2 * np.pi, d / 2) * np.sqrt(np.linalg.det(sig)))
    expos = -1 / 2 * np.transpose(x - mu) * np.linalg.inv(sig) * (x - mu)
    dens = dens * (np.power(np.exp, expos))

    return dens

def genGauss(N, mu, sig):
    data = np.random.randn(2, N)
    A = np.linalg.cholesky(sig)
    data = A @ data
    for i in range(len(data[0])):
        data[0][i]+= mu[0]
        data[1][i] += mu[1]
    return data

def plotGauss(data, par):
    plt.plot(data[0], data[1], par)

"""
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
"""
