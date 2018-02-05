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

nb_points = 20

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

def MGD(point, mu, sig):
    q =  (1 / (2 *np.pi * np.sqrt(np.linalg.det(sig))))
    e = np.exp((-1/2)  * np.transpose(point - mu) @ np.invert(sig)  @ (point - mu))
    return q * e

def eStep(m1, m2, s1, s2, al1, al2):
    probs = np.zeros((2, nb_points))
    for i in range(len(data[0])):
        point = np.array([data[0][i], data[1][i]])
        probs[0][i] = MGD(point, m1, sig1)
        probs[1][i] = MGD(point, m2, sig2)
    weights = np.zeros((2, nb_points))
    for i in range(len(data[0])):
        tot =  (probs[0][i] * al1 + probs[1][i] * al2)
        weights[0][i] = (probs[0][i] * al1) / tot
        weights[1][i] = (probs[1][i] * al2) / tot
    return weights

def updateMu(data, weights,  nk, g):
    mu = 1 / nk
    tmp = np.array([data[0][0], data[1][0]]) * weights[g][0]
    for i in range(1, nb_points):
        tmp  += np.array([data[0][i], data[1][i]]) * weights[g][i]
    return mu * tmp

def updateSig(data, weights, nk, g, mu):
    sig = 1 / nk
    x = np.array([np.array([data[0][0], data[1][0]]) - mu])
    tmp = weights[g][0] * (x.T @ x)
    for i in range(1, nb_points):
        x = np.array([[data[0][i], data[1][i]] - mu])
        tmp += weights[g][i] * (x @ x.T)
    return tmp * sig

def EM(data):
    m1 =np.array([4, 4])
    m2 = np.array([0.5, 6])
    s1 = sig1
    s2 = sig2
    al1 = 0.5
    al2 = 0.5
    for i in range(5):
        weights = eStep(m1, m2, s1, s2, al1, al2)
        n1 = sum(weights[0])
        n2 = sum(weights[1])
        al1 = n1 / nb_points
        al2 = n2 / nb_points
        m1 = updateMu(data, weights, n1, 0)
        m2 = updateMu(data, weights, n2, 1)
        s1 = updateSig(data, weights, n1, 0,  m1)
        s2 = updateSig(data, weights, n2, 1,  m2)
    print("G1: ")
    print(m1)
    print(s1)
    print('\n' + "G2:")
    print(m2)
    print(s2)

data = genGMM(nb_points, mu1, mu2, sig1, sig2, w1, w2) #Will plot the 2 gaussians

plt.axis([-10, 10, -10, 10])
plt.show()
plotGauss(data, 'go')                                                           #Plot the resulting GMM
plt.axis([-10, 10, -10, 10])
plt.show()

EM(data)