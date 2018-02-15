import gen
import numpy as np
import matplotlib.pyplot as plt

llk_thresh = 0.0001

def genGMM(N, mus, sigs, ws):
    l = len(ws)
    gaussians = np.array([[([None] * gen.nb_points)] * 2] * l)
    for i in range(l):
        g = gen.genGauss(gen.nb_points, mus[i], sigs[i])
        gaussians[i] = g
    gen.plotGauss(gaussians[0], 'ro')
    gen.plotGauss(gaussians[1], 'go')
    gen.plotGauss(gaussians[2], 'yo')
    gen.plotGauss(gaussians[3], 'bo')
    t = np.random.rand(N)
    data = np.copy(gaussians[2])
    for i in range(N):
        tmp = 0
        for j in range(l):
            tmp += ws [j]
            if t[i] < tmp:
                data[0][i] = gaussians[j][0][i]
                data[1][i] = gaussians[j][1][i]
                break
    return data

def LMGD(point, mu, sig, w):
    q = np.log(2 * np.pi) - .5 * np.log(np.linalg.det(sig))
    e = ((- 1 / 2) * (point - mu).T @ np.linalg.inv(sig) @ (point - mu))
    res = q + e + np.log((w))
    return res

def eStep(mus, sigs, ws, old_llk):
    nb_gaussians = len(ws)
    probs = np.zeros((nb_gaussians, gen.nb_points))
    weights = np.zeros((nb_gaussians, gen.nb_points))
    for i in range(gen.nb_points):
        point = np.array([data[0][i], data[1][i]])
        for j in range(nb_gaussians):
            probs[j][i] = LMGD(point, mus[j], sigs[j], ws[j])

    llk = 0
    for i in range(gen.nb_points):
        m = probs[:, i].max()
        tot = 0
        for j in range(nb_gaussians):
            probs[j][i] -= m
            tot += np.exp(probs[j][i])
        for j in range(nb_gaussians):
            weights[j][i] = (np.exp(probs[j][i]) / tot)
        llk += m + np.log(sum(np.exp(weights[:, 0])))
    llk /= gen.nb_points
    print("llk:")
    print(llk)
    return (weights, llk, old_llk - llk > -llk_thresh and old_llk <= llk)

def updateMus(mus, ns, weights):
    for i in range(len(ns)):
        tmp = np.array([data[0][0], data[1][0]]) * weights[i][0]
        for j in range(1, gen.nb_points):
            tmp  += np.array([data[0][j], data[1][j]]) * weights[i][j]
        mus[i] = (1 / ns[i]) * tmp
    return mus

def updateSigs(sigs, mus, ns, weights):
    for i in range(len(ns)):
        x = np.array([np.array([data[0][0], data[1][0]]) - mus[i]])
        tmp = weights[i][0] * (x.T @ x)
        for j in range(1, gen.nb_points):
            x = np.array([np.array([data[0][j], data[1][j]]) - mus[i]])
            tmp += weights[i][j] * (x.T @ x)
        sigs[i] = (1 / ns[i]) * tmp
    sigs = varianceFlooring(sigs)
    return sigs

def varianceFlooring(sigs):
    Ng = sigs[0]
    for i in range(1, len(sigs)):
        Ng = np.add(Ng, sigs[i])

    Ng = Ng * 0.2
    L = np.linalg.cholesky(Ng)
    Li = np.linalg.inv(L)

    for i in range(len(sigs)):
        T = Li @ sigs[i] @ Li.T
        D, U = np.linalg.eig(T)
        D = np.diag(D)
        for j in range(len(D[1])):
            D[j][j] = max(D[j][j], 1)
        Tb = U @ D @ U.T
        sigs[i] = L @ Tb @ L.T
    return sigs

def plotMu(mus):
    plt.plot(ms[0][0], ms[0][1], 'ro')
    plt.plot(ms[1][0], ms[1][1], 'bo')
    plt.plot(ms[2][0], ms[2][1], 'yo')
    plt.plot(ms[3][0], ms[3][1], 'co')
    plt.plot(mus[0][0], mus[0][1], 'go')
    plt.plot(mus[1][0], mus[1][1], 'go')
    plt.plot(mus[2][0], mus[2][1], 'go')
    plt.plot(mus[3][0], mus[3][1], 'go')
    plt.axis([-20, 20, -20, 20])
    plt.show()

def EM(data, mus, sigs, ws):
    plotMu(mus)
    llk = -10000
    while(True):
        (weights, llk, cond) = eStep(mus, sigs, ws, llk)
        if (cond):
            break
        ns = np.zeros((len(ws)))
        for i in range(len(ns)):
            ns[i] = max(sum(weights[i]), 0.1)
            ws[i] = ns[i] / gen.nb_points
        mus = updateMus(mus, ns, weights)
        sigs = updateSigs(sigs, mus, ns, weights)
        print("ws:")
        print(ws)
        plotMu(mus)
    plotMu(mus)

md = np.array([[1, 0], [10, 10], [-1, 15], [0, -10]])
sd = np.array([[[6, 1], [1, 7]], [[2, 1], [1, 3]],[[6, 1], [1, 7]],[[2, 1], [1, 3]]])
wd = [0.25, 0.25, 0.25, 0.25]

ms =np.array([[-15, 1], [11, -8], [1, 11], [18, 19]])
ss = np.array([[[2, 1],[1, 3]], [[6, 1],[1, 7]], [[2, 1],[1, 3]], [[6, 1],[1, 7]]])
wss = [0.2, 0.3, 0.4, 0.1]
data = genGMM(gen.nb_points, ms, ss, wss)
plt.axis([-20, 20, -20, 20])
plt.show()
gen.plotGauss(data, 'go')                                                           #Plot the resulting GMM
plt.axis([-20, 20, -20, 20])
plt.show()

EM(data, md, sd, wd)
