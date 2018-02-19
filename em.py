import math
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

colors = ["r", "g", "b", "peachpuff", "fuchsia"]
size_color = len(colors)

def plot_data(nb_gauss, data):
    for j in range (nb_gauss):
        plot(data[j], colors[j % size_color])

def sdp_mat(dim, sd):
    M = np.matrix(np.random.rand(dim, dim))
    M = 0.5 * (M + M.T)
    M = M + sd * np.eye(dim)
    return M

"""
Generate a symmetric definite positive matrice
"""
def get_sdp_mat(sd_diff, nb_gauss, dim):
    SDClass = np.random.rand(1, nb_gauss) + sd_diff

    return [sdp_mat(dim, i) for i in SDClass[0]]

def gen_gmm(theta, nb_pts, nb_gauss, mu, cov):

    r = np.random.multinomial(nb_pts, theta)

    data = [np.random.multivariate_normal(mu[i], cov[i], r[i]) for i in range(0, nb_gauss)]
    return data

def update_mu(s_weight, data_array, weights_array):
    mu = np.zeros([dim])

    for i in range (len(data_array)):
        for d in range (dim):
            mu[d] += data_array[i][d] * weights_array[i]

    return mu / s_weight

def var_flor(s, f):
    l = np.linalg.cholesky(s)

    l_inv = np.linalg.inv(l)
    t = l_inv * s * l_inv.T

    u, d = np.linalg.eig(t)

    for i in range (len(d)):
        d[i][i] = max(1, d[i][i])

    t_ = u * d * u.T

    return l * t_ * l.T

def update_cov(s_weight, data_array, mu_array, weights_array):
    cov = np.zeros([dim, dim])

    for i in range (len(data_array)):
#        print ("update cov ")
#        print (cov)

        mat = np.asmatrix(data_array[i]) - np.asmatrix(mu_array)

        cov += weights_array[i] * (mat.T @ mat)

    return cov / s_weight

def log_multivar(point, dim, mu, cov):
    q =  -(dim / 2) * np.log (2 * math.pi) - 0.5 * np.log(np.linalg.det(cov))
    e = -0.5 * (point - mu).T @ np.linalg.inv(cov) @ (point - mu)
    return q + e

def compute_weight(nb_pts, nb_gauss, dim, alpha, data, mu, cov):
    weights = np.zeros([nb_gauss, nb_pts])

    llk = 0

    for j in range (nb_gauss):
        maxi = -math.inf
        for i in range (len(data[j])):
            point = np.empty([dim])
            for d in range (dim):
                point[d] = data[j][i][d]

            weights[j][i] = log_multivar(point, dim, mu[j], cov[j]) + np.log(alpha[j])

            if maxi < weights[j][i]:
                maxi = weights[j][i]

        for i in range (len(data[j])):
            tot = 0
            for j in range (nb_gauss):
                weights[j][i] -= maxi
                weights[j][i] = np.exp(weights[j][i])

                tot += weights[j][i]

            for j in range (nb_gauss):
                weights[j][i] = weights[j][i] / tot

            llk += maxi + np.log(tot)

    return (weights, llk)

if __name__ == '__main__':
    nb_pts = 3000
    nb_gauss = 2
    dim = 2

    distanceBTWclasses = 10

    alpha = np.repeat(1.0 / nb_gauss, nb_gauss)
    # The mean matrix
    mu = [(np.random.random(dim) * distanceBTWclasses * i) for i in range(1, nb_gauss + 1)]

    cov = get_sdp_mat(4, nb_gauss, dim)
    data = gen_gmm(alpha, nb_pts, nb_gauss, mu, cov)

    print ("MU")
    print (mu)
#    print ("data")
#    print (data)
    print ("cov")
    print (cov)

    print ("alpha")
    print (alpha)

    print ("START")

    prev = -math.inf

    cov = get_sdp_mat(4, nb_gauss, dim)
    mu = [(np.random.random(dim) * distanceBTWclasses * i) for i in range(1, nb_gauss + 1)]

    print ("New MU")
    print (mu)
    print ("New cov")
    print (cov)

    for i in range (10):

        (weights, llk) = compute_weight(nb_pts, nb_gauss, dim, alpha, data, mu, cov)
#        if prev >= llk:
#           print (llk)
#       if llk - prev < 0.1:
#            print ("No progression")
#           break

        prev = llk

        for j in range (nb_gauss):
            s_weight = np.sum(weights[j])

            alpha[j] = s_weight / len(data[j])
            mu[j] = update_mu(s_weight, data[j], weights[j])
            cov[j] = update_cov(s_weight, data[j], mu[j], weights[j])

#  print ("s weight")
#           print (s_weight)
    print ("alpha")
    print (alpha)

    print ("mu")
    print (mu)

    print ("cov")
    print (cov)

