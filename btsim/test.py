import pandas as pd  # (*) pandas for dataframe manipulation
import csv
import matplotlib.pyplot as plt # module for plotting 
import sklearn
import scipy
import numpy as np
import random

from mpl_toolkits.mplot3d import Axes3D
 
# not needed
def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X
 
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

# not needed 
def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

data = []
dataFile = "/Users/kaichang/Documents/classes/ay119/final_project/baseball-team-similarity-master/results/2015 Philadelphia Phillies.csv"
f = open(dataFile)
csv_f = csv.reader(f)

for row in csv_f:
    try:
        data.append([float(row[5]), float(row[6])]) #BABIP, wOBA
    except ValueError as err:
        print(err.args)

data.pop(0) #removes tags or headers of column
data = np.asarray(data)


(mu, clusters) = find_centers(data, 5)

k1x, k1y = mu[0]
k2x, k2y = mu[1]
k3x, k3y = mu[2]
k4x, k4y = mu[3]
k5x, k5y = mu[4]

c1 = clusters[0]
c2 = clusters[1]
c3 = clusters[2]
c4 = clusters[3]
c5 = clusters[4]

np.savetxt("c1.csv",c1,delimiter=",")
np.savetxt("c2.csv",c2,delimiter=",")
np.savetxt("c3.csv",c3,delimiter=",")
np.savetxt("c4.csv",c4,delimiter=",")
np.savetxt("c5.csv",c5,delimiter=",")
a = np.array([[k1x,k1y],[k2x,k2y],[k3x,k3y],[k4x,k4y],[k5x,k5y]])
np.savetxt("kvalues.csv",a,delimiter=",")

b1 = []
b2 = []
b3 = []
b4 = []
b5 = []

w1 = []
w2 = []
w3 = []
w4 = []
w5 = []

for i in clusters[0]:
    b1.append(i[0])
    w1.append(i[1])

for i in clusters[1]:
    b2.append(i[0])
    w2.append(i[1])

for i in clusters[2]:
    b3.append(i[0])
    w3.append(i[1])

for i in clusters[3]:
    b4.append(i[0])
    w4.append(i[1])

for i in clusters[4]:
    b5.append(i[0])
    w5.append(i[1])


fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(b1,w1, c='b', marker='.', label = 'k1')
ax1.scatter(b2,w2, c='r', marker=',', label = 'k2')
ax1.scatter(b3,w3, c='g', marker='o', label = 'k3')
ax1.scatter(b4,w4, c='y', marker='v', label = 'k4')
ax1.scatter(b5,w5, c='c', marker='^', label = 'k5')

ax1.scatter(k1x,k1y, c='b', marker='*', s=100)
ax1.scatter(k2x,k2y, c='r', marker='*', s=100)
ax1.scatter(k3x,k3y, c='g', marker='*', s=100)
ax1.scatter(k4x,k4y, c='y', marker='*', s=100)
ax1.scatter(k5x,k5y, c='c', marker='*', s=100)

plt.legend(loc='upper left')
#plt.show()
plt.savefig('KCluster1.png')





(mu, clusters) = find_centers(data, 4)

k1x, k1y = mu[0]
k2x, k2y = mu[1]
k3x, k3y = mu[2]
k4x, k4y = mu[3]

c1 = clusters[0]
c2 = clusters[1]
c3 = clusters[2]
c4 = clusters[3]

b1 = []
b2 = []
b3 = []
b4 = []

w1 = []
w2 = []
w3 = []
w4 = []

for i in clusters[0]:
    b1.append(i[0])
    w1.append(i[1])

for i in clusters[1]:
    b2.append(i[0])
    w2.append(i[1])

for i in clusters[2]:
    b3.append(i[0])
    w3.append(i[1])

for i in clusters[3]:
    b4.append(i[0])
    w4.append(i[1])


fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(b1,w1, c='b', marker='.', label = 'k1')
ax1.scatter(b2,w2, c='r', marker=',', label = 'k2')
ax1.scatter(b3,w3, c='g', marker='o', label = 'k3')
ax1.scatter(b4,w4, c='y', marker='v', label = 'k4')

ax1.scatter(k1x,k1y, c='b', marker='*', s=100)
ax1.scatter(k2x,k2y, c='r', marker='*', s=100)
ax1.scatter(k3x,k3y, c='g', marker='*', s=100)
ax1.scatter(k4x,k4y, c='y', marker='*', s=100)

plt.legend(loc='upper left')
#plt.show()
plt.savefig('KCluster2.png')





(mu, clusters) = find_centers(data, 3)

k1x, k1y = mu[0]
k2x, k2y = mu[1]
k3x, k3y = mu[2]

c1 = clusters[0]
c2 = clusters[1]
c3 = clusters[2]

b1 = []
b2 = []
b3 = []

w1 = []
w2 = []
w3 = []

for i in clusters[0]:
    b1.append(i[0])
    w1.append(i[1])

for i in clusters[1]:
    b2.append(i[0])
    w2.append(i[1])

for i in clusters[2]:
    b3.append(i[0])
    w3.append(i[1])


fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(b1,w1, c='b', marker='.', label = 'k1')
ax1.scatter(b2,w2, c='r', marker=',', label = 'k2')
ax1.scatter(b3,w3, c='g', marker='o', label = 'k3')

ax1.scatter(k1x,k1y, c='b', marker='*', s=100)
ax1.scatter(k2x,k2y, c='r', marker='*', s=100)
ax1.scatter(k3x,k3y, c='g', marker='*', s=100)

plt.legend(loc='upper left')
#plt.show()
plt.savefig('KCluster3.png')






(mu, clusters) = find_centers(data, 6)

k1x, k1y = mu[0]
k2x, k2y = mu[1]
k3x, k3y = mu[2]
k4x, k4y = mu[3]
k5x, k5y = mu[4]
k6x, k6y = mu[5]

c1 = clusters[0]
c2 = clusters[1]
c3 = clusters[2]
c4 = clusters[3]
c5 = clusters[4]
c6 = clusters[5]

b1 = []
b2 = []
b3 = []
b4 = []
b5 = []
b6 = []

w1 = []
w2 = []
w3 = []
w4 = []
w5 = []
w6 = []

for i in clusters[0]:
    b1.append(i[0])
    w1.append(i[1])

for i in clusters[1]:
    b2.append(i[0])
    w2.append(i[1])

for i in clusters[2]:
    b3.append(i[0])
    w3.append(i[1])

for i in clusters[3]:
    b4.append(i[0])
    w4.append(i[1])

for i in clusters[4]:
    b5.append(i[0])
    w5.append(i[1])

for i in clusters[5]:
    b6.append(i[0])
    w6.append(i[1])

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(b1,w1, c='b', marker='.', label = 'k1')
ax1.scatter(b2,w2, c='r', marker=',', label = 'k2')
ax1.scatter(b3,w3, c='g', marker='o', label = 'k3')
ax1.scatter(b4,w4, c='y', marker='v', label = 'k4')
ax1.scatter(b5,w5, c='c', marker='^', label = 'k5')
ax1.scatter(b6,w6, c='m', marker='^', label = 'k6')

ax1.scatter(k1x,k1y, c='b', marker='*', s=100)
ax1.scatter(k2x,k2y, c='r', marker='*', s=100)
ax1.scatter(k3x,k3y, c='g', marker='*', s=100)
ax1.scatter(k4x,k4y, c='y', marker='*', s=100)
ax1.scatter(k5x,k5y, c='c', marker='*', s=100)
ax1.scatter(k6x,k6y, c='m', marker='^', s=100)

plt.legend(loc='upper left')
#plt.show()
plt.savefig('KCluster4.png')
