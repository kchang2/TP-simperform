import pandas as pd  # (*) pandas for dataframe manipulation
import csv
import matplotlib.pyplot as plt # module for plotting 
import sklearn
import scipy
import numpy as np

from mpl_toolkits.mplot3d import Axes3D




dataFile = "/Users/kaichang/Documents/classes/ay119/final_project/baseball-team-similarity-master/results/2015 Philadelphia Phillies.csv"
df = pd.read_csv(dataFile, sep=',')

#df = df.drop(df.index[0])  # drop first row (US totals) 
#df = df[df['murder'] < 11] # drop out-of-range rows

fig = plt.figure()         # (!) set new mpl figure object
ax = fig.add_subplot(111, projection='3d')  # add axis

ax.set_xlabel('BABIP of offense')
ax.set_ylabel('wOBA of offense')
ax.set_zlabel('BABIP of pitchers')
plt.title('Similarity comparisons amongst teams')

scatter = ax.scatter(
    df['BABIP'], 
    df['wOBA'], 
    df['PBABIP'],
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim1.png')


ax.set_xlabel('BABIP of offense')
ax.set_ylabel('wOBA of offense')
ax.set_zlabel('FIP of pitchers')
plt.title('Similarity comparisons amongst teams')

scatter = ax.scatter(
    df['BABIP'], 
    df['wOBA'], 
    df['FIP'],
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim2.png')



ax.set_xlabel('BABIP of offense')
ax.set_ylabel('wOBA of offense')
ax.set_zlabel('effERA of pitchers')
plt.title('Similarity comparisons amongst teams')

scatter = ax.scatter(
    df['BABIP'], 
    df['wOBA'], 
    df['FIP'],
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim3.png')



ax.set_xlabel('BABIP of offense')
ax.set_ylabel('FIP of pitchers')
ax.set_zlabel('PBABIP of pitchers')
plt.title('Similarity comparisons amongst teams')

scatter = ax.scatter(
    df['BABIP'], 
    df['FIP'], 
    df['PBABIP'],
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim4.png')



ax.set_xlabel('wOBA of offense')
ax.set_ylabel('ERA of pitchers')
ax.set_zlabel('PBABIP of pitchers')
plt.title('Similarity comparisons amongst teams')

scatter = ax.scatter(
    df['wOBA'], 
    df['ERA'], 
    df['PBABIP'],
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim5.png')



ax.set_xlabel('FIP of pitchers')
ax.set_ylabel('ERA of pitchers')
ax.set_zlabel('PBABIP of pitchers')
plt.title('Similarity comparisons amongst teams')

scatter = ax.scatter(
    df['FIP'], 
    df['ERA'], 
    df['PBABIP'],
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

plt.show()
plt.savefig('Sim6.png')



fig = plt.figure()         # (!) set new mpl figure object

plt.xlabel('BABIP of offense')
plt.ylabel('wOBA of offense')
plt.title('Similarity offensive comparisons amongst teams')

scatter = plt.scatter(
    df['BABIP'], 
    df['wOBA'], 
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim7.png')


plt.xlabel('BABIP')
plt.ylabel('PBABIP')
plt.title('Similarity comparisons amongst teams')

scatter = plt.scatter(
    df['BABIP'], 
    df['PBABIP'], 
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim8.png')



plt.xlabel('BABIP')
plt.ylabel('FIP')
plt.title('Similarity comparisons amongst teams')

scatter = plt.scatter(
    df['BABIP'], 
    df['FIP'], 
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim9.png')


plt.xlabel('BABIP')
plt.ylabel('ERA')
plt.title('Similarity comparisons amongst teams')

scatter = plt.scatter(
    df['BABIP'], 
    df['ERA'], 
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim10.png')



plt.xlabel('wOBA')
plt.ylabel('PBABIP')
plt.title('Similarity comparisons amongst teams')

scatter = plt.scatter(
    df['BABIP'], 
    df['PBABIP'], 
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim11.png')


plt.xlabel('BABIP')
plt.ylabel('effERA')
plt.title('Similarity comparisons amongst teams')

scatter = plt.scatter(
    df['BABIP'], 
    df['effERA'], 
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim12.png')


plt.xlabel('wOBA')
plt.ylabel('effERA')
plt.title('Similarity comparisons amongst teams')

scatter = plt.scatter(
    df['wOBA'], 
    df['effERA'], 
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim13.png')


plt.xlabel('wOBA')
plt.ylabel('FIP')
plt.title('Similarity comparisons amongst teams')

scatter = plt.scatter(
    df['wOBA'], 
    df['FIP'], 
    # linewidths=2, 
    # edgecolor='w',
    # alpha=0.6
)

#plt.show()
plt.savefig('Sim14.png')