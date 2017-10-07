# -*- coding: utf-8 -*
from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import random
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import csr_matrix, coo_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import time
import datetime
import scipy
import pylab
import scipy.cluster.hierarchy as sch
from numpy import array,zeros
from scipy import signal
import math



fp_path = "../data/fingerprints.csv"
des_path = "../data/descriptors.csv"
label_path = "../data/label.csv"



up_level = 36
low_level = 2
label_unit = ' hour'

which_des = 0  # choose a descriptor 

#load fingerprints and descriptors
fp = pd.read_csv(fp_path)
des = pd.read_csv(des_path)
labels = pd.read_csv(label_path)


fp_data = fp[fp.columns[1:]]
des_data = des[des.columns[1:]]

Chem_property = des_data.columns[which_des]  #$$$$$$$$$$$#


LogP = des_data[Chem_property].tolist()

idlogp = sorted(range(len(LogP)), key=lambda k: LogP[k])  # sort list and then return index of the sorted list


#load label and make y value became array
y_label = labels[labels.columns[1]]

#calculate the distence, for fingerpint: tanimoto distence; descriptor:euclidean distence
D_fp= pairwise_distances(fp_data.values, fp_data.values, metric='rogerstanimoto', n_jobs=-1)

# Compute and plot first fingerprints dendrogram.
fig = pylab.figure(figsize=(50,30))
ax1 = fig.add_axes([0.09,0.1,0.2,0.8]) #(left, bottom, width, height)
Y_fp = sch.linkage(D_fp, method='complete')
Z1 = sch.dendrogram(Y_fp, orientation='right')
idx1 = Z1['leaves']
ax1.set_xticks([])
ax1.set_yticks([])

# Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])

X = des_data[Chem_property].values[idx1] #chage sqence, only array and series can be done like this
Y = np.array(range(len(X))) # this Y is the sequnce of the tree
Z = np.array(y_label[idx1]) #chage sqence, only array and series can be done like this

X_Y_Z = pd.DataFrame({'X':X,'Y':Y,'Z':Z}) 

gt_24 = X_Y_Z['Z'] >= math.log(up_level,(10)) # greater than 24 hours
ls_4 = X_Y_Z[-gt_24]['Z'] <= math.log(low_level,(10)) # less than 4 hours
between_4_24h = X_Y_Z[-gt_24][-ls_4]


'''
scatter parameters see :http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
                       :http://matplotlib.org/gallery.html#lines_bars_and_markers
'''

im_24h = plt.scatter(X_Y_Z[gt_24]['X'],X_Y_Z[gt_24]['Y'], c = 'r', s = np.power(X_Y_Z[gt_24]['Z'].values,1.5)*120, marker='o',alpha=0.7) #, edgecolors = 'face' http://matplotlib.org/examples/lines_bars_and_markers/marker_reference.html 

im_4h = plt.scatter(X_Y_Z[-gt_24][ls_4]['X'], X_Y_Z[-gt_24][ls_4]['Y'], c = 'b', s = np.power(X_Y_Z[-gt_24][ls_4]['Z'].values,1.2)*200, marker='D',alpha=0.7)

im_4_24h =plt.scatter(between_4_24h['X'], between_4_24h['Y'], c = 'g', s = np.power(between_4_24h['Z'].values,2.5)*150, marker='D',alpha=0.5)



plt.ylim(0,len(LogP))
plt.xlim(X.min(), X.max())
plt.legend((im_24h,im_4_24h,im_4h),('>= '+ str(up_level) + label_unit, str(low_level) + '-'+ str(up_level) + label_unit, '<= ' + str(low_level) + label_unit),scatterpoints = 1, loc = 'upper right', ncol = 1, fontsize = 15, borderaxespad=0.) #http://matplotlib.org/users/legend_guide.html;  http://matplotlib.org/api/legend_api.html

axmatrix.set_xlabel(Chem_property, fontsize = 20, ha='left')
axmatrix.xaxis.set_label_position('bottom')
axmatrix.xaxis.tick_bottom()
axmatrix.yaxis.set_label_position('right')
axmatrix.yaxis.tick_right()

plt.show()

#fig.savefig('dendrogram.png',dpi = 600)
