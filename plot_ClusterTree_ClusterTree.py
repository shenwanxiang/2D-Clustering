# -*- coding: utf-8 -*
from __future__ import print_function
from __future__ import division
import math
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

#load fingerprints and descriptors
fp = pd.read_csv("../data/fingerprints.csv")
des = pd.read_csv("../data/descriptors.csv")
fp_data = fp[fp.columns[1:]]
des_data = des[des.columns[1:]]



#load label and make y value became array

y = pd.read_csv("../data/label.csv")
y_label = y[y.columns[1]] #value
#y_label = y[y.columns[2]] #class


def get_dis_matrix(N,y_label):
    distance_matrix = zeros((N, N))
    for i in xrange(N):
        for j in xrange(N):
            if j == i:
                distance_matrix[i, j] = y_label.values[i]
            else:
                distance_matrix[i, j] = y_label.values.min()-0.01
    return distance_matrix


DD_label = get_dis_matrix(y_label.shape[0],y_label)





#calculate the distence, for fingerpint: tanimoto distence; descriptor:euclidean distence
D_fp= pairwise_distances(fp_data.values, fp_data.values, metric='rogerstanimoto', n_jobs=-1)
D_des = pairwise_distances(des_data.values, des_data.values, metric='euclidean', n_jobs=-1)


# Compute and plot first fingerprints dendrogram.
fig = pylab.figure(figsize=(30,30))
ax1 = fig.add_axes([0.09,0.1,0.2,0.6]) #(left, bottom, width, height)
Y_fp = sch.linkage(D_fp, method='complete')
Z1 = sch.dendrogram(Y_fp, orientation='right')
ax1.set_xticks([])
ax1.set_yticks([])



# Compute and plot second descriptors dendrogram.
ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
Y_des = sch.linkage(D_des, method='complete')
Z2 = sch.dendrogram(Y_des)
ax2.set_xticks([])
ax2.set_yticks([])



# Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
D = DD_label[idx1,:]
D = D[:,idx2]




# define the cell to convolve

cell = np.array([[0.3, 0.6, 0.3],
                 [0.6, 0.9, 0.6],
                 [0.3, 0.6, 0.3]])

cell = np.array([[0.01, 0.03, 0.05, 0.07, 0.09, 0.07, 0.05, 0.03, 0.01],
                 [0.03, 0.05, 0.07, 0.09, 0.11, 0.09, 0.07, 0.05, 0.03],
                 [0.05, 0.07, 0.09, 0.11, 0.13, 0.11, 0.09, 0.07, 0.05],
                 [0.07, 0.09, 0.11, 0.13, 0.15, 0.13, 0.11, 0.09, 0.07],
                 [0.09, 0.11, 0.13, 0.15, 0.20, 0.15, 0.13, 0.11, 0.09],
                 [0.07, 0.09, 0.11, 0.13, 0.15, 0.13, 0.11, 0.09, 0.07],
                 [0.05, 0.07, 0.09, 0.11, 0.13, 0.11, 0.09, 0.07, 0.05],
                 [0.03, 0.05, 0.07, 0.09, 0.11, 0.09, 0.07, 0.05, 0.03],
                 [0.01, 0.03, 0.05, 0.07, 0.09, 0.07, 0.05, 0.03, 0.01]])

'''
    cell = np.array([[0.1, 0.3, 0.5, 0.3, 0.1],
                     [0.3, 0.5, 0.7, 0.5, 0.3],
                     [0.5, 0.7, 0.9, 0.7, 0.5],
                     [0.3, 0.5, 0.7, 0.5, 0.3],
                     [0.1, 0.3, 0.5, 0.3, 0.1]])

'''

# convolve Distence D
D_con = signal.convolve2d(D,cell,boundary='symm', mode='same')

D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')
D_con = signal.convolve2d(D_con,cell,boundary='symm', mode='same')

int_level = D_con.min()+0.01
step = (D_con.max()-D_con.min())/5
big_level = D_con.max()

#im = plt.contour(D_con, 20, levels=[int_level,int_level+step,int_level+step*2,int_level+step*3,int_level+step*4,big_level], cmap=plt.cm.jet) #vist http://matplotlib.org/examples/color/colormaps_reference.html to choose a camp
im = plt.contour(D_con, 20, cmap=plt.cm.jet)
#im = plt.contour(D_con, 15, linewidth=.5, alpha=.75)
#im = axmatrix.matshow(D_con, aspect='auto', origin='lower') #, cmap=pylab.cm.YlGnBu cm : colormap
#im = axmatrix.imshow(D_con, aspect='auto', origin='lower') #, cmap=pylab.cm.YlGnBu cm : colormap
axmatrix.set_xticks([])
axmatrix.set_yticks([])



# Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
pylab.colorbar(im, cax=axcolor)
fig.show()
#fig.savefig('dendrogram.png',dpi = 600)






