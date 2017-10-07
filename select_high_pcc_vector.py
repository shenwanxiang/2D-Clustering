# -*- coding: utf-8 -*
from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd


fp_path = "../data/fingerprints_881.csv"
des_path = "../data/descriptors_40.csv"
label_path = "../data/label.csv"



n = 10 # selected desriptors number
m = 50 # selected fingerprints number


# load fingerprints and descriptors
fp = pd.read_csv(fp_path)
des = pd.read_csv(des_path)
labels = pd.read_csv(label_path)
ids = labels[labels.columns[0]].to_frame(name = 'Ids')


fp_data = fp[fp.columns[1:]]
des_data = des[des.columns[1:]]
Y = labels[labels.columns[1]].to_frame(name = 'Y') 

# get selected descriptors
y_des = pd.concat([Y,des_data],axis = 1)
des_pcc = y_des.corr()['Y'].order()
neg_s = des_pcc[: n]
pos_s = des_pcc[des_pcc.shape[0]-n-1:des_pcc.shape[0]-1]
selected_des = pd.concat([des_data[neg_s.index],des_data[pos_s.index]],axis =1)
Ids_selected_des = pd.concat([ids, selected_des],axis = 1)
Ids_selected_des.to_csv("../data/descriptors.csv", index = None)

# get selected fingerprints
y_fp = pd.concat([Y,fp_data],axis = 1)
fp_pcc = y_fp.corr()['Y'].order()
neg_s_fp = fp_pcc[: m]
pos_s_fp = fp_pcc[fp_pcc.shape[0]-m-1:fp_pcc.shape[0]-1]
selected_fp = pd.concat([fp_data[neg_s_fp.index],fp_data[pos_s_fp.index]],axis =1)
Ids_selected_fp = pd.concat([ids, selected_fp],axis = 1)
Ids_selected_fp.to_csv("../data/fingerprints.csv", index = None)

