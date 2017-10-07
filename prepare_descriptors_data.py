import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler




all_data = pd.read_csv("../data/descriptors_tobedone.csv")

new_data = all_data[all_data.columns[1:]]

new_data = new_data.fillna(new_data.mean())

scaler = MinMaxScaler()

all_data_scaled = pd.DataFrame(scaler.fit_transform(new_data), columns=new_data.columns)




final = pd.concat([all_data['IDs'].to_frame(name = 'Ids'), all_data_scaled], axis=1)

final.to_csv('../data/descriptors.csv', index = None)


