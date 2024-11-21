#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import load
best_RF_model = load('best_RFR_model.joblib')
best_SVR_model = load('best_SVR_model.joblib')
best_XGB_model = load('best_XGB_model.joblib')


# In[2]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = r'Combinatorial_space.csv'
data = pd.read_csv(file_path, encoding='gbk')

data['Original_Index'] = data.index
data.replace({
    'L26': 9.1, 'L27': 7.6, 'L30': 2.9, 'L31': 1.2,
    'L6': 2.6, 'L20': 12.9, 'L22': 8.7, 'L23': 7.8,
    'L24': 4.8, 'L25': 1.3, 'L16': 16.5, 'L17': 15.5,
    'L19': 1.7, 'L11': 33.6, 'L12': 15.8, 'L13': 2.2,
    'L14': 25.5,
    'BCD2': 3132, 'BCD5': 2209, 'BCD11': 1855,
    'BCD13': 933, 'BCD18': 984, 'BCD20': 612,
    'BCD21': 410, 'BCD16': 289, 'BCD24': 211,
    'BCD22': 105
}, inplace=True)

Combinatorial_space = data.iloc[:, 1:4].values
min_val = 0.01 
max_val = 1.0
scaler = MinMaxScaler(feature_range=(min_val, max_val))
Combinatorial_space_scaled = scaler.fit_transform(Combinatorial_space)


# In[3]:


XGB_predicted_NMN = best_XGB_model.predict(Combinatorial_space_scaled)
data['NMN_production'] = XGB_predicted_NMN 

top_combinations_XGB = data.nlargest(17, 'NMN_production')[['Strain'] + list(data.columns[1:4]) + ['NMN_production']]
print('Top 1% combinations predicted by XGB')
print(top_combinations_XGB)

output_file_path = r'TableS4.csv'
top_combinations_XGB.to_csv(output_file_path, index=False)


# In[4]:


RF_predicted_NMN = best_RF_model.predict(Combinatorial_space_scaled)
data['NMN_production'] = RF_predicted_NMN 

top_combinations_RF= data.nlargest(17, 'NMN_production')[['Strain'] + list(data.columns[1:4]) + ['NMN_production']]
print('Top 1% combinations predicted by RF')
print(top_combinations_RF)

output_file_path = r'TableS3.csv'
top_combinations_RF.to_csv(output_file_path, index=False)


# In[5]:


SVR_predicted_NMN = best_SVR_model.predict(Combinatorial_space_scaled)
data['NMN_production'] = SVR_predicted_NMN 

top_combinations_SVR = data.nlargest(17, 'NMN_production')[['Strain'] + list(data.columns[1:4]) + ['NMN_production']]
print('Top 1% combinations predicted by SVR')
print(top_combinations_SVR)

output_file_path = r'TableS5.csv'
top_combinations_SVR.to_csv(output_file_path, index=False)


# In[6]:


file_path = r'FileS2.csv'
data = pd.read_csv(file_path, encoding='gbk')
data.dropna(axis=1, how='all', inplace=True)
data.dropna(axis=0, how='any', inplace=True)
data.shape
X=data.iloc[:,1:4].values
y=data.iloc[:,-1].values


# In[7]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

scatter_ = ax.scatter(X[:, 0], X[:, 1], X[:, 2],alpha=0.5, color='gray', s=80)
scatter_RF = ax.scatter(
   top_combinations_RF.iloc[:, 1], top_combinations_RF.iloc[:, 2], top_combinations_RF.iloc[:, 3], 
    color='red', s=200, marker='*',alpha=1
)
scatter_XGB = ax.scatter(
    top_combinations_XGB.iloc[:, 1], top_combinations_XGB.iloc[:, 2], top_combinations_XGB.iloc[:, 3], 
    color='red', s=200, marker='*',alpha=1
)

scatter_SVR = ax.scatter(
    top_combinations_SVR.iloc[:, 1], top_combinations_SVR.iloc[:, 2], top_combinations_SVR.iloc[:, 3], 
    color='red', s=200, marker='*',alpha=1
)

ax.set_xlabel('EsaI',fontsize=12)
ax.set_ylabel('GDH1',fontsize=12)
ax.set_zlabel('GDH2',fontsize=12)

ax.set_xlim([0, 35])  
ax.set_ylim([0, 3300])  
ax.set_zlim([0, 3300]) 

ax.view_init(elev=30, azim=-50)
plt.tick_params(axis='both', labelsize=10)
plt.rcParams['font.family'] = 'Arial'
plt.savefig('Distribution.tiff', format='tiff', transparent=True, dpi=100)
plt.title("Distribution of predicted top 1% combinations in combinatorial space")
plt.show()


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

xgb_strains = set(top_combinations_XGB['Strain'])
rf_strains = set(top_combinations_RF['Strain'])
svr_strains = set(top_combinations_SVR['Strain'])

plt.figure(figsize=(6, 6))
venn3([xgb_strains, rf_strains, svr_strains], ('XGB', 'RF', 'SVR'))
plt.title("Coverage of combinations between the three models")
plt.show()



