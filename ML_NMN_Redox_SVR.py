#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mglearn
import numpy as np
import pandas as pd


# In[2]:


file_path = r'filename2.csv'
data = pd.read_csv(file_path, encoding='gbk')
X=data.iloc[:,1:4].values
y=data.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
min_val = 0.01 
max_val = 1.0
scaler = MinMaxScaler(feature_range=(min_val, max_val))
X_scaled = scaler.fit_transform(X)
pd.DataFrame(X_scaled)


# In[3]:


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=35)

def SVR_optimize(C, gamma, epsilon):
    model = SVR(C=C, 
                gamma=gamma, 
                epsilon=epsilon
)
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    return scores.mean()


# In[4]:


param_bounds = {
    'C': (0.1, 10),         
    'gamma': (1e-4, 1),    
    'epsilon': (1e-4, 1) 
}


# In[5]:


optimizer = BayesianOptimization(
    f=SVR_optimize,            
    pbounds=param_bounds,      
    random_state=35, 
    verbose=2              
)

optimizer.maximize(init_points=5, n_iter=20)
print("Best parameters:", optimizer.max['params'])
print(f"Best R² score: {optimizer.max['target']:.2f}")


# In[6]:


best_params = optimizer.max['params']
best_SVR_model = SVR(C=best_params['C'], gamma=best_params['gamma'], epsilon=best_params['epsilon'])
best_SVR_model.fit(X_train, y_train)

y_test_pred = best_SVR_model.predict(X_test)
y_train_pred = best_SVR_model.predict(X_train)

test_r2 = r2_score(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
print(f"Test R² score: {test_r2:.2f}")
print(f"Train R² score: {train_r2:.2f}")


# In[7]:


import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
plt.scatter(y_train, y_train_pred, color='lightcoral', alpha=0.6,label=f'training set R²= {train_r2:.2f}')
plt.scatter(y_test, y_test_pred, color='royalblue', alpha=0.6, label=f'testing set R²= {test_r2:.2f}')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--',  lw=2)
plt.xlabel('Measured NMN production', fontsize=18)
plt.ylabel('Predicted NMN production', fontsize=18)
plt.title('SVR model', fontsize=18)
plt.legend()
plt.tick_params(axis='both', labelsize=18)
plt.rcParams['font.family'] = 'Arial'
plt.legend(fontsize=14, loc='upper left',handlelength=0.2,frameon=0)
plt.savefig('SVR_model.tiff', format='tiff', dpi=100)
plt.show()


# In[8]:


from joblib import dump
dump(best_SVR_model, 'best_SVR_model.joblib')
