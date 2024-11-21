#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import pandas as pd

feature1 = ['L6', 'L11', 'L12', 'L13', 'L14', 'L16', 'L17', 'L19', 'L20', 'L22', 'L23', 'L24', 'L25', 'L26', 'L27', 'L30', 'L31']
feature2 = ['BCD2', 'BCD5', 'BCD11', 'BCD13', 'BCD18', 'BCD20', 'BCD21', 'BCD16', 'BCD24', 'BCD22']
feature3 = ['BCD2', 'BCD5', 'BCD11', 'BCD13', 'BCD18', 'BCD20', 'BCD21', 'BCD16', 'BCD24', 'BCD22']

data_combinations = list(itertools.product(feature1, feature2, feature3))

df = pd.DataFrame(data_combinations, columns=['Feature1', 'Feature2', 'Feature3'])

df.to_csv(r'Combinatorial_space.csv', index=False)

print("The dataset has been successfully written into Combinatorial_space.csv")






