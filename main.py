from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
  
# fetch dataset 
iris = fetch_ucirepo(id=53)
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

classes = np.unique(y)
print("Clases:", classes)

# encuentra valor en determiando index
print(X.iloc[51])
print(y.iloc[51])

# choose attribute
column_name = "petal length" # seleccion de columna a discretizar
min_value = X[column_name].min()
max_value = X[column_name].max()
print(f"Minimum value of column '{column_name}': {min_value}")
print(f"Maximum value of column '{column_name}': {max_value}")

# print(X.min())
# print(X.max())