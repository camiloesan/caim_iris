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

def caim(column_name):
    # 1.1 min y max de la columna
    min_value_in_col = X[column_name].min()
    max_value_in_col = X[column_name].max()
    
    # 1.2 min, max y puntos medios
    ordered_unique_values = np.unique(X[column_name])
    print(f"Unique values in '{column_name}':", ordered_unique_values)
    B = (ordered_unique_values[1:] + ordered_unique_values[:-1]) / 2
    B = np.concatenate(([min_value_in_col], B, [max_value_in_col]))
    print(f"Variable B '{column_name}':", B)
    
    # 1.3 inicializacion del esquema de discretizacion y caim global
    D = [min_value_in_col, max_value_in_col]
    global_caim = 0
    print()
    
    # 2
    # 2.1 inicializacion k = 1
    k = 1
    
    # 2.2 adicion de limite interno y calcular quanta matrix (inicia loop)
    lims = [D[0] ,B[6], B[12], D[1]]
    # calcular valor caim (quanta matrix)
    length = len(lims)
    q1 = np.array([])
    q2 = np.array([])
    q3 = np.array([])
    for index, value in enumerate(lims):
        if index == length - 1:
            break

        v1 = ((X[column_name].iloc[0:50] >= value) & (X[column_name].iloc[0:50] <= lims[index+1])).sum()
        v2 = ((X[column_name].iloc[50:100] >= value) & (X[column_name].iloc[50:100] <= lims[index+1])).sum()
        v3 = ((X[column_name].iloc[100:150] >= value) & (X[column_name].iloc[100:150] <= lims[index+1])).sum()
        q1 = np.append(q1, v1)
        q2 = np.append(q2, v2)
        q3 = np.append(q3, v3)

    print("q1 values:", q1)
    print("q2 values:", q2)
    print("q3 values:", q3)
    
    # 2.3 aceptar limite con mayor valor caim *pendiente*
    maxr = 0
    mpr = 0
    N = length - 1
    for i in range(N):
        maxr = max(q1[i], q2[i], q3[i])
        mpr = sum([q1[i], q2[i], q3[i]])
        print(f"maxr: {maxr}, mpr: {mpr}")
        
    # calcular valor caim? o solo parametros para...
    caim_value = (1/N) * (maxr/mpr)
    print(f"CAIM value: {caim_value}")
    
    if caim_value > global_caim | k < 3:
        global_caim = caim_value
    else:
        print("CAIM value is not greater than global CAIM value")
        
    # 2.5 aumentar k
    k = k + 1
    
    return D
        
# choose attribute
column_name = "petal length" # seleccion de columna a discretizar?

print()
caim(column_name)