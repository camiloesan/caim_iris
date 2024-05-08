from ucimlrepo import fetch_ucirepo 
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

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
    midpoints = (ordered_unique_values[1:] + ordered_unique_values[:-1]) / 2
    B = np.concatenate(([min_value_in_col], midpoints, [max_value_in_col]))
    print(f"Variable B '{column_name}':", B)
    
    # 1.3 inicializacion del esquema de discretizacion y caim global
    D = [min_value_in_col, max_value_in_col]
    global_caim = 0
    print()
    
    # 2
    # 2.1 inicializacion k = 1
    k = 1
    
    # 2.2 adicion de UN limite interno y calcular quanta matrix (inicia loop?)
    while True:
        print(f"Loop {k}")
        print("D values:", D)
        random_index = random.randint(1, len(B) - 2)
        # lim = B[random_index] # limite interno
        lims = np.unique([D[0], B[7], B[14], D[-1]])
        D = lims
        # calcular quanta matrix
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
        
        # 2.3 aceptar limite(s)? con mayor valor caim *pendiente*
        # obtener parametros de qmatrix y calcular valor caim
        maxr = 0
        mpr = 0
        N = length - 1
        for i in range(N):
            maxr = max(q1[i], q2[i], q3[i])
            mpr = sum([q1[i], q2[i], q3[i]])
            print(f"maxr: {maxr}, mpr: {mpr}")
        caim_value = (maxr/mpr) / N
        print(f"CAIM value: {caim_value}")
        
        if caim_value > global_caim or k < 3:
            # D = lims # *pendiente si se busca el valor*
            global_caim = caim_value
        else:
            print("CAIM value is not greater than global CAIM value")
            return D
        
        # 2.5 aumentar k
        k = k + 1
        

# print column names
print("Column names:", X.columns)
# choose attribute
column_name = "petal length" # seleccion de columna a discretizar?

print()
discretization_vector = caim(column_name)
print("Discretization:", discretization_vector)

data = np.unique(X[column_name])
# Plot the values of the array
plt.plot(range(len(data)), data, 'o-', color='blue')

ranges = [(discretization_vector[i], discretization_vector[i + 1]) for i in range(len(discretization_vector) - 1)]
print("Ranges:", ranges)
# Draw vertical lines representing ranges
for range_start, range_end in ranges:
    plt.hlines(range_start, xmin=0, xmax=50, color='red', linestyles='dashed')
    plt.hlines(range_end, xmin=0, xmax=50, color='red', linestyles='dashed')
# Set labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Data Plot')
# Show plot
plt.grid(True)
plt.show()