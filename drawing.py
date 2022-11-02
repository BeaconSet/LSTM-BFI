import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series


f = h5py.File('gtau.mat', 'r')
data = f.get('gtau')
data = np.array(data)  # For converting to a NumPy array
# print("data=", data)
t=np.arange(1, 502)

fig=plt.figure()
plt.plot(t[0:len(data[1,:,1])],data[157,:,0], label='5,20,1')
plt.plot(t[0:len(data[1,:,1])],data[157,:,1], label='5,20,2')
plt.plot(t[0:len(data[1,:,1])],data[157,:,2], label='5,20,3')
plt.plot(t[0:len(data[1,:,1])],data[158,:,0], label='20,5,1')
plt.plot(t[0:len(data[1,:,1])],data[158,:,1], label='20,5,2')
plt.plot(t[0:len(data[1,:,1])],data[158,:,2], label='20,5,3')

fig.legend()
plt.show()

init_data = pd.read_csv("./gtau2.3edited.csv")

