import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.cm as cm

df = pd.read_csv('hyperparameter-lag3-training-20-percents.csv')
min = df['mse'].min()
max = df['mse'].max()
print(df.loc[df['mse'] == min])
print(df.loc[df['mse'] == max])
#print(df)
df = df.loc[df['leak_rate'] == 0.1]
points = plt.scatter(df['units'], df['spectral_radius'], c=df['r_square'],cmap="jet", lw=0, s=500)
plt.colorbar(points)
plt.show()
""" xi, yi = np.linspace(df['units'].min(), df['units'].max(), 100), np.linspace(df['leak_rate'].min(), df['leak_rate'].max(), 100)
xi, yi = np.meshgrid(xi, yi)

rbf = scipy.interpolate.Rbf(df['units'], df['leak_rate'], df['mse'], function='linear')
zi = rbf(xi, yi)

plt.imshow(zi, vmin=df['mse'].min(), vmax=df['mse'].max(), origin='lower', extent=[df['units'].min(), df['units'].max(), df['leak_rate'].min(), df['leak_rate'].max()])
plt.scatter(df['units'], df['leak_rate'], c=df['mse'])
plt.colorbar()
plt.show() """
""" 
nrows, ncols = 100, 100
grid = df['mse'].reshape((nrows, ncols))

plt.imshow(grid, extent=(df['units'].min(), df['units'].max(), df['leak_rate'].max(), df['leak_rate'].min()),
           interpolation='nearest', cmap=cm.gist_rainbow)
plt.show() """

#df.plot.hexbin(x="units", y="leak_rate", C="units", reduce_C_function=np.max, gridsize=25)
#plt.show()
""" x = df['units'].values
y = df['leak_rate'].values
z = df['mse'].values
X,Y = np.meshgrid(x,y)
Z = z.T
plt.contourf(X,Y,Z,20,cmap='jet')
plt.colorbar()
plt.show() """

""" xlist = df['units'].values
ylist = df['leak_rate'].values
X, Y = np.meshgrid(xlist, ylist)
Z = df['mse'].values
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
#ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.show() """
