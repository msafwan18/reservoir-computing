import pandas as pd
from sklearn.metrics import r2_score
from reservoirpy.observables import nrmse
import matplotlib.pyplot as plt
import openpyxl

df_predictions = pd.read_csv('10.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (units = 10) = " + str(nrmse(y_test, predictions)))
print("R-squared (units = 10) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('20.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (units = 20) = " + str(nrmse(y_test, predictions)))
print("R-squared (units = 20) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('30.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (units = 30) = " + str(nrmse(y_test, predictions)))
print("R-squared (units = 30) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('40.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (units = 40) = " + str(nrmse(y_test, predictions)))
print("R-squared (units = 40) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('50.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (units = 50) = " + str(nrmse(y_test, predictions)))
print("R-squared (units = 50) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('60.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (units = 60) = " + str(nrmse(y_test, predictions)))
print("R-squared (units = 60) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('70.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (units = 70) = " + str(nrmse(y_test, predictions)))
print("R-squared (units = 70) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('80.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (units = 80) = " + str(nrmse(y_test, predictions)))
print("R-squared (units = 80) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('90.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (units = 90) = " + str(nrmse(y_test, predictions)))
print("R-squared (units = 90) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('100.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (units = 100) = " + str(nrmse(y_test, predictions)))
print("R-squared (units = 100) = " + str(r2_score(y_test, predictions)))

""" df = pd.read_excel('units.xlsx')
print(df)
df.index = df['units']
#df = df.drop(40)
#df.plot(x ='units', y='nrmse', kind='line')
plt.title("Number of neurons with NRMSE")
plt.xticks(df['units'])
plt.plot(df["nrmse"], '.-')
plt.xlabel("Number of neurons")
plt.ylabel("NRMSE")
plt.show()

plt.title("Number of neurons with R-squared")
plt.xticks(df['units'])
plt.plot(df["r2"], '.-')
plt.xlabel("Number of neurons")
plt.ylabel("R-squared")
plt.show() """

df = pd.read_excel('units.xlsx')
print(df)
df.index = df['units']
#df = df.drop(40)
fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('Leak rate with performance')

#df.plot(x ='units', y='nrmse', kind='line')
ax1.set_title("Units with NRMSE")
ax1.plot(df["nrmse"], '.-')
#ax1.set_xlabel("Leak rate")
ax1.set_ylabel("NRMSE")

ax2.set_title("Units with R-squared")
ax2.plot(df["r2"], '.-')
ax2.set_xlabel("Units")
ax2.set_ylabel("R-squared")
plt.show()