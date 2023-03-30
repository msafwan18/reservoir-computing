import pandas as pd
from sklearn.metrics import r2_score
from reservoirpy.observables import nrmse
import matplotlib.pyplot as plt
import openpyxl
import matplotlib

df_predictions = pd.read_csv('energy-consumption/hyperparameters/leak_rate/0.1.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (leak_rate = 0.1) = " + str(nrmse(y_test, predictions)))
print("R-squared (leak_rate = 0.1) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('energy-consumption/hyperparameters/leak_rate/0.2.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (leak_rate = 0.2) = " + str(nrmse(y_test, predictions)))
print("R-squared (leak_rate = 0.2) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('energy-consumption/hyperparameters/leak_rate/0.3.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (leak_rate = 0.3) = " + str(nrmse(y_test, predictions)))
print("R-squared (leak_rate = 0.3) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('energy-consumption/hyperparameters/leak_rate/0.4.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (leak_rate = 0.4) = " + str(nrmse(y_test, predictions)))
print("R-squared (leak_rate = 0.4) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('energy-consumption/hyperparameters/leak_rate/0.5.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (leak_rate = 0.5) = " + str(nrmse(y_test, predictions)))
print("R-squared (leak_rate = 0.5) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('energy-consumption/hyperparameters/leak_rate/0.6.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (leak_rate = 0.6) = " + str(nrmse(y_test, predictions)))
print("R-squared (leak_rate = 0.6) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('energy-consumption/hyperparameters/leak_rate/0.7.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (leak_rate = 0.7) = " + str(nrmse(y_test, predictions)))
print("R-squared (leak_rate = 0.7) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('energy-consumption/hyperparameters/leak_rate/0.8.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (leak_rate = 0.8) = " + str(nrmse(y_test, predictions)))
print("R-squared (leak_rate = 0.8) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('energy-consumption/hyperparameters/leak_rate/0.9.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (leak_rate = 0.9) = " + str(nrmse(y_test, predictions)))
print("R-squared (leak_rate = 0.9) = " + str(r2_score(y_test, predictions)))

df_predictions = pd.read_csv('energy-consumption/hyperparameters/leak_rate/1.0.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][950 : 1900]
y_test = df_predictions['DAYTON_MW'][950 : 1900]
print("NRMSE (leak_rate = 1.0) = " + str(nrmse(y_test, predictions)))
print("R-squared (leak_rate = 1.0) = " + str(r2_score(y_test, predictions)))

df = pd.read_excel('energy-consumption/hyperparameters/leak_rate/leak_rate.xlsx')
print(df)
df.index = df['leak_rate']
fig, (ax1, ax2) = plt.subplots(2)
ax1.set_title("Leak rate with NRMSE")
ax1.plot(df["nrmse"], '.-')
ax1.set_ylabel("NRMSE")

ax2.set_title("Leak rate with R-squared")
ax2.plot(df["r2"], '.-')
ax2.set_xlabel("Leak rate")
ax2.set_ylabel("R-squared")
plt.show()