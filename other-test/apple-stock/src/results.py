import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from reservoirpy.observables import nrmse, rsquare, mse
from sklearn.metrics import r2_score

""" df = pd.read_csv('recursive-results/test1/predictions-train-80-test1.csv')
dataset = df.values
training_data_len = math.ceil(len(dataset) * .8)
day = math.ceil(len(dataset) * .2) - 1

predictions = df['Predictions'][training_data_len:training_data_len + day]
print(predictions)
y_test = df['Close'][training_data_len:training_data_len + day]
print(nrmse(y_test, predictions))
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Datetime', fontsize=18)
plt.ylabel('Price')
plt.plot(df['Close'])
plt.plot(df['Predictions'])
plt.legend(['Train', 'Predictions'], loc='lower right')
plt.show()
 """

df_predictions = pd.read_csv('energy-consumption/test2/energy-reservoir-year.csv')
#df_predictions = pd.read_csv('ridge-train-80.csv')
dataset = df_predictions.values
training_data_len = math.ceil(len(dataset) * .8)
day = math.ceil(len(dataset) * .2) - 1
predictions = df_predictions['Predictions'][training_data_len:training_data_len + day]
y_test = df_predictions['DAYTON_MW'][training_data_len:training_data_len + day]
print(nrmse(y_test, predictions))
print(r2_score(y_test, predictions))
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Datetime', fontsize=18)
plt.ylabel('Price')
plt.plot(y_test)
plt.plot(predictions)
plt.legend(['Train', 'Predictions'], loc='lower right')
plt.show()