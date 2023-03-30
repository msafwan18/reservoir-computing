import pandas as pd
import numpy as np
import math
from reservoirpy.nodes import Reservoir, Ridge
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.observables import mse
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import time

units = 30
leak_rate = 0.5
spectral_radius = 0.5
connectivity = 0.5
input_connectivity = 0.5
regularization = 1e-5
seed = 1234

reservoir = Reservoir(units, 
                    sr=spectral_radius,
                    lr=leak_rate, 
                    rc_connectivity=connectivity,
                    input_connectivity=input_connectivity, 
                    seed=seed)

readout = Ridge(ridge=regularization)
reservoir <<= readout
model = reservoir >> readout 

scaler = MinMaxScaler(feature_range=(0,1))

lag = 210

def recursive(day, train_data, data):
    if day == 0:

        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(dataset)
        predictions = scaler.inverse_transform(train_data)
        df = pd.DataFrame(predictions, columns = ['Predictions'])
        data.insert(1, "Predictions", predictions, True)
        data.to_csv('energy-reservoir-year.csv')
        #print(df)
        return df

    else:
        x_train = []
        y_train = []

        for i in range(lag, len(train_data)):
            x_train.append(train_data[i-lag:i, 0])
            y_train.append([train_data[i, 0]])

        x_train, y_train = np.array(x_train), np.array(y_train)
        #print(np.shape(x_train))
        #print(np.shape(y_train))
        #exit()
        model.fit(x_train, y_train)
        predictions = model.run(x_train[-1])
        new_train_data = np.append(train_data, predictions)
        new_train_data = new_train_data.reshape(len(new_train_data), 1)
        recursive(day - 1, new_train_data, data)   # Recursive call

def ridge_recursive(day, train_data, data):
    if day == 0:

        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(dataset)
        predictions = scaler.inverse_transform(train_data)
        df = pd.DataFrame(predictions, columns = ['Predictions'])
        data.insert(1, "Predictions", predictions, True)
        data.to_csv('energy-ridge-year.csv')
        #print(df)
        return df

    else:
        x_train = []
        y_train = []

        for i in range(lag, len(train_data)):
            x_train.append(train_data[i-lag:i, 0])
            y_train.append([train_data[i, 0]])

        x_train, y_train = np.array(x_train), np.array(y_train)
        ridge = Ridge(ridge=1e-5)
        ridge = ridge.fit(x_train, y_train)
        predictions = ridge.run(x_train[-1])
        new_train_data = np.append(train_data, predictions)
        new_train_data = new_train_data.reshape(len(new_train_data), 1)
        ridge_recursive(day - 1, new_train_data, data)   # Recursive call

def dt_recursive(day, train_data, data):
    if day == 0:

        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(dataset)
        predictions = scaler.inverse_transform(train_data)
        df = pd.DataFrame(predictions, columns = ['Predictions'])
        data.insert(1, "Predictions", predictions, True)
        data.to_csv('energy-dt-year.csv')
        #print(df)
        return df

    else:
        x_train = []
        y_train = []

        for i in range(lag, len(train_data)):
            x_train.append(train_data[i-lag:i, 0])
            y_train.append([train_data[i, 0]])

        x_train, y_train = np.array(x_train), np.array(y_train)
        decision_tree_model = DecisionTreeRegressor(max_depth=5).fit(x_train,y_train)
        #print(x_train[-1])
        #x_train = x_train[-1].reshape(len(x_train[-1]), 1)
        #print(x_train[-1].reshape(1, -1))
        predictions = decision_tree_model.predict(x_train[-1].reshape(1, -1))
        #print(predictions)
        new_train_data = np.append(train_data, predictions)
        new_train_data = new_train_data.reshape(len(new_train_data), 1)
        dt_recursive(day - 1, new_train_data, data)   # Recursive call

def knn_recursive(day, train_data, data):
    if day == 0:

        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(dataset)
        predictions = scaler.inverse_transform(train_data)
        df = pd.DataFrame(predictions, columns = ['Predictions'])
        data.insert(1, "Predictions", predictions, True)
        data.to_csv('energy-knn-year.csv')
        #print(df)
        return df

    else:
        x_train = []
        y_train = []

        for i in range(lag, len(train_data)):
            x_train.append(train_data[i-lag:i, 0])
            y_train.append([train_data[i, 0]])

        x_train, y_train = np.array(x_train), np.array(y_train)
        KNN_model = KNeighborsRegressor(n_neighbors=3).fit(x_train,y_train)
        predictions = KNN_model.predict(x_train[-1].reshape(1, -1))
        new_train_data = np.append(train_data, predictions)
        new_train_data = new_train_data.reshape(len(new_train_data), 1)
        knn_recursive(day - 1, new_train_data, data)   # Recursive call

df = pd.read_csv('apple.csv')
df["Date"] = pd.to_datetime(df.Date, format="%m/%d/%Y")
df.sort_values(by='Date', inplace=True)
df = df.replace({'\$':''}, regex = True)
df.index = df['Date']
pd.set_option('display.max_rows', df.shape[0]+1)
df = df.astype({"Close": float})
data =  df.filter(['Close'])
print(len(data))
data = data[0:500]
dataset = data.values
training_data_len = math.ceil(len(dataset) * .5)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len , :]
day = math.ceil(len(dataset) * .5)

knn_recursive(day, train_data, data)
dt_recursive(day, train_data, data)
#start_time = time.time()
recursive(day, train_data, data)
#print("--- %s seconds ---" % (time.time() - start_time))
ridge_recursive(day, train_data, data)

n = 0
n_last = 0

df_predictions = pd.read_csv('energy-reservoir-year.csv')
df_predictions.index = df_predictions['Date']
predictions = df_predictions['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test = df_predictions['Close'][training_data_len + n_last :training_data_len + day - n]
print("NRMSE (Reservoir) = " + str(nrmse(y_test, predictions)))
print("R-squared (Reservoir) = " + str(r2_score(y_test, predictions)))

df_predictions_ridge = pd.read_csv('energy-ridge-year.csv')
df_predictions_ridge.index = df_predictions_ridge['Date']
predictions_ridge = df_predictions_ridge['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test_ridge = df_predictions_ridge['Close'][training_data_len + n_last : training_data_len + day - n]
print("NRMSE (Ridge regression) = " + str(nrmse(y_test_ridge, predictions_ridge)))
print("R-squared (Ridge regression) = " + str(r2_score(y_test_ridge, predictions_ridge)))

df_predictions_dt = pd.read_csv('energy-dt-year.csv')
df_predictions_dt.index = df_predictions_dt['Date']
predictions_dt = df_predictions_dt['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test_dt = df_predictions_dt['Close'][training_data_len + n_last : training_data_len + day - n]
print("NRMSE (Decision Tree Regression) = " + str(nrmse(y_test_dt, predictions_dt)))
print("R-squared (Decision Tree Regression) = " + str(r2_score(y_test_dt, predictions_dt)))

df_predictions_knn = pd.read_csv('energy-knn-year.csv')
df_predictions_knn.index = df_predictions_knn['Date']
predictions_knn = df_predictions_knn['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test_knn = df_predictions_knn['Close'][training_data_len + n_last : training_data_len + day - n]
print("NRMSE (KNN Regression) = " + str(nrmse(y_test_knn, predictions_knn)))
print("R-squared (KNN Regression) = " + str(r2_score(y_test_knn, predictions_knn)))


#n = 850
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Datetime', fontsize=18)
plt.ylabel('Energy consumption hourly/MW')
#print(data['DAYTON_MW'][0 : training_data_len])
#plt.plot(data['DAYTON_MW'])#[0 : training_data_len], '.-')
#plt.plot(df_predictions_ridge['DAYTON_MW'][:training_data_len], '.-')
plt.plot(df_predictions['Close'][training_data_len + n_last : training_data_len + day - n])
plt.plot(df_predictions['Predictions'][training_data_len + n_last : training_data_len + day - n])
plt.plot(df_predictions_ridge['Predictions'][training_data_len + n_last : training_data_len + day - n], '.-')
plt.plot(df_predictions_dt['Predictions'][training_data_len + n_last : training_data_len + day - n], '.-')
plt.plot(df_predictions_knn['Predictions'][training_data_len + n_last : training_data_len + day - n], '.-')
plt.legend(['Testing', 'Reservoir', 'Ridge', 'DT', 'KNN'], loc='lower right')
plt.xticks(rotation=90)
plt.xticks(fontsize=5)
plt.show()