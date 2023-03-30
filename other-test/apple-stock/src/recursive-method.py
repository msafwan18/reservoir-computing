import pandas as pd
import numpy as np
import math
from reservoirpy.nodes import Reservoir, Ridge
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.observables import mse
from reservoirpy.hyper import research
from reservoirpy.hyper import plot_hyperopt_report
import json
import plotly.express as px


df = pd.read_csv('apple-stock/data/apple.csv')
df["Date"] = pd.to_datetime(df.Date, format="%m/%d/%Y")
df.sort_values(by='Date', inplace=True)
df = df.replace({'\$':''}, regex = True)
df.index = df['Date']
pd.set_option('display.max_rows', df.shape[0]+1)
df = df.astype({"Close": float})
data =  df.filter(['Close'])
dataset = data.values

units = 50
leak_rate = 0.3
spectral_radius = 0.43 #0.48
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

def recursive(day, train_data, data):
    if day == 0:

        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(dataset)
        predictions = scaler.inverse_transform(train_data)
        df = pd.DataFrame(predictions, columns = ['Predictions'])
        data.insert(1, "Predictions", predictions, True)
        data.to_csv('predictions-train-80.csv')
        #print(df)
        return df

    else:
        x_train = []
        y_train = []

        lag = 40

        for i in range(lag, len(train_data)):
            x_train.append(train_data[i-lag:i, 0])
            y_train.append([train_data[i, 0]])

        x_train, y_train = np.array(x_train), np.array(y_train)
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
        data.to_csv('ridge-train-80.csv')
        #print(df)
        return df

    else:
        x_train = []
        y_train = []

        lag = 60

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

def main():
    df = pd.read_csv('apple-stock/data/apple.csv')
    df["Date"] = pd.to_datetime(df.Date, format="%m/%d/%Y")
    df.sort_values(by='Date', inplace=True)
    df = df.replace({'\$':''}, regex = True)
    df.index = df['Date']
    pd.set_option('display.max_rows', df.shape[0]+1)
    df = df.astype({"Close": float})
    data =  df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len , :]
    day = math.ceil(len(dataset) * .2) - 1
    print(len(data))
    #recursive(day, train_data, data)
    #ridge_recursive(day, train_data, data)

    """predictions = pd.read_csv('ridge-train-80.csv')
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)')
    plt.plot(predictions['Close'])
    plt.plot(predictions['Predictions'])
    plt.legend(['Train', 'Predictions'], loc='lower right')
    plt.show() """

    predictions = pd.read_csv('predictions-train-80.csv')
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)')
    plt.plot(predictions['Close'])
    plt.plot(predictions['Predictions'])
    plt.legend(['Train', 'Predictions'], loc='lower right')
    plt.show()
 
if __name__ == "__main__":
    main()