import pandas as pd
import numpy as np
import math
from reservoirpy.nodes import Reservoir, Ridge
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from reservoirpy.observables import nrmse
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import time
from reservoirpy.datasets import mackey_glass

units = 60
leak_rate = 0.5
spectral_radius = 0.5
connectivity = 0.5
input_connectivity = 0.5
regularization = 1e-5
seed = 1234

reservoir = Reservoir(units, 
                    sr=spectral_radius,
                    lr=leak_rate, 
                    #rc_connectivity=connectivity,
                    #input_connectivity=input_connectivity, 
                    seed=seed)

readout = Ridge(ridge=regularization)
reservoir <<= readout
model = reservoir >> readout 

#scaler = MinMaxScaler(feature_range=(0,1))

lag = 250

def recursive(day, train_data, data, train_data_len):
    if day == 0:

        #scaler = MinMaxScaler(feature_range=(0,1))
        #scaler.fit_transform(data)
        #predictions = scaler.inverse_transform(train_data)
        predictions = train_data[timestep: train_data_len + timestep]
        print("NRMSE (Ridge regression) = " + str(nrmse(data, predictions)))
        print("R-squared (Ridge regression) = " + str(r2_score(data, predictions)))
        plt.plot(predictions[timestep: train_data_len + timestep])
        plt.plot(data)
        plt.show()
        #df = pd.DataFrame(predictions, columns = ['Predictions'])
        #data.insert(1, "Predictions", predictions, True)
        #data.to_csv('mg-reservoir.csv')
        #return df

    else:
        x_train = []
        y_train = []

        for i in range(lag, len(train_data)):
            x_train.append(train_data[i-lag:i, 0])
            y_train.append([train_data[i, 0]])

        x_train, y_train = np.array(x_train), np.array(y_train)
        model.fit(x_train, y_train)
        predictions = model.run(x_train[-1])
        new_train_data = np.append(train_data, predictions)
        new_train_data = new_train_data.reshape(len(new_train_data), 1)
        recursive(day - 1, new_train_data, data, train_data_len)   # Recursive call

def ridge_recursive(day, train_data, data, train_data_len):
    if day == 0:

        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(data)
        predictions = scaler.inverse_transform(train_data)
        #predictions = train_data
        print("NRMSE (Ridge regression) = " + str(nrmse(data, predictions[timestep: train_data_len + timestep])))
        print("R-squared (Ridge regression) = " + str(r2_score(data, predictions[timestep: train_data_len + timestep])))
        plt.plot(predictions[timestep: train_data_len + timestep])
        plt.plot(data)
        plt.show()
        #df = pd.DataFrame(predictions, columns = ['Predictions'])
        #data.insert(1, "Predictions", predictions, True)
        #data.to_csv('mg-ridge.csv')
        #print(df)
        #return df

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
        ridge_recursive(day - 1, new_train_data, data, train_data_len)   # Recursive call

def dt_recursive(day, train_data, data):
    if day == 0:

        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(dataset)
        predictions = scaler.inverse_transform(train_data)
        df = pd.DataFrame(predictions, columns = ['Predictions'])
        data.insert(1, "Predictions", predictions, True)
        data.to_csv('mg-dt.csv')
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
        data.to_csv('mg-knn.csv')
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


timesteps = 1800
tau = 17
X = mackey_glass(timesteps, tau=tau)
#print(X)

# Reading the input data
training_data_len = math.ceil(len(X) * .5)
#scaler = MinMaxScaler(feature_range=(0,1))
#scaled_data = scaler.fit_transform(X)
#train_data = scaled_data[0:training_data_len , :]
train_data = X[0:training_data_len , :]
timestep = math.ceil(len(X) * .5)
print(timestep)

#knn_recursive(timestep, train_data, data)
#dt_recursive(timestep, train_data, data)
recursive(timestep, train_data, X[timestep:timestep + training_data_len], training_data_len)
ridge_recursive(timestep, train_data, X[timestep:timestep + training_data_len], training_data_len)

""" n = 0
n_last = 0

df_predictions = pd.read_csv('mg-reservoir.csv')
predictions = df_predictions['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test = df_predictions['DAYTON_MW'][training_data_len + n_last :training_data_len + day - n]
print("NRMSE (Reservoir) = " + str(nrmse(y_test, predictions)))
print("R-squared (Reservoir) = " + str(r2_score(y_test, predictions)))

df_predictions_ridge = pd.read_csv('energy-ridge-year.csv')
df_predictions_ridge.index = df_predictions_ridge['Datetime']
predictions_ridge = df_predictions_ridge['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test_ridge = df_predictions_ridge['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n]
print("NRMSE (Ridge regression) = " + str(nrmse(y_test_ridge, predictions_ridge)))
print("R-squared (Ridge regression) = " + str(r2_score(y_test_ridge, predictions_ridge)))

df_predictions_dt = pd.read_csv('energy-dt-year.csv')
df_predictions_dt.index = df_predictions_dt['Datetime']
predictions_dt = df_predictions_dt['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test_dt = df_predictions_dt['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n]
print("NRMSE (Decision Tree Regression) = " + str(nrmse(y_test_dt, predictions_dt)))
print("R-squared (Decision Tree Regression) = " + str(r2_score(y_test_dt, predictions_dt)))

df_predictions_knn = pd.read_csv('energy-knn-year.csv')
df_predictions_knn.index = df_predictions_knn['Datetime']
predictions_knn = df_predictions_knn['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test_knn = df_predictions_knn['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n]
print("NRMSE (KNN Regression) = " + str(nrmse(y_test_knn, predictions_knn)))
print("R-squared (KNN Regression) = " + str(r2_score(y_test_knn, predictions_knn))) """


""" #n = 850
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Datetime', fontsize=18)
plt.ylabel('Energy consumption hourly/MW')
#print(data['DAYTON_MW'][0 : training_data_len])
#plt.plot(data['DAYTON_MW'])#[0 : training_data_len], '.-')
#plt.plot(df_predictions_ridge['DAYTON_MW'][:training_data_len], '.-')
plt.plot(df_predictions['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n])
plt.plot(df_predictions['Predictions'][training_data_len + n_last : training_data_len + day - n])
plt.plot(df_predictions_ridge['Predictions'][training_data_len + n_last : training_data_len + day - n], '.-')
plt.plot(df_predictions_dt['Predictions'][training_data_len + n_last : training_data_len + day - n], '.-')
plt.plot(df_predictions_knn['Predictions'][training_data_len + n_last : training_data_len + day - n], '.-')
plt.legend(['Testing', 'Reservoir'], loc='lower right')
plt.xticks(rotation=90)
plt.xticks(fontsize=5)
plt.show() """

""" # making subplots
#fig, ax = plt.subplots(4)
fig = plt.figure()
fig.text(0.89, 0.06, 'Hour', ha='center', fontsize=20)
fig.text(0.07, 0.5, 'Energy consumption hourly/MW', va='center', rotation='vertical', fontsize=20)

#ax = fig.add_subplot(111)    # The big subplot
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
#fig(figsize=(16,8))
# set data with subplots and plot

# First subplot
ax1.plot(df_predictions['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
ax1.plot(df_predictions['Predictions'][training_data_len + n_last : training_data_len + day - n], 'g', linewidth=3)
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax1.legend(['Actual', 'Ridge with reservoir'], loc='center left', prop={'size': 15})

# Second subplot
ax2.plot(df_predictions['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
ax2.plot(df_predictions_ridge['Predictions'][training_data_len + n_last : training_data_len + day - n], '#d62728', linewidth=3)
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.legend(['Actual', 'Ridge'], loc='center left', prop={'size': 15})

# Third subplot
ax3.plot(df_predictions['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
ax3.plot(df_predictions_dt['Predictions'][training_data_len + n_last : training_data_len + day - n], '#9467bd', linewidth=3)
ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax3.legend(['Actual', 'Decision Tree'], loc='center left', prop={'size': 15})

# Fourth subplot
df_predictions_knn.index = np.arange(1, len(df_predictions_knn) + 1)
#print(df_predictions_knn)
ax4.plot(df_predictions_knn['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
ax4.plot(df_predictions_knn['Predictions'][training_data_len + n_last : training_data_len + day - n], '#8c564b', linewidth=3)
#ax4.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax4.legend(['Actual', 'KNN'], loc='center left', prop={'size': 15})
ax4.set_xticks(np.arange(training_data_len + n_last + 1,  training_data_len + day - n  + 1, 20))
#ax4.set_xticklabels(df_predictions.index[training_data_len + n_last : training_data_len + day - n][::2], rotation=90)
#print(df_predictions_dt['Predictions'][training_data_len + n_last : training_data_len + day - n])

# set the title to subplots
ax1.set_title("Reservoir", fontsize=15)
ax2.set_title("Ridge", fontsize=15)
ax3.set_title("Decision Tree", fontsize=15)
ax4.set_title("KNN", fontsize=15)
ax1.grid(axis='y')
ax2.grid(axis='y')
ax3.grid(axis='y')
ax4.grid(axis='y')
plt.xticks(rotation=90)
#plt.xticks(fontsize=8)
# set spacing
#fig.tight_layout()
plt.show() """