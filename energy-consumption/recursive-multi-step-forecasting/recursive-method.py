import pandas as pd
import numpy as np
import math
import os.path
from reservoirpy.nodes import Reservoir, Ridge
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from reservoirpy.observables import nrmse, rsquare
from datetime import datetime
from sklearn.metrics import r2_score
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

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
lag = 250

def recursive(day, train_data, data):
    if day == 0:

        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(dataset)
        predictions = scaler.inverse_transform(train_data)
        data.insert(1, "Predictions", predictions, True)
        data.to_csv(os.path.join('energy-consumption/recursive-multi-step-forecasting','energy-reservoir.csv'))

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
        recursive(day - 1, new_train_data, data)   # Recursive call

def ridge_recursive(day, train_data, data):
    if day == 0:

        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(dataset)
        predictions = scaler.inverse_transform(train_data)
        data.insert(1, "Predictions", predictions, True)
        data.to_csv(os.path.join('energy-consumption/recursive-multi-step-forecasting','energy-ridge.csv'))

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
        data.insert(1, "Predictions", predictions, True)
        data.to_csv(os.path.join('energy-consumption/recursive-multi-step-forecasting','energy-dt.csv'))

    else:
        x_train = []
        y_train = []

        for i in range(lag, len(train_data)):
            x_train.append(train_data[i-lag:i, 0])
            y_train.append([train_data[i, 0]])

        x_train, y_train = np.array(x_train), np.array(y_train)
        decision_tree_model = DecisionTreeRegressor(max_depth=5).fit(x_train,y_train)
        predictions = decision_tree_model.predict(x_train[-1].reshape(1, -1))
        new_train_data = np.append(train_data, predictions)
        new_train_data = new_train_data.reshape(len(new_train_data), 1)
        dt_recursive(day - 1, new_train_data, data)   # Recursive call

def knn_recursive(day, train_data, data):
    if day == 0:

        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(dataset)
        predictions = scaler.inverse_transform(train_data)
        data.insert(1, "Predictions", predictions, True)
        data.to_csv(os.path.join('energy-consumption/recursive-multi-step-forecasting','energy-knn.csv'))

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

# Reading the input data
d = pd.read_csv('energy-consumption/recursive-multi-step-forecasting/DAYTON_hourly.csv')

# Formating to datetime
d['Datetime'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in d['Datetime']]

# Making sure there are no duplicated data
# If there are some duplicates, average the data during those duplicated days
d = d.groupby('Datetime', as_index=False)['DAYTON_MW'].mean()

# Sorting the values
d.sort_values('Datetime', inplace=True)
d.index = d['Datetime']
data =  d.filter(['DAYTON_MW'])
data = data[:1900]
dataset = data.values
training_data_len = math.ceil(len(dataset) * .5)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len , :]
day = math.ceil(len(dataset) * .5)

print("----------Reservoir recursive starts----------")
recursive(day, train_data, data)
print("----------Reservoir recursive ends----------")
print("----------Ridge recursive starts----------")
ridge_recursive(day, train_data, data)
print("----------Ridge recursive ends----------")
print("----------KNN recursive starts----------")
knn_recursive(day, train_data, data)
print("----------KNN recursive ends----------")
print("----------Decision Tree recursive starts----------")
dt_recursive(day, train_data, data)
print("----------Decision Tree recursive ends----------")


n = 0
n_last = 0

print("----------NRMSE and R2 for 950 hours----------")
df_predictions = pd.read_csv('energy-consumption/recursive-multi-step-forecasting/energy-reservoir.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test = df_predictions['DAYTON_MW'][training_data_len + n_last :training_data_len + day - n]
print("NRMSE (Reservoir) = " + str(nrmse(y_test, predictions)))
print("R-squared (Reservoir) = " + str(r2_score(y_test, predictions)))
print("MAPE (Reservoir) = " + str(mean_absolute_percentage_error(y_test, predictions)) + "\n")

df_predictions_ridge = pd.read_csv('energy-consumption/recursive-multi-step-forecasting/energy-ridge.csv')
df_predictions_ridge.index = df_predictions_ridge['Datetime']
predictions_ridge = df_predictions_ridge['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test_ridge = df_predictions_ridge['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n]
print("NRMSE (Ridge regression) = " + str(nrmse(y_test_ridge, predictions_ridge)))
print("R-squared (Ridge regression) = " + str(r2_score(y_test_ridge, predictions_ridge)))
print("MAPE (Ridge regression) = " + str(mean_absolute_percentage_error(y_test_ridge, predictions_ridge)) + "\n")

df_predictions_dt = pd.read_csv('energy-consumption/recursive-multi-step-forecasting/energy-dt.csv')
df_predictions_dt.index = df_predictions_dt['Datetime']
predictions_dt = df_predictions_dt['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test_dt = df_predictions_dt['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n]
print("NRMSE (Decision Tree Regression) = " + str(nrmse(y_test_dt, predictions_dt)))
print("R-squared (Decision Tree Regression) = " + str(r2_score(y_test_dt, predictions_dt)))
print("MAPE (Decision Tree Regression) = " + str(mean_absolute_percentage_error(y_test_dt, predictions_dt)) + "\n")

df_predictions_knn = pd.read_csv('energy-consumption/recursive-multi-step-forecasting/energy-knn.csv')
df_predictions_knn.index = df_predictions_knn['Datetime']
predictions_knn = df_predictions_knn['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test_knn = df_predictions_knn['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n]
print("NRMSE (KNN Regression) = " + str(nrmse(y_test_knn, predictions_knn)))
print("R-squared (KNN Regression) = " + str(r2_score(y_test_knn, predictions_knn)))
print("MAPE (KNN Regression) = " + str(mean_absolute_percentage_error(y_test_knn, predictions_knn)) + "\n")

# Training and testing data sets
plt.figure(figsize=(16,8))
plt.xlabel('Hour', fontsize=20)
plt.ylabel('Energy consumption hourly/MW', fontsize=20)
plt.title("Hourly energy consumption in Dayton", fontsize=20)
plt.plot(df_predictions['DAYTON_MW'][: training_data_len], 'orange')
plt.plot(df_predictions['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4')
plt.legend(['Training', 'Testing'], loc='lower right', prop={'size': 20})
plt.xticks(rotation=90)
plt.xticks(fontsize=3)
plt.ylim(0, 5000)
plt.grid(axis='y')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.show()

# Alternate plot for results
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.xlabel('Datetime')
plt.ylabel('Energy consumption hourly/MW')
plt.plot(df_predictions['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
plt.plot(df_predictions['Predictions'][training_data_len + n_last : training_data_len + day - n], 'g')
plt.legend(['Actual', 'Ridge with reservoir'], loc='lower right')
plt.xticks(rotation=90)
plt.xticks(fontsize=3)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(2,2,2)
plt.xlabel('Datetime')
plt.ylabel('Energy consumption hourly/MW')
plt.plot(df_predictions['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
plt.plot(df_predictions_ridge['Predictions'][training_data_len + n_last : training_data_len + day - n], '#d62728')
plt.legend(['Actual', 'Ridge'], loc='lower right')
plt.xticks(rotation=90)
plt.xticks(fontsize=3)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(2,2,3)
plt.xlabel('Datetime')
plt.ylabel('Energy consumption hourly/MW')
plt.plot(df_predictions['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
plt.plot(df_predictions_dt['Predictions'][training_data_len + n_last : training_data_len + day - n], '#9467bd')
plt.legend(['Actual', 'Decision Tree'], loc='lower right')
plt.xticks(rotation=90)
plt.xticks(fontsize=3)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(2,2,4)
plt.xlabel('Datetime')
plt.ylabel('Energy consumption hourly/MW')
plt.plot(df_predictions['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
plt.plot(df_predictions_knn['Predictions'][training_data_len + n_last : training_data_len + day - n], '#8c564b')
plt.legend(['Actual', "KNN"], loc='lower right')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.xticks(rotation=90)
plt.xticks(fontsize=3)
plt.show()

# Plots for 950 hours
# Making subplots
fig = plt.figure()
fig.text(0.89, 0.06, 'Hour', ha='center', fontsize=20)
fig.text(0.5, 0.95, '950 hours', ha='center', fontsize=20)
fig.text(0.07, 0.5, 'Energy consumption hourly/MW', va='center', rotation='vertical', fontsize=20)

ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

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
ax4.plot(df_predictions_knn['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
ax4.plot(df_predictions_knn['Predictions'][training_data_len + n_last : training_data_len + day - n], '#8c564b', linewidth=3)
#ax4.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax4.legend(['Actual', 'KNN'], loc='center left', prop={'size': 15})
ax4.set_xticks(np.arange(training_data_len + n_last + 1,  training_data_len + day - n  + 1, 20))

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
plt.show()

n = 850
n_last = 0

# Plots for first 100 hours
# Making subplots
#fig, ax = plt.subplots(4)
fig = plt.figure()
fig.text(0.89, 0.06, 'Hour', ha='center', fontsize=20)
fig.text(0.5, 0.95, 'First 100 hours', ha='center', fontsize=20)
fig.text(0.07, 0.5, 'Energy consumption hourly/MW', va='center', rotation='vertical', fontsize=20)

#ax = fig.add_subplot(111)    # The big subplot
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)


# First subplot
ax1.plot(df_predictions['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
ax1.plot(df_predictions['Predictions'][training_data_len + n_last : training_data_len + day - n], 'g', linewidth=3)
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax1.legend(['Actual', 'Ridge with reservoir'], loc='center right', prop={'size': 15})

# Second subplot
ax2.plot(df_predictions['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
ax2.plot(df_predictions_ridge['Predictions'][training_data_len + n_last : training_data_len + day - n], '#d62728', linewidth=3)
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.legend(['Actual', 'Ridge'], loc='center right', prop={'size': 15})

# Third subplot
ax3.plot(df_predictions['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
ax3.plot(df_predictions_dt['Predictions'][training_data_len + n_last : training_data_len + day - n], '#9467bd', linewidth=3)
ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax3.legend(['Actual', 'Decision Tree'], loc='center right', prop={'size': 15})

# Fourth subplot
df_predictions_knn.index = np.arange(1, len(df_predictions_knn) + 1)
ax4.plot(df_predictions_knn['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n], color='#1f77b4', linestyle='--')
ax4.plot(df_predictions_knn['Predictions'][training_data_len + n_last : training_data_len + day - n], '#8c564b', linewidth=3)
#ax4.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax4.legend(['Actual', 'KNN'], loc='center right', prop={'size': 15})
ax4.set_xticks(np.arange(training_data_len + n_last + 1,  training_data_len + day - n  + 1, 2))

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
plt.show()

print("----------NRMSE and R2 for first 100 hours----------")
df_predictions = pd.read_csv('energy-consumption/recursive-multi-step-forecasting/energy-reservoir.csv')
df_predictions.index = df_predictions['Datetime']
predictions = df_predictions['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test = df_predictions['DAYTON_MW'][training_data_len + n_last :training_data_len + day - n]
print("NRMSE (Reservoir) = " + str(nrmse(y_test, predictions)))
print("R-squared (Reservoir) = " + str(r2_score(y_test, predictions)))
print("MAPE (Reservoir) = " + str(mean_absolute_percentage_error(y_test, predictions)) + "\n")

df_predictions_ridge = pd.read_csv('energy-consumption/recursive-multi-step-forecasting/energy-ridge.csv')
df_predictions_ridge.index = df_predictions_ridge['Datetime']
predictions_ridge = df_predictions_ridge['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test_ridge = df_predictions_ridge['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n]
print("NRMSE (Ridge regression) = " + str(nrmse(y_test_ridge, predictions_ridge)))
print("R-squared (Ridge regression) = " + str(r2_score(y_test_ridge, predictions_ridge)))
print("MAPE (Ridge regression) = " + str(mean_absolute_percentage_error(y_test_ridge, predictions_ridge)) + "\n")

df_predictions_dt = pd.read_csv('energy-consumption/recursive-multi-step-forecasting/energy-dt.csv')
df_predictions_dt.index = df_predictions_dt['Datetime']
predictions_dt = df_predictions_dt['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test_dt = df_predictions_dt['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n]
print("NRMSE (Decision Tree Regression) = " + str(nrmse(y_test_dt, predictions_dt)))
print("R-squared (Decision Tree Regression) = " + str(r2_score(y_test_dt, predictions_dt)))
print("MAPE (Decision Tree Regression) = " + str(mean_absolute_percentage_error(y_test_dt, predictions_dt)) + "\n")

df_predictions_knn = pd.read_csv('energy-consumption/recursive-multi-step-forecasting/energy-knn.csv')
df_predictions_knn.index = df_predictions_knn['Datetime']
predictions_knn = df_predictions_knn['Predictions'][training_data_len + n_last : training_data_len + day - n]
y_test_knn = df_predictions_knn['DAYTON_MW'][training_data_len + n_last : training_data_len + day - n]
print("NRMSE (KNN Regression) = " + str(nrmse(y_test_knn, predictions_knn)))
print("R-squared (KNN Regression) = " + str(r2_score(y_test_knn, predictions_knn)))
print("MAPE (KNN Regression) = " + str(mean_absolute_percentage_error(y_test_knn, predictions_knn)) + "\n")