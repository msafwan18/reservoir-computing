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

# Objective functions accepted by ReservoirPy must respect some conventions:
#  - dataset and config arguments are mandatory, like the empty '*' expression.
#  - all parameters that will be used during the search must be placed after the *.
#  - the function must return a dict with at least a 'loss' key containing the result
# of the loss function. You can add any additional metrics or information with other
# keys in the dict. See hyperopt documentation for more informations.
""" def objective(dataset, config, *, iss, N, sr, lr, ridge, seed):

    # This step may vary depending on what you put inside 'dataset'
    train_data, validation_data = dataset
    X_train, y_train = train_data
    X_val, y_val = validation_data

    # You can access anything you put in the config
    # file from the 'config' parameter.
    instances = config["instances_per_trial"]

    # The seed should be changed across the instances,
    # to be sure there is no bias in the results
    # due to initialization.
    variable_seed = seed

    losses = []; r2s = [];
    for n in range(instances):
        # Build your model given the input parameters
        reservoir = Reservoir(N,
                              sr=sr,
                              lr=lr,
                              inut_scaling=iss,
                              seed=variable_seed)

        readout = Ridge(ridge=ridge)

        model = reservoir >> readout


        # Train your model and test your model.
        predictions = model.fit(X_train, y_train) \
                           .run(x_test)

        loss = nrmse(y_test, predictions, norm_value=np.ptp(X_train))
        r2 = rsquare(y_test, predictions)

        # Change the seed between instances
        variable_seed += 1

        losses.append(loss)
        r2s.append(r2)

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': np.mean(losses),
            'r2': np.mean(r2s)}

hyperopt_config = {
    "exp": f"hyperopt-multiscroll", # the experimentation name
    "hp_max_evals": 200,             # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",           # the method used by hyperopt to chose those sets (see below)
    "seed": 42,                      # the random state seed, to ensure reproducibility
    "instances_per_trial": 3,        # how many random ESN will be tried with each sets of parameters
    "hp_space": {                    # what are the ranges of parameters explored
        "N": ["choice", 500],             # the number of neurons is fixed to 300
        "sr": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-6 and 10
        "leak": ["loguniform", 1e-3, 1],  # idem with the leaking rate, from 1e-3 to 1
        "iss": ["choice", 0.9],           # the input scaling is fixed
        "ridge": ["choice", 1e-7],        # and so is the regularization parameter.
        "seed": ["choice", 1234]          # an other random seed for the ESN initialization
    }
}


# we precautionously save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f) """

def ridge_regression():
    ridge = Ridge(ridge=1e-5)
    ridge = ridge.fit(x_train, y_train)
    predictions = ridge.run(x_test)
    predictions = scaler.inverse_transform(predictions)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    print("Mean squared error (Ridge Regression) = " , nrmse(y_test, predictions))
    print("Rsquare = " , rsquare(y_test, predictions))
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['Close'])
    plt.plot(valid['Close'])
    plt.plot(valid['Predictions'])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    #plt.show()

def reservoir_parameter():

    units_list = []
    leak_rate_list = []
    spectral_radius_list = []
    mse_list = []
    r_square_list = []
    nrmse_list = []
    list_of_tuples = []

    #for i in  range(10, 110, 10): #np.arange(0.1, 1.1, 0.1):
    #for i in np.arange(0.1, 1.1, 0.1): 
    for j in np.arange(0.1, 1.1, 0.1):
        for k in np.arange(0.1, 1.1, 0.1):
            units = 90
            leak_rate = 0.4
            spectral_radius = 1.0
            connectivity = j
            input_connectivity = k
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
            model.fit(x_train, y_train)
            predictions = model.run(x_test)

            predictions = scaler.inverse_transform(predictions)
            train = data[:training_data_len]
            valid = data[training_data_len:]
            valid['Predictions'] = predictions

            units_list.append(i)
            leak_rate_list.append(j)
            spectral_radius_list.append(k)
            mse_list.append(mse(y_test, predictions))
            r_square_list.append(rsquare(y_test, predictions))
            nrmse_list.append(nrmse(y_test, predictions))
    
    list_of_tuples = list(zip(units_list, leak_rate_list, spectral_radius_list, mse_list, r_square_list, nrmse_list))
    df = pd.DataFrame(list_of_tuples, columns=['units', 'leak_rate', 'spectral_radius', 'mse', 'r_square', 'nrmse'])
    min = df['mse'].min()
    max = df['r_square'].max()
    print(df.loc[df['mse'] == min])
    print(df.loc[df['r_square'] == max])
    #df.to_csv('hyperparameter-lag3-training-20-percents.csv')
    #fig = px.scatter(df, x="units", y="leak_rate", size="mse", size_max=45, log_x=False, log_y=False)
    #fig.show()


def reservoir_60():

    units = 60
    leak_rate = 0.36
    spectral_radius = 0.61
    connectivity = 0.6
    input_connectivity = 0.1
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
    model.fit(x_train, y_train)
    predictions = model.run(x_test)

    predictions = scaler.inverse_transform(predictions)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
 
    print("Reservoir NRMSE (loss) = " , mse(y_test, predictions))
    print("Rsquare = " , rsquare(y_test, predictions))
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['Close'])
    plt.plot(valid['Close'])
    plt.plot(valid['Predictions'])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    #plt.show()

def reservoir_lag3_training80():

    units = 10
    leak_rate = 0.9
    spectral_radius = 0.2
    connectivity = 0.4
    input_connectivity = 0.3
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
    model.fit(x_train, y_train)
    predictions = model.run(x_test)

    predictions = scaler.inverse_transform(predictions)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
 
    print("Reservoir NRMSE (loss) = " , mse(y_test, predictions))
    print("Rsquare = " , rsquare(y_test, predictions))
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['Close'])
    plt.plot(valid['Close'])
    plt.plot(valid['Predictions'])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

def reservoir_lag3_training20():

    units = 90
    leak_rate = 0.4
    spectral_radius = 1.0
    connectivity = 0.7
    input_connectivity = 0.1
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
    model.fit(x_train, y_train)
    predictions = model.run(x_test)

    predictions = scaler.inverse_transform(predictions)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
 
    print("Reservoir NRMSE (loss) = " , nrmse(y_test, predictions))
    print("Rsquare = " , rsquare(y_test, predictions))
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['Close'])
    plt.plot(valid['Close'])
    plt.plot(valid['Predictions'])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

df = pd.read_csv('data/apple.csv')
df["Date"] = pd.to_datetime(df.Date, format="%m/%d/%Y")
df.sort_values(by='Date', inplace=True)
df = df.replace({'\$':''}, regex = True)
df.index = df['Date']
pd.set_option('display.max_rows', df.shape[0]+1)
df = df.astype({"Close": float})
data =  df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .2)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len , :]

x_train = []
y_train = []

lag = 8

for i in range(lag, len(train_data)):
    x_train.append(train_data[i-lag:i, 0])
    y_train.append([train_data[i, 0]])


x_train, y_train = np.array(x_train), np.array(y_train)
test_data = scaled_data[training_data_len - lag:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(lag, len(test_data)):
    x_test.append(test_data[i-lag:i, 0])

x_test = np.array(x_test)

#reservoir_parameter()
ridge_regression()
#reservoir_3()
reservoir_lag3_training20()
#dataset = ((x_train, y_train), (x_test, y_test))
#best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")
#fig = plot_hyperopt_report(hyperopt_config["exp"], ("leak", "sr"), metric="r2")