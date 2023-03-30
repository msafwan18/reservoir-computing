import pandas as pd
import reservoirpy as rpy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge, ESN
from reservoirpy.datasets import to_forecasting
from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import nrmse, rsquare
import numpy as np
rpy.set_seed(42)

def reservoir(X_train, y_train, X_test, y_test):
    units = 100
    leak_rate = 0.3
    spectral_radius = 0.99
    input_scaling = 1.0
    connectivity = 0.1      # - density of reservoir internal matrix
    input_connectivity = 0.2  # and of reservoir input matrix
    regularization = 1e-4
    seed = 1234             # for reproducibility

    reservoir = Reservoir(units, sr=spectral_radius,
                        lr=leak_rate, rc_connectivity=connectivity,
                        input_connectivity=input_connectivity, seed=seed, input_scaling=input_scaling)

    readout = Ridge(1, ridge=regularization)
    esn = reservoir >> readout
    esn = esn.fit(X_train, y_train)
    reservoir_predictions = esn.run(X_test)
    units_rsquare = 70
    return reservoir_predictions, y_test


def plot_results(y_pred, y_test, sample=400):

    fig = plt.figure(figsize=(15, 7))
    plt.subplot(211)
    plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")
    plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")
    plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")

    plt.legend()
    plt.show()


def units(X_train, y_train, X_test, y_test):
    units_list = []
    leak_rate_list = []
    r_square_list = []
    nrmse_list = []
    for i in range(10, 100, 10):
        units = i
        leak_rate = 0.3
        spectral_radius = 0.99
        input_scaling = 1.0
        connectivity = 0.1      # - density of reservoir internal matrix
        input_connectivity = 0.2  # and of reservoir input matrix
        regularization = 1e-4
        seed = 1234             # for reproducibility

        reservoir = Reservoir(units, sr=spectral_radius,
                            lr=leak_rate, rc_connectivity=connectivity,
                            input_connectivity=input_connectivity, seed=seed, input_scaling=input_scaling)

        readout = Ridge(1, ridge=regularization)
        esn = reservoir >> readout
        esn = esn.fit(X_train, y_train)
        reservoir_predictions = esn.run(X_test)
        units_list.append(i)
        leak_rate_list.append(j)
        r_square = rsquare(y_test, reservoir_predictions)
        r_square_list.append(r_square)
        nrmse_measured = nrmse(y_test, reservoir_predictions)
        nrmse_list.append(nrmse_measured)


    plt.plot(units_list, r_square_list, '-o', label="r-squared value",)
    plt.title("Relationship of number of units neurons with accuracy")
    plt.xlabel("Number of units")
    #plt.ylabel("r-squared value")
    #plt.show()
    plt.plot(units_list, nrmse_list, '-o', label="nrmse value")
    #plt.title("Relationship of number of units neurons with accuracy")
    #plt.xlabel("Number of units")
    #plt.ylabel("Normalised root mean square")
    plt.legend()
    plt.show()


def leak_rate(X_train, y_train, X_test, y_test):
    leak_rate_list = []
    r_square_list = []
    nrmse_list = []
    for i in np.arange(0.1, 1, 0.1):
        units = 10
        leak_rate = i
        spectral_radius = 0.99
        input_scaling = 1.0
        connectivity = 0.1      # - density of reservoir internal matrix
        input_connectivity = 0.2  # and of reservoir input matrix
        regularization = 1e-4
        seed = 1234             # for reproducibility

        reservoir = Reservoir(units, sr=spectral_radius,
                            lr=leak_rate, rc_connectivity=connectivity,
                            input_connectivity=input_connectivity, seed=seed, input_scaling=input_scaling)

        readout = Ridge(1, ridge=regularization)
        esn = reservoir >> readout
        esn = esn.fit(X_train, y_train)
        reservoir_predictions = esn.run(X_test)
        leak_rate_list.append(i)
        r_square = rsquare(y_test, reservoir_predictions)
        r_square_list.append(r_square)
        nrmse_measured = nrmse(y_test, reservoir_predictions)
        nrmse_list.append(nrmse_measured)


    plt.plot(leak_rate_list, r_square_list, '-o', label="r-squared value",)
    plt.title("Relationship of leak rate with accuracy")
    plt.xlabel("Leak rate")
    plt.plot(leak_rate_list, nrmse_list, '-o', label="nrmse value")
    plt.legend()
    plt.show()


def spectral_radius(X_train, y_train, X_test, y_test):
    spectral_radius_list = []
    r_square_list = []
    nrmse_list = []
    for i in np.arange(0.1, 1, 0.1):
        units = 10
        leak_rate = 0.3
        spectral_radius = i
        input_scaling = 1.0
        connectivity = 0.1      # - density of reservoir internal matrix
        input_connectivity = 0.2  # and of reservoir input matrix
        regularization = 1e-4
        seed = 1234             # for reproducibility

        reservoir = Reservoir(units, sr=spectral_radius,
                            lr=leak_rate, rc_connectivity=connectivity,
                            input_connectivity=input_connectivity, seed=seed, input_scaling=input_scaling)

        readout = Ridge(1, ridge=regularization)
        esn = reservoir >> readout
        esn = esn.fit(X_train, y_train)
        reservoir_predictions = esn.run(X_test)
        spectral_radius_list.append(i)
        r_square = rsquare(y_test, reservoir_predictions)
        r_square_list.append(r_square)
        nrmse_measured = nrmse(y_test, reservoir_predictions)
        nrmse_list.append(nrmse_measured)


    plt.plot(spectral_radius_list, r_square_list, '-o', label="r-squared value",)
    plt.title("Relationship of spectral radius with accuracy")
    plt.xlabel("Spectral radius")
    plt.plot(spectral_radius_list, nrmse_list, '-o', label="nrmse value")
    plt.legend()
    plt.show()


def input_scaling(X_train, y_train, X_test, y_test):
    input_scaling_list = []
    r_square_list = []
    nrmse_list = []
    for i in np.arange(0.1, 20, 0.5):
    #for i in range(10, 100, 10):
        units = 10
        leak_rate = 0.3
        spectral_radius = 0.99
        input_scaling = i
        connectivity = 0.1      # - density of reservoir internal matrix
        input_connectivity = 0.2  # and of reservoir input matrix
        regularization = 1e-4
        seed = 1234             # for reproducibility

        reservoir = Reservoir(units, sr=spectral_radius,
                            lr=leak_rate, rc_connectivity=connectivity,
                            input_connectivity=input_connectivity, seed=seed, input_scaling=input_scaling)

        readout = Ridge(1, ridge=regularization)
        esn = reservoir >> readout
        esn = esn.fit(X_train, y_train)
        reservoir_predictions = esn.run(X_test)
        input_scaling_list.append(i)
        r_square = rsquare(y_test, reservoir_predictions)
        r_square_list.append(r_square)
        nrmse_measured = nrmse(y_test, reservoir_predictions)
        nrmse_list.append(nrmse_measured)


    plt.plot(input_scaling_list, r_square_list, '-o', label="r-squared value",)
    plt.title("Relationship of input scaling with accuracy")
    plt.xlabel("Input scaling")
    plt.plot( input_scaling_list, nrmse_list, '-o', label="nrmse value")
    plt.legend()
    plt.show()


def connectivity(X_train, y_train, X_test, y_test):
    connectivity_list = []
    r_square_list = []
    nrmse_list = []
    for i in np.arange(0.1, 1, 0.1):
    #for i in range(10, 100, 10):
        units = 10
        leak_rate = 0.3
        spectral_radius = 0.99
        input_scaling = 1.0
        connectivity = i      # - density of reservoir internal matrix
        input_connectivity = 0.2  # and of reservoir input matrix
        regularization = 1e-4
        seed = 1234             # for reproducibility

        reservoir = Reservoir(units, sr=spectral_radius,
                            lr=leak_rate, rc_connectivity=connectivity,
                            input_connectivity=input_connectivity, seed=seed, input_scaling=input_scaling)

        readout = Ridge(1, ridge=regularization)
        esn = reservoir >> readout
        esn = esn.fit(X_train, y_train)
        reservoir_predictions = esn.run(X_test)
        connectivity_list.append(i)
        r_square = rsquare(y_test, reservoir_predictions)
        r_square_list.append(r_square)
        nrmse_measured = nrmse(y_test, reservoir_predictions)
        nrmse_list.append(nrmse_measured)


    plt.plot(connectivity_list, r_square_list, '-o', label="r-squared value",)
    plt.title("Relationship of connectivity with accuracy")
    plt.xlabel("Connectivity")
    plt.plot(connectivity_list, nrmse_list, '-o', label="nrmse value")
    plt.legend()
    plt.show()


def input_connectivity(X_train, y_train, X_test, y_test):
    input_connectivity_list = []
    r_square_list = []
    nrmse_list = []
    for i in np.arange(0.1, 1, 0.05):
    #for i in range(10, 100, 10):
        units = 10
        leak_rate = 0.3
        spectral_radius = 0.99
        input_scaling = 1.0
        connectivity = 0.1      # - density of reservoir internal matrix
        input_connectivity = i  # and of reservoir input matrix
        regularization = 1e-4
        seed = 1234             # for reproducibility

        reservoir = Reservoir(units, sr=spectral_radius,
                            lr=leak_rate, rc_connectivity=connectivity,
                            input_connectivity=input_connectivity, seed=seed, input_scaling=input_scaling)

        readout = Ridge(1, ridge=regularization)
        esn = reservoir >> readout
        esn = esn.fit(X_train, y_train)
        reservoir_predictions = esn.run(X_test)
        input_connectivity_list.append(i)
        r_square = rsquare(y_test, reservoir_predictions)
        r_square_list.append(r_square)
        nrmse_measured = nrmse(y_test, reservoir_predictions)
        nrmse_list.append(nrmse_measured)


    plt.plot(input_connectivity_list, r_square_list, '-o', label="r-squared value",)
    plt.title("Relationship of input connectivity with accuracy")
    plt.xlabel("Input connectivity")
    plt.plot(input_connectivity_list, nrmse_list, '-o', label="nrmse value")
    plt.legend()
    plt.show()


def multiple(X_train, y_train, X_test, y_test):
    units_list = []
    leak_rate_list = []
    r_square_list = []
    nrmse_list = []
    #list_variables = []
    #df = pd.DataFrame(columns=['units', 'leak_rate', 'spectral_radius', 'input_scaling', 'connectivity', 'input_connectivity', 'r_sqare', 'nrmse'])
    for i in range(10, 100, 10):
        for j in np.arange(0.1, 1, 0.1):
            units = i
            leak_rate = j
            spectral_radius = 0.99
            input_scaling = 1.0
            connectivity = 0.1      # - density of reservoir internal matrix
            input_connectivity = 0.2  # and of reservoir input matrix
            regularization = 1e-4
            seed = 1234             # for reproducibility

            reservoir = Reservoir(units, sr=spectral_radius,
                                lr=leak_rate, rc_connectivity=connectivity,
                                input_connectivity=input_connectivity, seed=seed, input_scaling=input_scaling)

            readout = Ridge(1, ridge=regularization)
            esn = reservoir >> readout
            esn = esn.fit(X_train, y_train)
            reservoir_predictions = esn.run(X_test)
            units_list.append(i)
            leak_rate_list.append(j)
            r_square = rsquare(y_test, reservoir_predictions)
            r_square_list.append(r_square)
            nrmse_measured = nrmse(y_test, reservoir_predictions)
    #nrmse_list.append(nrmse_measured)
    #list_of_tuples.append(units, leak_rate, spectral_radius, input_scaling, connectivity, input_connectivity, r_square, nrmse_measured)
    #list_variables.append(units)
    #list_variables.append(leak_rate)
    #list_variables.append(spectral_radius)
    #list_variables.append(input_scaling)
    #list_variables.append(connectivity)
    #list_variables.append(input_connectivity)
    #list_variables.append(r_square)
    #list_variables.append(nrmse_measured)
    #variables_df = pd.DataFrame(list_variables, columns=['units', 'leak_rate', 'spectral_radius', 'input_scaling', 'connectivity', 'input_connectivity', 'r_sqare', 'nrmse'])
    #print(variables_df)

    plt.plot(units_list, r_square_list, '-o', label="r-squared value",)
    plt.title("Relationship of number of units neurons with accuracy")
    plt.xlabel("Number of units")
    plt.ylabel("r-squared value")
    plt.show()
    plt.plot(units_list, nrmse_list, '-o', label="nrmse value")
    plt.title("Relationship of number of units neurons with accuracy")
    plt.xlabel("Number of units")
    plt.ylabel("Normalised root mean square")
    plt.legend()
    plt.show()

def new(X_train, y_train, X_test, y_test):
    units_list = []
    leak_rate_list = []
    r_square_list = []
    nrmse_list = []
    spectral_radius_list = []
    input_scaling_list = []
    connectivity_list = []
    input_connectivity_list = []

    for i in range(10, 100, 10):
        for j in np.arange(0.1, 1, 0.1):
            #for k in np.arange(0.1, 1, 0.1):
                #for l in np.arange(0.1, 1, 0.1):
                    #for m in np.arange(0.1, 1, 0.1):
                        #for n in np.arange(0.1, 1, 0.1):
            units = i
            leak_rate = j
            spectral_radius = 0.4
            input_scaling = 0.2
            connectivity = 0.3     # - density of reservoir internal matrix
            input_connectivity = 0.3  # and of reservoir input matrix
            regularization = 1e-4
            seed = 1234             # for reproducibility

            reservoir = Reservoir(units, sr=spectral_radius,
                                lr=leak_rate, rc_connectivity=connectivity,
                                input_connectivity=input_connectivity, seed=seed, input_scaling=input_scaling)

            readout = Ridge(1, ridge=regularization)
            esn = reservoir >> readout
            esn = esn.fit(X_train, y_train)
            reservoir_predictions = esn.run(X_test)
            units_list.append(i)
            leak_rate_list.append(leak_rate)
            spectral_radius_list.append(spectral_radius)
            input_scaling_list.append(input_scaling)
            connectivity_list.append(connectivity)
            input_connectivity_list.append(input_connectivity)
            r_square = rsquare(y_test, reservoir_predictions)
            r_square_list.append(r_square)
            nrmse_measured = nrmse(y_test, reservoir_predictions)
            nrmse_list.append(nrmse_measured)
        
    variables_df = pd.DataFrame(list(zip(units_list, leak_rate_list, spectral_radius_list, input_scaling_list, connectivity_list, input_connectivity_list, r_square_list, nrmse_list )), columns=['units', 'leak_rate', 'spectral_radius', 'input_scaling', 'connectivity', 'input_connectivity', 'r_sqare', 'nrmse'])
    variables_df.to_csv('hyperparameter.csv')

def main():
    timesteps = 2510
    tau = 17
    X = mackey_glass(timesteps, tau=tau)

    # rescale between -1 and 1
    X = 2 * (X - X.min()) / (X.max() - X.min()) - 1


    x, y = to_forecasting(X, forecast=100)
    X_train, y_train = x[:2000], y[:2000]
    X_test, y_test = x[2000:], y[2000:]
    y_pred, y_test =  reservoir(X_train, y_train, X_test, y_test)
    multiple(X_train, y_train, X_test, y_test)
  

if __name__ == "__main__":
    main()