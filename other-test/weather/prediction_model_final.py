import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from reservoirpy.nodes import Reservoir, Ridge, Input
from reservoirpy.observables import mse, nrmse, rsquare
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

def ridge_regression(X, y, test_size):
    X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    #ridge = make_pipeline(StandardScaler(with_mean=False), R())
    ridge = Ridge(ridge=1e-5)
    ridge = ridge.fit(X_train, y_train)
    #ridge_predictions = ridge.predict(X_test)
    ridge_predictions = ridge.run(X_test)
    plt.title("Rainfall prediction using Ridge Regression")
    plt.plot(y_test, label="Ground truth")
    plt.plot(ridge_predictions, label="Predictions")
    plt.ylabel("Rainfall/mm")
    plt.legend()
    #plt.show()
    #ridge_predictions = scaler.inverse_transform(ridge_predictions)
    #points = list(range(1, len(ridge_predictions)+1))
    return nrmse(y_test, ridge_predictions), r2_score(y_test, ridge_predictions)
    """print("Mean squared error (Ridge Regression) = " , mse(y_test, ridge_predictions))
    plt.title("Snow depth prediction using Ridge Regression")
    plt.plot(y[:len(ridge_predictions)], label="Ground truth")
    plt.plot(ridge_predictions, label="Predictions")
    plt.ylabel("Snow depth/cm")
    plt.legend()
    plt.show() """

def decision_tree_regression(X,y,test_size):
    X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    decision_tree_model = DecisionTreeRegressor(max_depth=5).fit(X_train,y_train)
    decision_tree_predictions = decision_tree_model.predict(X_test)
    decision_tree_predictions = decision_tree_predictions.reshape(len(decision_tree_predictions), 1)
    plt.title("Snow depth prediction using Decision Tree Regression")
    plt.plot(y_test, label="Ground truth")
    plt.plot(decision_tree_predictions, label="Predictions")
    plt.ylabel("Rainfall/mm")
    plt.legend()
    #plt.show()
    return nrmse(y_test, decision_tree_predictions), r2_score(y_test, decision_tree_predictions)
    """print("Mean squared error (Decision Tree Regression) = " , mean_squared_error(y_test, decision_tree_predict))
    plt.title("Snow depth prediction using Decision Tree Regression")
    plt.plot(y[:len(decision_tree_predict)], label="Ground truth")
    plt.plot(decision_tree_predict, label="Predictions")
    plt.ylabel("Snow depth/cm")
    plt.legend()
    plt.show() """

def KNN(X,y,test_size):
    X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    KNN_model = KNeighborsRegressor(n_neighbors=3).fit(X_train,y_train)
    KNN_predict = KNN_model.predict(X_test)
    plt.title("Rainfall prediction using KNN")
    plt.plot(y_test, label="Ground truth")
    plt.plot(KNN_predict, label="Predictions")
    plt.ylabel("Rainfall/mm")
    plt.legend()
    #plt.show()
    return nrmse(y_test, KNN_predict), r2_score(y_test, KNN_predict)
    """print("Mean squared error (KNN) = " , mse(y_test, KNN_predict))
    plt.title("Snow depth prediction using KNN")
    plt.plot(y[:len(KNN_predict)], label="Ground truth")
    plt.plot(KNN_predict, label="Predictions")
    plt.ylabel("Snow depth/cm")
    plt.legend()
    plt.show() """

def reservoir(X,y,test_size):
    #X = X.to_numpy()
    #y = y.to_numpy()
    X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    # Create a reservoir
    #reservoir = Reservoir(100, lr=0.5, sr=0.9)

    # Readout parameters
    #regularization_coef = 1e-8
    #warmup = 100

    # create a readout layer equiped with an offline learning rule
    #readout = Ridge(ridge=regularization_coef, name="readout")
    #esn = reservoir >> readout
    #reservoir.reset()
    units = 30
    leak_rate = 0.3
    spectral_radius = 1.0
    connectivity = 1.0
    input_connectivity = 0.7
    regularization = 1e-5
    seed = 1234


    reservoir = Reservoir(units, 
                          sr=spectral_radius,
                          lr=leak_rate, 
                          rc_connectivity=connectivity,
                          input_connectivity=input_connectivity, 
                          seed=seed)

    #reservoir = Reservoir(50, lr=0.5, sr=0.9)
    readout = Ridge(ridge=regularization)
    reservoir <<= readout
    esn = reservoir >> readout
    esn = esn.fit(X_train, y_train, force_teachers=True)
    reservoir_predictions = esn.run(X_test)
    #reservoir_predictions = scaler.inverse_transform(reservoir_predictions)
    plt.title("Rainfall prediction using Reservoir")
    plt.plot(y_test, label="Ground truth", )
    plt.plot(reservoir_predictions, label="Predictions")
    plt.ylabel("Rainfall/mm")
    plt.legend()
    #plt.show()
    return nrmse(y_test, reservoir_predictions), r2_score(y_test, reservoir_predictions)
    """ print("Mean squared error (Reservoir) = " , mse(y_test, reservoir_predictions))
    plt.title("Snow depth prediction using Reservoir")
    plt.plot(y_test, label="Ground truth", )
    plt.plot(reservoir_predictions, label="Predictions")
    plt.ylabel("Snow depth/cm")
    plt.legend()
    plt.show() """

def reservoir_parameter(X, y, test_size):
    X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    units_list = []
    leak_rate_list = []
    spectral_radius_list = []
    mse_list = []
    r_square_list = []
    nrmse_list = []
    list_of_tuples = []

    for i in  range(10, 110, 10): #np.arange(0.1, 1.1, 0.1):
    #for i in np.arange(0.1, 1.1, 0.1): 
        for j in np.arange(0.1, 1.1, 0.1):
            for k in np.arange(0.1, 1.1, 0.1):
                units = i
                leak_rate = j
                spectral_radius = k
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
                model.fit(X_train, y_train)
                predictions = model.run(X_test)

                units_list.append(i)
                leak_rate_list.append(j)
                spectral_radius_list.append(k)
                nrmse_list.append(nrmse(y_test, predictions))
                r_square_list.append(r2_score(y_test, predictions))
    
    list_of_tuples = list(zip(units_list, leak_rate_list, spectral_radius_list, nrmse_list, r_square_list))
    df = pd.DataFrame(list_of_tuples, columns=['units', 'leak_rate', 'spectral_radius', 'nrmse', 'r_squared'])
    #list_of_tuples = list(zip(units_list, leak_rate_list, spectral_radius_list, mse_list))
    #df = pd.DataFrame(list_of_tuples, columns=['units', 'leak_rate', 'spectral_radius', 'mse'])
    min = df['nrmse'].min()
    print(df.loc[df['nrmse'] == min])
    max = df['r_squared'].max()
    print(df.loc[df['r_squared'] == max])
    

def main():
    data = pd.read_csv("weather.csv").dropna() #.drop('date', axis=1)
    #scaled_data = scaler.fit_transform(data)
    print(data.columns.values)
    predictors = "Rainfall"
    columns = list(data.columns.values)
    target = columns
    target.remove(predictors)

    X = data.drop(predictors, axis=1)
    y = data.drop(target, axis=1)

    scaled_X = scaler.fit_transform(X)
    scaled_y = scaler.fit_transform(y)

    test_size = 0.3
    X = X.to_numpy()
    y = y.to_numpy()

    reservoir_parameter(X, y, test_size)
    """ ridge_nrmse, r_square_ridge = ridge_regression(X, y, test_size)
    print('Ridge (NRMSE) =', + ridge_nrmse)
    print('Ridge (r2) =', + r_square_ridge)
    reservoir_nrmse, r_square_reservoir = reservoir(X, y, test_size)
    print('Reservoir (NRMSE)=', +reservoir_nrmse)
    print('Reservoir (r2) =', + r_square_reservoir)
    dt_nrmse, r_square_dt = decision_tree_regression(X, y, test_size)
    print('DT (NRMSE)=', + dt_nrmse)
    print('DT (r2) =', + r_square_dt)
    knn_nrmse, r_square_knn = KNN(X, y, test_size)
    print('KNN (NRMSE)=', + knn_nrmse)
    print('KNN (r2) =', + r_square_knn) """

if __name__ == "__main__":
    main()