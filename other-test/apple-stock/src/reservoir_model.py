import pandas as pd
from reservoirpy.nodes import Reservoir, Ridge
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier

""" def reservoir(X,y,test_size):
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    units = 500
    leak_rate = 0.5
    spectral_radius = 0.5
    connectivity = 0.1
    input_connectivity = 0.1
    regularization = 1e-5
    seed = 1234

    reservoir = Reservoir(units, sr=spectral_radius,
                        lr=leak_rate, rc_connectivity=connectivity,
                        input_connectivity=input_connectivity, seed=seed)

    readout = Ridge(ridge=regularization)
    esn = reservoir >> readout
    reservoir <<= readout
    esn = esn.fit(X_train, y_train)
    reservoir_predictions = esn.run(X_test) """

def data_processing(df):
    df = df.replace({'\$':''}, regex = True)
    df = df.astype({"Close": float})
    df = df.astype({"Volume": float})
    df = df.astype({"Open": float})
    df = df.astype({"High": float})
    df = df.astype({"Low": float})
    df["Date"] = pd.to_datetime(df.Date, format="%m/%d/%Y")
    df.sort_values(by='Date', inplace=True)
    df.index = df['Date']
    df.drop("Date", axis=1, inplace=True)
    return df

data = pd.read_csv('apple.csv')
df = data_processing(data)
df.dropna()
predictors = "Close"
columns = list(df.columns.values)
target = columns
target.remove(predictors)

X = df.drop(predictors, axis=1)
y = df.drop(target, axis=1)


""" data = pd.read_csv("london_weather.csv").dropna().drop('date', axis=1)

predictors = "snow_depth"
columns = list(data.columns.values)
target = columns
target.remove(predictors)

X = data.drop(predictors, axis=1)
y = data.drop(target, axis=1) """

X = X.to_numpy()
y = y.to_numpy() 
#X = X[:1500]
#y = y[:1500]
print(X)

units = 50
leak_rate = 0.2
spectral_radius = 0.4
connectivity = 0.6
input_connectivity = 0.4
regularization = 1e-4
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
model = model.fit(X, y)
joblib.dump(model, 'apple-stock-predictor.joblib')
#model = joblib.load('apple-stock-predictor.joblib')
#predictions = model.run(X)
""" print("Mean squared error (Reservoir) = " , mean_squared_error(y, predictions))
plt.plot(y, label="Ground truth")
plt.plot(predictions, label="Predictions")
plt.ylabel("Close price ($)")
plt.legend()
plt.show() """