import pandas as pd
from reservoirpy.nodes import Reservoir, Ridge
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
import numpy.ma as ma

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
    df.drop("Volume", axis=1, inplace=True)
    df.drop("Open", axis=1, inplace=True)
    df.drop("High", axis=1, inplace=True)
    df.drop("Low", axis=1, inplace=True)
    return df

def model(X, y):
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

data = pd.read_csv('apple-stock/data/apple.csv')
df = data_processing(data)
df.dropna()
predictors = "Close"
columns = list(df.columns.values)
target = columns
target.remove(predictors)

df['close_t1'] = df['Close'].shift(1)
df['close_t2'] = df['Close'].shift(2)
df['close_t3'] = df['Close'].shift(3)
df = df.fillna(0)

predictors = "Close"
columns = list(df.columns.values)
target = columns
target.remove(predictors)

X_df = df.drop(predictors, axis=1)
y_df = df.drop(target, axis=1)
#y_date = y.index + pd.Timedelta(days=1)
#y.loc[len(y.index)] = 109
#y.append(pd.DataFrame(190, index=[y_date]), columns=y.columns)
#y.loc[y_date, :] = [25000]
#print(y)

X = X_df.to_numpy()
y = y_df.to_numpy()
print(y)
units = 100
leak_rate = 0.9
spectral_radius = 0.9
connectivity = 0.9
input_connectivity = 0.9
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

X = X_df.iloc[-1]
X = X.to_numpy()
#print(df)
predictions = model.run(X)
#predictions = predictions.astype(float)
predictions = ma.masked_array(predictions, mask =[False]).__float__()
#print(predictions)
new_row = {'Close' : predictions,}
new_df = df.append(new_row, ignore_index=True)
print(new_df)
df.plot(use_index=True, y="Close")
#plt.show()