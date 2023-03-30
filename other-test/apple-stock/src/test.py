import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


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


data_before = pd.read_csv('apple.csv')
df_before = data_processing(data_before)
data = pd.read_csv('apple2023.csv')
df = data_processing(data)
predictors = "Close"
columns = list(df.columns.values)
target = columns
target.remove(predictors)

X = df.drop(predictors, axis=1)
y = df.drop(target, axis=1)
print(df_before)
X = X.to_numpy()
y = y.to_numpy()



""" test_size = 0.1
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
forecast = 2000
X =X [-forecast:] """

""" model = joblib.load('apple-stock-predictor.joblib')
predictions = model.run(X)
predictions_df = pd.DataFrame(predictions, columns = ['Predictions'], index=df.index)

new_df = pd.concat([df_before, df])
print("Mean squared error (Reservoir) = " , mean_squared_error(y, predictions))
plt.plot(new_df['Close'], label="Ground truth")
plt.plot(predictions_df['Predictions'], label="Predictions")
plt.legend()
plt.show() """