import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv("Time-series-datasets\co2.csv")
data["time"] = pd.to_datetime(data["time"], yearfirst=True)
data["co2"] = data["co2"].interpolate()
# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# plt.show()
window_size = 5
target_size = 3
train_ratio = 0.8
i = 1
while i < window_size:
    data["co2_{}".format(i)] = data["co2"].shift(-i)
    i += 1
i = 0
while i < target_size:
    data["target_{}".format(i)] = data["co2"].shift(-i-window_size)
    i += 1
data = data.dropna(axis=0)
# print(data.drop("time", axis=1).corr())
targets = ["target_{}".format(i) for i in range(target_size)]
x = data.drop(["time"] + targets, axis=1)
y = data[targets]
num_samples = len(x)
x_train = x[:int(train_ratio*num_samples)]
y_train = y[:int(train_ratio*num_samples)]
x_test = x[int(train_ratio*num_samples):]
y_test = y[int(train_ratio*num_samples):]

models = [LinearRegression() for _ in range(target_size)]
for i, model in enumerate(models):
    model.fit(x_train, y_train["target_{}".format(i)])
    y_predict = model.predict(x_test)
    print("MODEL {}:".format(i+1))
    print("MAE: {}".format(mean_absolute_error(y_test["target_{}".format(i)], y_predict)))
    print("MSE: {}".format(mean_squared_error(y_test["target_{}".format(i)], y_predict)))
    print("R2: {}".format(r2_score(y_test["target_{}".format(i)], y_predict)))
    print("------------------------")

