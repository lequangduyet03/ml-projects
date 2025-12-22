import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

data = pd.read_csv("Time-series-datasets\co2.csv")
data["time"] = pd.to_datetime(data["time"], yearfirst=True)
data["co2"] = data["co2"].interpolate()
# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# plt.show()
window_size = 5
train_ratio = 0.8
i = 1
while i < window_size:
    data["co2_{}".format(i)] = data["co2"].shift(-i)
    i = i + 1
data["target"] = data["co2"].shift(-i)
data= data.dropna(axis=0)
# print(data.drop("time",axis=1).corr())

x = data.drop(["time","target"], axis=1)
y = data["target"]
num_samples = len(x)
x_train = x[:int(train_ratio*num_samples)]
y_train = y[:int(train_ratio*num_samples)]
x_test = x[int(train_ratio*num_samples):]
y_test = y[int(train_ratio*num_samples):]

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
current_data = [368.1, 368.4, 369, 369.3, 369.5]
for i in range(10):
    print(current_data)
    prediction = model.predict([current_data]).tolist()
    print("Co2 in week {} is {}".format(i+1,prediction[0]))
    print("---------------------------------------")
    current_data = current_data[1:] + prediction



print("MAE: {}".format(mean_absolute_error(y_test,y_pred)))
print("MSE: {}".format(mean_squared_error(y_test,y_pred)))
print("R2: {}".format(r2_score(y_test,y_pred)))

# LinearRegression
# MAE: 0.3605603788359235
# MSE: 0.2204494736034648
# R2: 0.9907505918201437

fig, ax = plt.subplots()
ax.plot(data["time"][:int(train_ratio*num_samples)], data["co2"][:int(train_ratio*num_samples)], label = "train")
ax.plot(data["time"][int(train_ratio*num_samples):], data["co2"][int(train_ratio*num_samples):], label = "test")
ax.plot(data["time"][int(train_ratio*num_samples):], y_pred, label = "prediction")

ax.set_xlabel("Time")
ax.set_ylabel("CO2")
ax.legend()
ax.grid()
plt.show()