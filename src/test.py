import pandas


filename='airsim.csv'

file = pandas.read_csv(filename)
print(file.iloc[-1,0] is "safe")
# print(file['yaw_rate'][:-1])