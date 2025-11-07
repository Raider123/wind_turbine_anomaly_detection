import numpy as np
import pandas as pd

# Reading Data
train_set = pd.read_csv('data/Train.csv')
test_set = pd.read_csv('data/Test.csv')

#Overview on the data
print("TRAIN DATA #########################")
print(train_set.info())
print("TEST DATA #########################")
print(test_set.info())

# # Access to columns + Conversion to Np-Array

# power_data_train = np.array(train_set['Power'])
# wind_data_train = np.array(train_set['WS_100m'])

# print("Power Data Shape: ", power_data_train.shape)
# print("Wind Data Shape: ", wind_data_train.shape)


# Number of entries with missing data -> output is 0
print("Training data entries with missing data:", len(train_set) - len(train_set.dropna()))
print("Test data entries with missing data:", len(test_set) - len(test_set.dropna()))