import pandas as pd
import numpy as np

# This section is for handling NaNs in the data

def preprocess_dataset(dataset: pd.DataFrame):

    ## Handle missing data -> data is complete, so no further process needed
    count_nans = len(dataset) - len(dataset.dropna())
    print("Number of NaNs in this dataset:", count_nans)

    ##### Feature selection/engineering #####
    # Delete the Unnamed: 0 column
    dataset = dataset.drop("Unnamed: 0", axis=1)
    print(dataset.columns.tolist())

    # Since we have a WD in degrees, we have the issue that 359 and 1 degree are far apart 
    # We can use sin and cos -> This way we will have similar sin and cos results for 359 and 1 degree, which will improve the training
    # This means: For each WD column, we will replace it with two columns (sin and cos equivalents to represent the WD
    dataset["Sin_WD_10m"] = np.sin(dataset["WD_10m"])
    dataset["Cos_WD_10m"] = np.cos(dataset["WD_10m"])
    dataset["Sin_WD_100m"] = np.sin(dataset["WD_100m"])
    dataset["Cos_WD_100m"] = np.cos(dataset["WD_100m"])
    dataset = dataset.drop(columns = ["WD_10m", "WD_100m"])

    # Now we can also make use of sin and cos to transfor the Time Objects into sin cos features
    # Reason 1: days, months, hours and mins are all cyclical -> Hour 23 and Hour 1 are closer together as sin cos equivalents (like for wind directions)
    # Reason 2: Doing the same with one hot encoding would create date with high dimensions -> the approach with sin and cos does not
    dataset["Time"] = pd.to_datetime(dataset["Time"], format="%d-%m-%Y %H:%M")
    hour = dataset["Time"].dt.hour
    # min = dataset["Time"].dt.minute
    day = dataset["Time"].dt.day
    month = dataset["Time"].dt.month
    year = dataset["Time"].dt.year
    dataset["Sin_hour"] = np.sin(2 * np.pi * hour/24)
    dataset["Cos_hour"] = np.cos(2 * np.pi * hour/24)
    # dataset["Sin_min"] = np.sin(2 * np.pi * min/60)
    # dataset["Cos_min"] = np.cos(2 * np.pi * min/60)
    dataset["Sin_day"] = np.sin(2 * np.pi * day / dataset["Time"].dt.days_in_month)
    dataset["Cos_day"] = np.cos(2 * np.pi * day / dataset["Time"].dt.days_in_month)
    dataset["Sin_month"] = np.sin(2 * np.pi * month / 12)
    dataset["Cos_month"] = np.cos(2 * np.pi * month / 12)
    dataset["Year"] = year
    # Uncomment before training
    # dataset = dataset.drop(columns=["Time"]) 
    print(dataset[["Time","Sin_hour", "Cos_hour", "Sin_min", "Cos_min", "Sin_day", "Cos_day", "Sin_month", "Cos_month", "Year"]])

    # Location one hot encoding

    # maybe temp x relhum

    # maybe windspeed x winddirection

    ##### Normalize/Scale #####

    ##### Build sequences #####

    ##### Handle outliers #####



train_set = pd.read_csv(r"D:\Coding Projects\Machine Learning Projects\wind_turbine_anomaly_detection\data\Train.csv")
test_set = pd.read_csv(r"D:\Coding Projects\Machine Learning Projects\wind_turbine_anomaly_detection\data\Test.csv")

preprocess_dataset(train_set)