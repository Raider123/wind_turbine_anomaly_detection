import pandas as pd
import numpy as np

# This section is for handling NaNs in the data

def preprocess_dataset(dataset: pd.DataFrame):

    ## Handle missing data -> data is complete, so no further process needed
    count_nans = len(dataset) - len(dataset.dropna())
    print("Number of NaNs in this dataset:", count_nans)

    ## Feature selection
    # Delete the Unnamed: 0 column
    dataset = dataset.drop("Unnamed: 0", axis=1)
    print(dataset.columns.tolist())
    #
    print(len(dataset))
    print(dataset["WD_100m"])

    ## Normalize/Scale

    ## Build sequences

    ## Handle outliers



train_set = pd.read_csv(r"D:\Coding Projects\Machine Learning Projects\wind_turbine_anomaly_detection\data\Train.csv")
test_set = pd.read_csv(r"D:\Coding Projects\Machine Learning Projects\wind_turbine_anomaly_detection\data\Test.csv")

preprocess_dataset(train_set)