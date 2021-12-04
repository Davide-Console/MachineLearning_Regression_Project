import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_dataset(filepath):
    df = pd.read_csv(filepath, index_col=False)
    return df


def categorical_to_dummy(dataframe, variables2convert, verbose=False):
    for variable in variables2convert:
        dummy = pd.get_dummies(dataframe[variable], drop_first=True)
        dataframe = dataframe.drop([variable], axis=1, inplace=False)
        dataframe = pd.concat([dataframe, dummy], axis=1)

    if verbose is True:
        print(dataframe.head(5))

    return dataframe


def paired_plot(dataframe, target):
    sns.pairplot(dataframe, hue=target)
    plt.show()


def standardise(dataframe, features, verbose=False):
    """
    Applies the sklearn.preprocessing.StandardScaler to the features selected
    :param dataframe: the dataframe containing the variables to scale
    :param features: a list of all the attributes to be scaled
    :param verbose: True to print some information on the execution
    :return: the dataset with the converted attributes and the StandardScaler() fitted
    """
    scaler = StandardScaler()
    dataframe_stand = dataframe.copy()  # copy to keep the variables that should not be scaled
    scaler.fit(dataframe_stand.loc[:, :].astype(float))
    dataframe_stand = pd.DataFrame(scaler.transform(dataframe_stand.loc[:, :].astype(float)))
    dataframe_stand.columns = dataframe.columns

    dataframe[features] = dataframe_stand[features]

    if verbose is True:
        dataframe_stand.hist()
        plt.show()
        print(dataframe.head(5))

    return dataframe, scaler
