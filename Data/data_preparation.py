import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



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