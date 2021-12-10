# Packages
from training import *

from Data.data_preparation import *


def main():
    # Loading the dataset
    dataframe = load_dataset('model.csv')

    # Preparing the dataset
    # according to the dataframe description we set 'global subjectivity' in range [0, 1]
    # and all the 'polarity' attributes in the range [-1, 1]
    dataframe = correct_dataset(dataframe)

    # Converting categorical features to dummy
    dataframe = categorical_to_dummy(dataframe, variables2convert=['product_category', 'day'], verbose=False)

    # Setting all outliers to a value 10% outside the whiskers
    dataframe = outlier2whisker(dataframe, features_o2w)

    # StandardScaler() only to numerical variables
    dataframe, scaler = standardise(dataframe, features_s, verbose=False)

    # Splitting dataframe according to a given threshold
    threshold = 10000
    dataframe_wo_outliers = dataframe[dataframe['likes'] <= threshold]
    dataframe_outliers = dataframe[dataframe['likes'] > threshold]

    # N. B. Dataframe_wo_outliers and dataframe_outliers do not have outliers for all numerical attribute.
    #       Here the term 'outlier' refers to the outliers in the target category 'likes'.

    # Model training & Grid Search
    test_model(dataframe_wo_outliers, dataframe_outliers, scaler)


if __name__ == '__main__':
    main()
