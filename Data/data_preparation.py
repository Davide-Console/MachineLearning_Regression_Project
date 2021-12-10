#Packages
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.cbook import boxplot_stats

from sklearn.preprocessing import StandardScaler


def load_dataset(filepath):
    '''
    Load the dataset given a csv file
    :param filepath: path to the file from the working directory
    :return: the dataset stored in a pandas.DataFrame
    '''
    df = pd.read_csv(filepath, index_col=False)
    return df


def correct_dataset(dataframe):
    '''
    Sets all the 'global subjectivity' values greater then 1 to 1 and lesser then 0 to 0
    Sets all the 'polarity' attributes' values greater then 1 to 1 and lesser then -1 to -1
    :param dataframe: the dataframe containing the values to be tranformed
    :return: a pandas.DataFrame with fixed values
    '''
    dataframe.loc[dataframe[' global_subjectivity'] < 0, ' global_subjectivity'] = 0
    dataframe.loc[dataframe[' global_subjectivity'] > 1, ' global_subjectivity'] = 1
    for column in dataframe.columns:
        if 'polarity' in column:
            dataframe.loc[dataframe[column] < -1, column] = -1
            dataframe.loc[dataframe[column] > 1, column] = 1

    return dataframe


def categorical_to_dummy(dataframe, variables2convert, verbose=False):
    '''
    Converts the selected attributes into dummy variables. Drops the last dummy variable for each attribute
    :param dataframe: the pandas.DataFrame with the attributes to convert
    :param variables2convert: a list containing the column names to convert
    :param verbose: True to print information on the execution
    :return: the dataset with the dummy variables converted
    '''
    for variable in variables2convert:
        dummy = pd.get_dummies(dataframe[variable], drop_first=True)
        dataframe = dataframe.drop([variable], axis=1, inplace=False)
        dataframe = pd.concat([dataframe, dummy], axis=1)

    if verbose is True:
        print(dataframe.head(5))

    return dataframe


def standardise(dataframe, features, verbose=False):
    """
    Applies the sklearn.preprocessing.StandardScaler to the features selected
    :param dataframe: the dataframe containing the variables to scale
    :param features: a list of all the attributes to be scaled
    :param verbose: True to print some information on the execution
    :return: a pandas.DataFrame with the converted attributes and the StandardScaler() fitted
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


def outlier2whisker(dataframe, features):
    '''
    Fix outliers above the high whisker and below the low whisker as follows:
    outlier high = whisker high + (whisker high / 10)
    outlier low = whisker low - (whisker low / 10)
    :param dataframe: the dataset containing the outliers to be fixed
    :param features: attributes of the dataframe which contains outliers
    :return: a pandas.DataFrame with fixed outliers
    '''
    df = pd.DataFrame(dataframe)
    for column in features:

        stat = boxplot_stats(dataframe[column])
        whisk_lo = stat[0].get('whislo')
        whisk_hi = stat[0].get('whishi')

        df[column] = dataframe[column].where(dataframe[column] >= whisk_lo, other=whisk_lo-abs(whisk_lo/10), inplace=False)
        df[column] = dataframe[column].where(dataframe[column] <= whisk_hi, other=whisk_hi+whisk_hi/10, inplace=False)

    return df


# variables2convert for function outlier2whisker
features_o2w = [' n_tokens_title', ' n_tokens_review', ' n_unique_tokens', ' n_non_stop_words',
                ' n_non_stop_unique_tokens', ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos',
                ' average_token_length', ' num_keywords', ' self_reference_min_shares', ' self_reference_max_shares',
                ' self_reference_avg_sharess', 'topic_quality', 'topic_shipping', 'topic_packaging', 'topic_description',
                ' global_subjectivity', ' global_sentiment_polarity', ' global_rate_positive_words',
                ' global_rate_negative_words', ' rate_positive_words', ' rate_negative_words', ' avg_positive_polarity',
                ' min_positive_polarity', ' max_positive_polarity', ' avg_negative_polarity', ' min_negative_polarity',
                ' max_negative_polarity', ' title_subjectivity', ' title_sentiment_polarity', ' abs_title_subjectivity',
                ' abs_title_sentiment_polarity']

# variables2convert for function standardise
features_s = ['age_days', ' n_tokens_title', ' n_tokens_review', ' n_unique_tokens', ' n_non_stop_words',
                ' n_non_stop_unique_tokens', ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos',
                ' average_token_length', ' num_keywords', ' self_reference_min_shares', ' self_reference_max_shares',
                ' self_reference_avg_sharess', 'topic_quality', 'topic_shipping', 'topic_packaging', 'topic_description',
                'topic_others', ' global_subjectivity', ' global_sentiment_polarity', ' global_rate_positive_words',
                ' global_rate_negative_words', ' rate_positive_words', ' rate_negative_words', ' avg_positive_polarity',
                ' min_positive_polarity', ' max_positive_polarity', ' avg_negative_polarity', ' min_negative_polarity',
                ' max_negative_polarity', ' title_subjectivity', ' title_sentiment_polarity', ' abs_title_subjectivity',
                ' abs_title_sentiment_polarity']
