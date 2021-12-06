import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


def apply_pca(dataframe, verbose=False):
    pca = PCA()
    pca.fit(dataframe)

    if verbose is True:
        explained_var = pd.DataFrame(pca.explained_variance_ratio_).transpose()
        print(explained_var)
        sns.barplot(data=explained_var)
        plt.show()
        cum_explained_var = np.cumsum(pca.explained_variance_ratio_)
        pd.DataFrame(cum_explained_var).transpose()
        print(cum_explained_var[-1])
        plt.plot(cum_explained_var)
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        plt.show()

    return pca, pca.transform(dataframe)




features_2_standardise = ['age_days',
                          'n_tokens_title',
                          'n_tokens_review',
                          'n_unique_tokens',
                          'n_non_stop_words',
                          'n_non_stop_unique_tokens',
                          'num_hrefs',
                          'num_self_hrefs',
                          'num_imgs',
                          'num_videos',
                          'average_token_length',
                          'num_keywords',
                          'self_reference_min_shares',
                          'self_reference_max_shares',
                          'self_reference_avg_sharess',
                          'topic_quality',
                          'topic_shipping',
                          'topic_packaging',
                          'topic_description',
                          'topic_others',
                          'global_subjectivity',
                          'global_sentiment_polarity',
                          'global_rate_positive_words',
                          'global_rate_negative_words',
                          'rate_positive_words',
                          'rate_negative_words',
                          'avg_positive_polarity',
                          'min_positive_polarity',
                          'max_positive_polarity',
                          'avg_negative_polarity',
                          'min_negative_polarity',
                          'max_negative_polarity',
                          'title_subjectivity',
                          'title_sentiment_polarity',
                          'abs_title_subjectivity',
                          'abs_title_sentiment_polarity'
                          ]
