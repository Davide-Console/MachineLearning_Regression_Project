from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.cluster import KMeans, Birch, DBSCAN
import seaborn as sns
from Data.data_preparation import categorical_to_dummy, paired_plot, load_dataset, standardise
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import numpy as np
import os
import pandas as pd
from training import test_model


def boxplot(data, cluster):
    os.makedirs('boxplots' + str(cluster), exist_ok=True)
    for column in data:
        # print(df_numerical[column].head(2))
        attribute = column
        # bold_attribute = "\033[1m" + attribute + "\033[0m"
        print('\n\n' + attribute)

        fig, ax = plt.subplots()
        sns.boxplot(data=data, x=column, ax=ax)
        # plt.show()

        filename = 'boxplots' + str(cluster) + '/' + column + '.png'
        plt.savefig(filename)
        plt.close(fig)

        outliers = [y for stat in boxplot_stats(data[column]) for y in stat['fliers']]
        print('#outliers: ', len(outliers))
        stat = boxplot_stats(data[column])
        whisk_lo = stat[0].get('whislo')
        whisk_hi = stat[0].get('whishi')
        print('lower whisker: ', whisk_lo)
        print('upper whisker: ', whisk_hi)
        if len(outliers) > 0:
            high_outliers = [x for x in outliers if x > whisk_hi]
            low_outliers = [x for x in outliers if x < whisk_lo]

            print('#Low outliers: ', len(low_outliers))
            print('#High outliers: ', len(high_outliers))
            print('mean: ', stat[0].get('mean'), '+-', np.std(data[column]))
            print('median: ', stat[0].get('med'), '+-', stat[0].get('iqr'))
            print('min value: ', min(outliers))
            print('max value: ', max(outliers))


complete_dataframe = load_dataset('model.csv')

dataframe = categorical_to_dummy(complete_dataframe, variables2convert=['product_category', 'day'], verbose=False)

dataframe, scaler = standardise(dataframe, features=['age_days',
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
                                                     ], verbose=False)

dataframe = dataframe.drop(['likes'], axis=1)

random_state = 123
n_clusters = 12
model = KMeans(n_clusters=n_clusters, random_state=random_state)
# model = Birch(n_clusters=n_clusters, random_state=random_state)
model.fit(dataframe)

all_predictions = model.predict(dataframe)

print(random_state)
dataframe['cluster'] = all_predictions

sum = 0

for cluster in np.unique(all_predictions):
    print('\ncluster: ', cluster)
    print(np.sum(all_predictions == cluster))
    if np.sum(all_predictions == cluster) > 1000:
        sum += np.sum(all_predictions == cluster)

print('\n\nThere are ', sum, ' samples in clusters with more than 1000 records')

dataframe['likes'] = complete_dataframe['likes']

outliers_cluster = []
cluster_df = []
for cluster in np.unique(all_predictions):

    if np.sum(all_predictions == cluster) > 1000:
        df = dataframe[dataframe['cluster'] == cluster]
        df.drop(['cluster'], axis=1)
        cluster_df.append(df)
    else:
        outliers_cluster.append(dataframe[dataframe['cluster'] == cluster])

outliers_df = pd.concat(outliers_cluster)

for data in cluster_df:
    print('\n\nCLUSTER: ', np.unique(data['cluster']), np.sum(all_predictions == np.unique(data['cluster'])[0]),
          ' samples')
    test_model(data.drop(['cluster'], axis=1), scaler)

print('\n\nCLUSTER OUTLIERS: ', outliers_df.shape[0], ' samples')
test_model(outliers_df.drop(['cluster'], axis=1), scaler)

print('\n\nOriginal Dataframe: ', )
test_model(dataframe.drop(['cluster'], axis=1), scaler)
'''
df_numerical = dataframe[['age_days',
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
                          'abs_title_sentiment_polarity', 'cluster']]

for cluster in np.unique(all_predictions):
    print('\n\n------------CLUSTER: ', cluster, '------------')
    print('\n Samples: ', np.sum(all_predictions == cluster))
    df = df_numerical.copy(deep=True)
    df = df[df['cluster'] == cluster]
    boxplot(df, cluster)
'''
