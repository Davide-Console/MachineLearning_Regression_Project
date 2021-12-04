
from Data.data_preparation import categorical_to_dummy, paired_plot, load_dataset

complete_dataframe = load_dataset('model.csv')

print(complete_dataframe.dtypes)
print(complete_dataframe.isnull().sum())

dataframe = categorical_to_dummy(complete_dataframe, variables2convert=['product_category', 'day'], verbose=True)

# paired_plot(dataframe, 'likes')

correlation = dataframe.corr()
correlation.to_csv('correlation.csv')

# Categorical Dataframe
df_categorical = complete_dataframe[['product_category', 'day', 'likes']]
df_categorical = categorical_to_dummy(df_categorical, variables2convert=['product_category', 'day'], verbose=True)

# Numerical Dataframe
df_numerical = complete_dataframe.loc[ : , complete_dataframe.columns != 'product_category']
df_numerical = df_numerical.loc[ : , df_numerical.columns != 'day']