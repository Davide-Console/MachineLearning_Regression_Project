import pandas as pd

from Data.data_preparation import load_dataset, categorical_to_dummy, standardise, features_2_standardise, apply_pca

complete_dataframe = load_dataset('model.csv')

dataframe = categorical_to_dummy(complete_dataframe, variables2convert=['product_category', 'day'], verbose=False)

dataframe, scaler = standardise(dataframe, features=features_2_standardise, verbose=False)

print(dataframe.columns)

dataframe = dataframe.drop(['likes'], axis=1)

print(dataframe.columns)

pca, dataframe_pca = apply_pca(dataframe, True)

principalDf = pd.DataFrame(data=dataframe_pca, columns=['pc0', 'pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10', 'pc11', 'pc12', 'pc13', 'pc14', 'pc15', 'pc16', 'pc17', 'pc18', 'pc19', 'pc20', 'pc21', 'pc22', 'pc23', 'pc24', 'pc25', 'pc26', 'pc27', 'pc28', 'pc29', 'pc30', 'pc31', 'pc32', 'pc33', 'pc34', 'pc35', 'pc36', 'pc37', 'pc38', 'pc39', 'pc40', 'pc41', 'pc42', 'pc43', 'pc44', 'pc45', 'pc46', 'pc47'])
print(principalDf.head())

principalDf['likes'] = complete_dataframe['likes']
