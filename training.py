# Packages
import pickle

import numpy as np

from math import ceil

import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, train_test_split

from model import *


def hyperp_search(classifier, parameters, dataframe_wo_outliers, dataframe_outliers, verbose=False):
    '''
    Function for hyperparameters search through Grid Search and for cross-validation of the model
    :param classifier: regression model to use or to test
    :param parameters: parameters of the chosen model
    :param dataframe_wo_outliers: the dataset with 'likes' values below the threshold
    :param dataframe_outliers: the dataset with 'likes' values above the threshold
    :param verbose: True to print information on the execution
    :return: best fitted model
    '''
    # Creating features and target array
    X = dataframe_wo_outliers.drop(['likes'], axis=1)
    y = dataframe_wo_outliers['likes']

    # Splitting data set in train set and test set
    test_size = 0.2964

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=313)

    # Computing the proper amount of 'likes' outliers to expect in the prediction test according to the chosen threshold
    n_outliers = ceil((len(X_test)*0.00511)/0.99489)

    dataframe_outliers = dataframe_outliers.sort_values(by='likes', ascending=False)

    # Creating dataframe with all the major outliers (Worst Case)
    dataframe_outliers1 = dataframe_outliers.iloc[0:n_outliers, :]
    X_test_wc = X_test.append(dataframe_outliers1.drop(['likes'], axis=1))
    y_test_wc = y_test.append(dataframe_outliers1['likes'])

    # Creating dataframe with randomly selected outliers (Regular Case)
    dataframe_outliers2 = dataframe_outliers.sample(n=n_outliers, random_state=313)
    X_test_rc = X_test.append(dataframe_outliers2.drop(['likes'], axis=1))
    y_test_rc = y_test.append(dataframe_outliers2['likes'])

    print(f'Train percentage: {round(((len(X_train)/(len(X_train)+len(X_test_rc)))*100), 2)}%\nTest percentage: {round(((len(X_test_rc)/(len(X_train)+len(X_test_rc)))*100), 2)}% (of whom {round(((n_outliers/len(X_test_rc))*100), 2)}% are outliers for "likes" attribute)\n')

    # Performing Cross-Validation for the chosen model
    Xt = X_train
    yt = y_train
    gs = GridSearchCV(classifier, parameters, cv=5, verbose=0, n_jobs=-1, scoring='neg_mean_absolute_error')
    gs = gs.fit(Xt, yt)

    best_model = gs.best_estimator_

    # Prediction with the worst case test set
    y_pred_wc = best_model.predict(X_test_wc)
    y_pred_train = best_model.predict(X_train)

    if verbose is True:
        print("Best score: %f using %s" % (gs.best_score_, gs.best_params_))
        gs.score(X_test_wc, y_test_wc)
        print("MAE train: ", metrics.mean_absolute_error(y_train, y_pred_train))
        print("MSE train: ", metrics.mean_squared_error(y_train, y_pred_train))
        print("RMSE train: ", np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))
        print("r2: ", np.sqrt(abs(metrics.r2_score(y_train, y_pred_train))))

        print('--WORST CASE--')
        print("MAE test: ", metrics.mean_absolute_error(y_test_wc, y_pred_wc))
        print("MSE test: ", metrics.mean_squared_error(y_test_wc, y_pred_wc))
        print("RMSE test: ", np.sqrt(metrics.mean_squared_error(y_test_wc, y_pred_wc)))
        print("r2: ", np.sqrt(abs(metrics.r2_score(y_test_wc, y_pred_wc))))

        # prediction with the random case test set
        y_pred_rc = best_model.predict(X_test_rc)

        if verbose is True:
            gs.score(X_test, y_test)
            print('--RANDOM CASE--')
            print("MAE test: ", metrics.mean_absolute_error(y_test_rc, y_pred_rc))
            print("MSE test: ", metrics.mean_squared_error(y_test_rc, y_pred_rc))
            print("RMSE test: ", np.sqrt(metrics.mean_squared_error(y_test_rc, y_pred_rc)))
            print("r2: ", np.sqrt(abs(metrics.r2_score(y_test_rc, y_pred_rc))))

    return best_model


def test_model(dataframe_wo_outliers, dataframe_outliers, scaler):
    '''
    Function to test and save scaler and model
    :param dataframe_wo_outliers: the dataset with 'likes' values below the threshold
    :param dataframe_outliers: the dataset with 'likes' values above the threshold
    :param scaler: fitted scaler
    '''
    model, params = get_SVR()
    hyperp_search(model, params, dataframe_wo_outliers, dataframe_outliers, True)

    pickle.dump(scaler, open("scaler.pkl", "wb"))
    pickle.dump(model, open("model.pkl", "wb"))
    print("\nscaler and model saved")
