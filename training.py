# Packages
import matplotlib.pyplot as plt

import pickle

import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, train_test_split

from models import *


def hyperp_search(classifier, parameters, dataframe, verbose=False):
    #---SOLO PER COMPILARE COL DATASET DI PROVA------------------------
    X, y = dataframe.data, dataframe.target
    X = X[y < 50]
    y = y[y < 50]
    #------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=313)

    gs = GridSearchCV(classifier, parameters, cv=5, verbose=10, n_jobs=-1)
    gs = gs.fit(X_train, y_train)

    best_model = gs.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)

    if verbose is True:
        print("Best score: %f using %s" % (gs.best_score_, gs.best_params_))
        gs.score(X_test, y_test)

        print("MAE train: ", metrics.mean_absolute_error(y_train, y_pred_train))
        print("MSE train: ", metrics.mean_squared_error(y_train, y_pred_train))
        print("RMSE train: ", np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))
        print("r2: ", np.sqrt(metrics.r2_score(y_train, y_pred_train)))
        print("MAE test: ", metrics.mean_absolute_error(y_test, y_pred))
        print("MSE test: ", metrics.mean_squared_error(y_test, y_pred))
        print("RMSE test: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print("r2: ", np.sqrt(metrics.r2_score(y_test, y_pred)))

        plt.scatter(y_pred_train, y_pred_train - y_train, c="b", label="training data")
        plt.scatter(y_pred, y_pred - y_test, c="g", label="hold out data")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.legend(loc="upper left")
        plt.hlines(y=0, xmin=-10, xmax=50, color="r")
        plt.xlim([-10, 50])
        plt.show()

    return best_model


def test_model(dataframe, scaler):
    model, params = get_KNR()
    hyperp_search(model, params, dataframe, True)
    Question = input("Do you want to save scaler and model?   <y/n>")
    if Question == ("y"):
        pickle.dump(scaler, open("scaler.pkl", "wb"))
        pickle.dump(model, open("model.pkl", "wb"))
        print("scaler and model saved")
    # return results