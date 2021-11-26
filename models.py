# LINEAR REGRESSION
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor


# Create linear regression object
def get_SimpleRegression(): #OK

    model = LinearRegression()
    parameters = {'normalize': [True, False]}

    return model, parameters



def get_RidgeRegression(): #OK
    model = Ridge()
    parameters = {"alpha": [0.001, 0.01, 0.1, 1, 10],
                  "normalize": [True, False]}

    return model, parameters

def get_LassoRegression(): #OK
    model = Lasso()
    parameters = {"alpha": [0.001, 0.01, 0.1, 1, 10],
                  "normalize": [True, False]}

    return model, parameters


def get_KNR(): #OK
    model = KNeighborsRegressor()
    parameters = {'n_neighbors': np.arange(5, 10, 5),
                  'p': [1, 2, 3],
                  'n_jobs': [-1]}

    return model, parameters

def get_TreeRegressor(): #OK
    model = DecisionTreeRegressor()
    parameters = {"max_depth": [3, 5, 7],
                  "min_samples_leaf": [10, 20, 30]}

    return model, parameters

def get_SVR():
    model = SVR()
    parameters = {'C': [100],
                  'epsilon': [0.01],
                  'gamma': ['auto'],
                  'kernel': ['linear', 'poly', 'rbf'],
                  'degree': [2, 3, 5]}

    return model, parameters

def get_MultyLayerPerceptronRegressor(): #OK
    model = MLPRegressor(random_state=0)
    parameters = {'hidden_layer_sizes': [(8), (10, 5), (20, 10, 5), (10, 5, 3)],
                  'solver': ['sgd'],
                  'batch_size': [20],
                  'learning_rate': ['constant'],
                  'alpha': 10.0 ** -np.arange(-1, 3),
                  'max_iter': [10000]}
    
    return model, parameters


def get_RandomForestRegressor(): #OK
    model = RandomForestRegressor()
    parameters = {"n_estimators":[5,10,100,200], "criterion": ['mse'],
                  "min_samples_leaf": [0.03,0.05,0.1,0.3], "random_state" : [42]}

    return model, parameters


def get_Adaboost(): #OK
    model = AdaBoostRegressor()
    parameters = {"n_estimators": [10, 50],
                  "base_estimator": [SVR(kernel='linear'), DecisionTreeRegressor(max_depth=3)],
                  "learning_rate": [0.5, 1.0, 1.1],
                  "random_state": [4]}

    return model, parameters

def get_GradientBoostingRegressor():
    model = GradientBoostingRegressor()  # base_estimator=DecisionTreeRegressor(max_depth=3)
    parameters = {"n_estimators": [20, 50, 70, 100, 200], "learning_rate": [0.1, 0.5, 1, 2],
                  "random_state": [0],
                  "max_depth": [1, 2]}

    return model, parameters