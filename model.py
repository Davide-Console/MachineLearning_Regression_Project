# Regression Model
from sklearn.svm import SVR

def get_SVR():
    '''
    Creates a Support Vector Regressor object
    :return: the SVR model and its parameters
    '''
    model = SVR()
    parameters = {'C': [100],
                  'epsilon': [0.01],
                  'gamma': ['auto'],
                  'kernel': ['rbf'],
                  'degree': [2]}

    print('\nChosen Model: Support Vector Regressor\n')

    return model, parameters
