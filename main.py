# Packages
from training import *
from sklearn.datasets import load_boston

def main():
    # DATASET PREPARATION
    dataframe = load_boston()
    scaler = []#...


    # MODEL TRAINING & GRID SEARCH
    test_model(dataframe, scaler)


if __name__ == '__main__':
    main()

