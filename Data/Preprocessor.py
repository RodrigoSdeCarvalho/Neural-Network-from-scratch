import pandas as pd
import numpy as np
import os

import sys
sys.dont_write_bytecode = True

class Preprocessor:
    SRC_PATH = os.getcwd()
    DATA_FOLDER_PATH = os.path.join(SRC_PATH, "Data")

    @staticmethod
    def load_train_data() -> tuple[np.array, int]:
        train_data = pd.read_csv(os.path.join(Preprocessor.DATA_FOLDER_PATH, "train.csv"))
        train_data = np.array(train_data)

        return train_data, train_data.shape[0]

    @staticmethod
    def load_test_data() -> tuple[np.array, int]:
        test_data = pd.read_csv(os.path.join(Preprocessor.DATA_FOLDER_PATH, "test.csv"))
        test_data = np.array(test_data)

        return test_data, test_data.shape[0]

    @staticmethod
    def preprocess_train_data(train_data: np.array) -> np.array:
        train_data = Preprocessor.shuffle_data(train_data)
        X_train, Y_train, X_dev, Y_dev = Preprocessor.split_train_data(train_data, 1000)
        X_train, X_dev = Preprocessor.standardize_data(X_train, X_dev)

        return X_train, Y_train, X_dev, Y_dev

    @staticmethod
    def shuffle_data(data: np.array) -> np.array:
        np.random.shuffle(data)

        return data

    @staticmethod
    def split_train_data(train_data: np.array, dev_size: int) -> tuple[np.array, np.array, np.array,np.array]:
        m, n = train_data.shape
    
        dev_data = train_data[0:dev_size].T
        Y_dev = dev_data[0]
        X_dev = dev_data[1:n]

        train_data = train_data[dev_size:m].T
        Y_train = train_data[0]
        X_train = train_data[1:n]
    
        return X_train, Y_train, X_dev, Y_dev

    @staticmethod
    def standardize_data(*args: tuple[np.array]) -> tuple[np.array]:
        normalized_data = []
        for data in args:
            data = data / 255
            normalized_data.append(data)

        return tuple(normalized_data)

    @staticmethod
    def one_hot_encode(*args: tuple[np.array]) -> np.array:
        encoded_labels_list = []
        for labels in args:
            encoded_labels = np.zeros((labels.size, labels.max() + 1))
            encoded_labels[np.arange(labels.size), labels] = 1
            encoded_labels = encoded_labels.T
            
            encoded_labels_list.append(encoded_labels)
    
        return tuple(encoded_labels_list)
