import math
import numpy as np

import sys
sys.dont_write_bytecode = True

class ActivationFunctions:
    @staticmethod
    def ReLU(layer_output:np.ndarray) -> np.ndarray:
        return np.maximum(layer_output, 0)

    @staticmethod
    def ReLU_derivative(layer_output:np.ndarray) -> np.ndarray:
        return layer_output > 0

    @staticmethod
    def softmax(layer_output:np.ndarray) -> np.ndarray:
        return np.exp(layer_output) / sum(np.exp(layer_output))

    @staticmethod
    def softmax_derivative(layer_output:np.ndarray) -> np.ndarray:
        return ActivationFunctions.softmax(layer_output) * (1 - ActivationFunctions.softmax(layer_output))

    @staticmethod
    def sigmoid(layer_output:np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-layer_output))

    @staticmethod
    def sigmoid_derivative(layer_output:np.ndarray) -> np.ndarray:
        return ActivationFunctions.sigmoid(layer_output) * (1 - ActivationFunctions.sigmoid(layer_output))
