import numpy as np
import math
from ANN.ActivationFunctions import ActivationFunctions

import sys
sys.dont_write_bytecode = True

class Layer:
    def __init__(self, input_size:int, output_size:int, activation_function:str) -> None:
        self.__W = np.random.rand(output_size, input_size) - 0.5
        self.__b = np.random.rand(output_size, 1) - 0.5
        self.__activation_function = activation_function

    @property
    def W(self) -> np.ndarray:
        return self.__W

    @W.setter
    def W(self, W:np.ndarray) -> None:
        self.__W = W

    @property
    def b(self) -> np.ndarray:
        return self.__b

    @b.setter
    def b(self, b:np.ndarray) -> None:
        self.__b = b

    def activatate(self, layer_output:np.ndarray, activation_function:str=None) -> np.ndarray:
        if activation_function == None:
            activation_function = self.__activation_function

        if activation_function == "ReLU":
            return ActivationFunctions.ReLU(layer_output)
        elif activation_function == "softmax":
            return ActivationFunctions.softmax(layer_output)
        elif activation_function == "sigmoid":
            return ActivationFunctions.sigmoid(layer_output)

    def activatate_derivative(self, layer_output:np.ndarray, activation_function:str=None) -> np.ndarray:
        if activation_function == None:
            activation_function = self.__activation_function

        if activation_function == "ReLU":
            return ActivationFunctions.ReLU_derivative(layer_output)
        elif activation_function == "softmax":
            return ActivationFunctions.softmax_derivative(layer_output)
        elif activation_function == "sigmoid":
            return ActivationFunctions.sigmoid_derivative(layer_output)
