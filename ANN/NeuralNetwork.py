import numpy as np
import math
from ANN.Layer import Layer
from matplotlib import pyplot as plt
from Data.Preprocessor import Preprocessor
import pickle
import os

import sys
sys.dont_write_bytecode = True

class NeuralNetwork:
    def __init__(self, m:int) -> None:
        self.__layers:list[Layer] = []
        self.__m = m

    @staticmethod
    def save_model(model:object) -> None:
        pickle.dump(model, open(os.path.join(Preprocessor.SRC_PATH, "ANN", "model", "network.sav"), 'wb'))

    @staticmethod
    def load_model() -> object:
        return pickle.load(open(os.path.join(Preprocessor.SRC_PATH, "ANN", "model", "network.sav"), 'rb'))

    def add_layer(self, input_size:int, output_size:int, activation_function:str) -> None:
        new_layer = Layer(input_size, output_size, activation_function)
        self.__layers.append(new_layer)

    def fit(self, X:np.array, Y:np.array, learning_rate:float, epochs:int) -> None:
        Y_enc = list(Preprocessor.one_hot_encode(Y))[0]
        self.gradient_descent(X, Y, Y_enc, learning_rate, epochs)

    def gradient_descent(self, X:np.array, Y:np.array, Y_enc:np.array, learning_rate:float, epochs:int) -> tuple[np.ndarray]:
        for epoch in range(epochs):
            layers_outputs, activated_layers_outputs = self.forward_propagation(X)
            weight_error, bias_error = self.backward_propagation(layers_outputs, activated_layers_outputs, X, Y_enc)
            self.update_parameters(weight_error, bias_error, learning_rate)

            print("Iteration: ", epoch)
            predictions = self.get_predictions(activated_layers_outputs[-1])
            print(self.get_accuracy(predictions, Y))

    def forward_propagation(self, X:np.array) -> tuple[np.ndarray]:
        layers_outputs:list[np.ndarray] = []
        activated_layers_outputs:list[np.ndarray] = []

        for i in range(len(self.__layers)):
            if i == 0:
                layer_output = self.__layers[i].W.dot(X) + self.__layers[i].b
                activated_layer_output = self.__layers[i].activatate(layer_output)

            else:
                layer_output = self.__layers[i].W.dot(activated_layers_outputs[i-1]) + self.__layers[i].b
                activated_layer_output = self.__layers[i].activatate(layer_output)

            layers_outputs.append(layer_output)
            activated_layers_outputs.append(activated_layer_output)

        return layers_outputs, activated_layers_outputs

    def backward_propagation(self, layers_outputs:list[np.ndarray], activated_layers_outputs:list[np.ndarray], X:np.array, Y:np.array) -> tuple[np.ndarray]:
        m = self.__m
        output_layer_errors = []
        weight_error = []
        bias_error = []

        for i in range(len(self.__layers) - 1, -1, -1):
            if i == 0:
                dZ = self.__layers[i+1].W.T.dot(output_layer_errors[-len(self.__layers) + 1]) * self.__layers[i].activatate_derivative(layers_outputs[i])
                dW = (1 / m) * dZ.dot(X.T)
                db = (1 / m) * np.sum(dZ)
            elif i == len(self.__layers) - 1:
                dZ = activated_layers_outputs[i] - Y
                dW = (1 / m) * dZ.dot(activated_layers_outputs[i-1].T)
                db = (1 / m) * np.sum(dZ)
            else:
                dZ = self.__layers[i+1].W.T.dot(output_layer_errors[-len(self.__layers) + 1]) * self.__layers[i].activatate_derivative(layers_outputs[i])
                dW = (1 / m) * dZ.dot(activated_layers_outputs[i-1].T)
                db = (1 / m) * np.sum(dZ)

            output_layer_errors.append(dZ)
            weight_error.append(dW)
            bias_error.append(db)
        
        return weight_error, bias_error

    def update_parameters(self, weight_error:np.ndarray, bias_error:np.ndarray, learning_rate:float) -> None:
        for i in range(len(self.__layers)):
            self.__layers[i].W = self.__layers[i].W - learning_rate * weight_error[len(self.__layers) - i - 1]
            self.__layers[i].b = self.__layers[i].b - learning_rate * bias_error[len(self.__layers) - i - 1]

    def get_predictions(self, network_output:np.ndarray) -> np.ndarray:
        return np.argmax(network_output, 0)

    def get_accuracy(self, predictions:np.ndarray, Y:np.ndarray) -> float:
        return np.sum(predictions == Y) / Y.size

    def predict(self, X:np.ndarray) -> np.ndarray:
        layers_outputs, activated_layers_outputs = self.forward_propagation(X)
        predictions = self.get_predictions(activated_layers_outputs[-1])

        return predictions

    def test_prediction(self, index:int, X:np.array, Y:np.array) -> float:
        current_image = X[:, index, None]
        prediction = self.predict(X[:, index, None])
        label = Y[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()
