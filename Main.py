import sys
sys.dont_write_bytecode = True

import Data
from ANN.NeuralNetwork import NeuralNetwork
import random
from os.path import exists
import os
import numpy as np

def main() -> None:
    train_data, num_of_images = Data.Preprocessor.load_train_data()
    X_train, Y_train, X_test, Y_test = Data.Preprocessor.preprocess_train_data(train_data)

    neural_network = get_classifier(num_of_images, X_train, Y_train)
    predictions = neural_network.predict(X=X_test)
    print("Accuracy:", neural_network.get_accuracy(predictions, Y_test))

    for i in range(10):
        index = random.randint(0, len(X_test))
        neural_network.test_prediction(index, X=X_test, Y=Y_test)


def get_classifier(num_of_images:int, X:np.array, Y:np.array) -> NeuralNetwork:
    if not exists(os.path.join(Data.Preprocessor.SRC_PATH, "ANN", "model", "network.sav")):
        neural_network = NeuralNetwork(num_of_images)

        neural_network.add_layer(784, 10, "ReLU")
        neural_network.add_layer(10, 10, "softmax")
        
        print("Training the neural network...")
        neural_network.fit(X=X, Y=Y, learning_rate=0.10, epochs=500)
        print("Training complete! Saving model...")

        NeuralNetwork.save_model(neural_network)
        print("Model saved!")

    else:
        print("Model already exists. Loading model...")
        neural_network = NeuralNetwork.load_model()
        print("Model loaded!")

    return neural_network 


if __name__ == "__main__":
    main()
