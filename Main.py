import sys
sys.dont_write_bytecode = True

import Data
from ANN.NeuralNetwork import NeuralNetwork
import random
from os.path import exists
import os

def main() -> None:
    train_data, num_of_images = Data.Preprocessor.load_train_data()
    X_train, Y_train, X_test, Y_test = Data.Preprocessor.preprocess_train_data(train_data)

    if not exists(os.path.join(Data.Preprocessor.SRC_PATH, "ANN", "model", "network.sav")):
        neural_network = NeuralNetwork(num_of_images)

        neural_network.add_layer(784, 10, "ReLU")
        neural_network.add_layer(10, 10, "softmax")

        neural_network.fit(X=X_train, Y=Y_train, learning_rate=0.10, epochs=500)
        NeuralNetwork.save_model(neural_network)
    else:
        neural_network = NeuralNetwork.load_model()

    predictions = neural_network.predict(X=X_test)
    print("Accuracy:", neural_network.get_accuracy(predictions, Y_test))

    for i in range(10):
        index = random.randint(0, len(X_test))
        neural_network.test_prediction(index, X=X_test, Y=Y_test)


if __name__ == "__main__":
    main()
