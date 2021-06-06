"""Command Line Interface to run and compare all models or test certain model"""

import fire
from spooky_author_identification import prediction_by_frequency
from spooky_author_identification import cnn
from spooky_author_identification import lstm_pos
from spooky_author_identification import dnn
from spooky_author_identification import dataset_separation

_MODEL_PATH = "model"

_DATA_PATH = "data/data.csv"
_TEST_DATA_PATH = "data/test_data.csv"
_TRAIN_DATA_PATH = "data/train_data.csv"

_PRED_BY_FREQ = "freq"
_CNN = "cnn"
_LSTM = "lstm"
_DNN = "dnn"


def run(data_path=_DATA_PATH, test_path=_TEST_DATA_PATH, method=None):
    """
    Run all models using pre-trained models
    :param method: methodology to test data
    :param data_path: path to whole data to divide testing and training
    :param test_path: path to testing data
    """

    dnn_model_path = _MODEL_PATH + "/dnn/model.tflearn"
    dnn_vocab_path = _MODEL_PATH + "/dnn/vocab.csv"

    if method is None:
        # prediction by dnn
        dnn.testing(dnn_model_path, dnn_vocab_path, test_path)
        print("\n-----------------------------------------------------------\n")

        # prediction by frequency
        prediction_by_frequency.testing(_MODEL_PATH, test_path)

        print("\n-----------------------------------------------------------\n")
        print("!!! PREDICTION BY LSTM WITH POS TAGGING")
        print("accuracy: ~65%")

        print("\n-----------------------------------------------------------\n")
        print("!!! PREDICTION BY CNN")
        print("accuracy: ~50%")

        print("\n-----------------------------------------------------------\n")
        print("Please note that accuracy for LSTM and CNN are based on previous experiment with the given data.")
        print("If you want to train and test LSTM or CNN models please run respective command: ")
        print("python cli.py run --method cnn")
        print("pyton cli.py run --method lstm")

    if method == _LSTM:
        lstm_pos.train_and_test(data_path)
    if method == _CNN:
        cnn.train_and_test(data_path)
    if method == _DNN:
        dnn.testing(dnn_model_path, dnn_vocab_path, test_path)
    if method == _PRED_BY_FREQ:
        prediction_by_frequency.testing(_MODEL_PATH, test_path)


def train(method, model_path=_MODEL_PATH, train_data_path=_TRAIN_DATA_PATH, test_data_path=_TEST_DATA_PATH,
          test_ratio=0.2):
    """
    training individual method
    :param method: methodology to train data
    :param model_path: path to store model
    :param train_data_path: path to train data
    :param test_data_path: path to test data
    :param test_ratio: ratio of the data to use for testing
    """

    if method == _PRED_BY_FREQ:
        prediction_by_frequency.create_model(train_data_path, model_path)
    if method == _DNN:
        dnn_vocab_path = model_path + "/dnn/vocab.csv"
        dnn_model_path = model_path + "/dnn/model.tflearn"
        dnn.training(train_data_path, test_data_path, dnn_vocab_path, dnn_model_path)
    if method == _CNN:
        cnn.train_and_test(train_data_path, test_ratio)
    if method == _LSTM:
        lstm_pos.train_and_test(train_data_path)


def separate_data(test_ratio=0.2, data_path=_DATA_PATH, train_path=_TRAIN_DATA_PATH, test_path=_TEST_DATA_PATH):
    """
    :param test_ratio: ratio of data to use for testing
    :param data_path: path to whole data
    :param train_path: path to store training data
    :param test_path: path to store testing data
    :return:
    """
    dataset_separation.separate_data(test_ratio, data_path, train_path, test_path)


if __name__ == '__main__':
    fire.Fire()
