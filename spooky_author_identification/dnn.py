import numpy as np
import pandas as pd
import nltk as nl
import tensorflow as tf
import tflearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

stemmer = nl.stem.lancaster.LancasterStemmer()


def read_file(file_path):
    return pd.read_csv(file_path)


def preprocess_data(train):
    """
    Process the train data to use a one hot representation for input features

    :param train: Training data
    :return: Process data, Vocabulary list
    """

    # tokenize the words
    train['tokens'] = [nl.word_tokenize(sentences) for sentences in train.text]

    # Create a voacbulary
    words = []
    for item in train.tokens:
        words.extend(item)

    # stem the words and removing the duplicates
    words = [stemmer.stem(word) for word in words]
    words = set(words)

    # Represent each sentence in form of a one vector of size v= Vocabulary size.
    # and set the index to one for those words that exist in the sentence
    training = []
    for index, item in train.iterrows():
        onehot = []
        token_words = [stemmer.stem(word) for word in item['tokens']]
        for w in words:
            onehot.append(1) if w in token_words else onehot.append(0)

        training.append([onehot, item['author']])

    training_new = np.array(training)
    return training_new, words


def one_hot_label(y):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded


def calculate_model_accuracy(result, testing_data):
    # Read the test labels
    test = pd.read_csv(testing_data)
    test.head()
    test['tokens'] = [nl.word_tokenize(sentences) for sentences in test.text]

    # using softmax results to predict the Author
    final_results = dict()
    real_result = dict()
    for index, row in result.iterrows():
        author = ""
        row = list(row)
        id = row.pop(0)
        index = row.index(max(row))
        if index == 0:
            author = "EAP"
        elif index == 1:
            author = "HPL"
        else:
            author = "MWS"
        final_results[id] = author

    for index, row in test.iterrows():
        row = list(row)
        id = row.pop(0)
        author = row.pop(1)

        real_result[id] = author

    total_test_samples = 0
    correct_samples = 0

    for id, author in final_results.items():
        total_test_samples += 1
        if real_result[id] == author:
            correct_samples += 1
    accuracy = correct_samples / total_test_samples
    return accuracy


def get_test_data(filepath, words):
    test = pd.read_csv(filepath)
    test.head()

    test['tokens'] = [nl.word_tokenize(sentences) for sentences in test.text]

    testing = []
    for index, item in test.iterrows():
        onehot = []
        token_words = [stemmer.stem(word) for word in item['tokens']]
        for w in words:
            onehot.append(1) if w in token_words else onehot.append(0)

        testing.append(onehot)

    testing = list(np.array(testing))
    return testing, test


def prediction_using_saved_model(model_path, testing_data, vocab_path):
    num_classes = 3

    # load the vocab file
    pd_word = pd.read_csv(vocab_path)
    loaded_vocab = list(pd_word['0'])
    num_features = len(loaded_vocab)

    # Loading test data
    test_data, test = get_test_data(testing_data, loaded_vocab)

    # design the same old network
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, num_features])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    model.load(model_path)
    predicted = model.predict(X=test_data)

    result_val = round(pd.DataFrame(predicted), 6)
    result_val.columns = ["EAP", "HPL", "MWS"]

    result = pd.DataFrame(columns=['id'])
    result['id'] = test['id']

    result['EAP'] = result_val['EAP']
    result['HPL'] = result_val['HPL']
    result['MWS'] = result_val['MWS']

    result.head()

    acc = calculate_model_accuracy(result, testing_data)
    print("Accuracy: ", acc)


def training(training_data, testing_data, vocab_path, model_path):
    # 1. Read the training data
    training_data = read_file(training_data)
    training_data.head()

    # 2. Preprocess the data and break into features and labels
    processed_training_data, words = preprocess_data(training_data)
    train_x = list(processed_training_data[:, 0])
    train_y = one_hot_label(processed_training_data[:, 1])

    pd_word = pd.DataFrame(list(words))
    pd_word.to_csv(vocab_path)

    # 3. Build a deep neural network with 2 linear fully connected, one fully connected layer with
    # softmax activation, followed by a regression output layer
    num_features = len(train_x[0])
    num_classes = len(train_y[0])
    print(num_features)
    print(num_classes)
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, num_features])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net)

    # 4. Define model and setup for tensorboard
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

    # 5. Train and save the model
    model.fit(train_x, train_y, n_epoch=10, batch_size=8, show_metric=True)
    model.save(model_path)

    # 6. Loading test data
    test_data, test = get_test_data(testing_data, words)

    # 7. Do Prediction
    predicted = model.predict(X=test_data)

    result_val = round(pd.DataFrame(predicted), 6)
    result_val.columns = ["EAP", "HPL", "MWS"]

    result = pd.DataFrame(columns=['id'])
    result['id'] = test['id']

    result['EAP'] = result_val['EAP']
    result['HPL'] = result_val['HPL']
    result['MWS'] = result_val['MWS']

    result.head()

    acc = calculate_model_accuracy(result, testing_data)
    print("\n-----------------------------------------------------------\n")
    print("Accuracy: ", acc)


def testing(model_path, vocab_path, testing_data):
    print("!!! PREDICTION BY DEEP NEURAL NETWORK USING TENSORFLOW !!!")
    print("tflearn is used for training and optimizing the model.")

    prediction_using_saved_model(model_path, testing_data, vocab_path)
