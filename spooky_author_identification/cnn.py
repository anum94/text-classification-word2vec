import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

PADWORD = 'PAD'
N_CLASSES = 3
# Epochs
training_iterations = 500
learning_rate = 0.01
batch_size = 128

# because we have seprarted the testing and training data therefore we need an upper bound that would
# for both data files
Max_sentence_length = 900


def read_data(data_file):
    '''
    Reads the csv format data file using a pandas framework
    :param data_file: Path to the input file
    :return: Pandas read vcs object
    '''

    return pd.read_csv(data_file)


def create_word_dict(texts):
    '''
    For train it is essential that the data be expressed in a numeric form,
    so we creature a vocabulary using all the words in the dataset and assign
    an id to each word.
    This is function generates 2 dictionaries.
    One for word to index mapping and other for index to word mapping.

    :param texts: Input data in form of a pandas dataframe.

    :return:
    index_word_dict: Index to word mapping,
    word_index_dict: Word to Index mapping

    '''
    word_set = set()
    counter = 0

    # Tokenize all data
    for sentence in texts["text"]:
        sentence_tok = nltk.word_tokenize(sentence)
        for word in sentence_tok:
            word_set.add(word)

    # Create dictionaries
    word_index_dict = dict()
    index_word_dict = dict()
    for word in word_set:
        index_word_dict[counter] = word
        word_index_dict[word] = counter
        counter += 1

    return index_word_dict, word_index_dict


def tokenize_pad_sentences(data, word_index_dict):
    '''

    For training it is important that all data samples be of equal length which is not
    the case for our data. To Fix this, we pad all the sentence with "PAD" so they have
    equal length. In training, this would make no difference.

    In this function, we replace the words in each sample with their respective ids, using
    the word_index_dict dictionary. We also pad the sentence so they are of equal length.
     The id for word "pad" is 0.

    :param data: Input data in form of a pandas dataframe.
    :param word_index_dict: Word to Index mapping

    :return:
    '''

    # extract the data and labels from the read data.
    df = pd.DataFrame(columns=['author', 'indexed_text'])
    max_sentence_length = 0

    # Convert words to ids for each sentence
    for _, row in data.iterrows():
        word_tokens = nltk.word_tokenize(row["text"])
        word_indexes = [word_index_dict[word] for word in word_tokens]
        if len(word_indexes) > max_sentence_length:
            max_sentence_length = len(word_indexes)
        df = df.append({'author': row["author"], 'indexed_text': word_indexes}, ignore_index=True)

    # pad the sentences

    for i, row in enumerate(df["indexed_text"]):
        if len(row) < Max_sentence_length:
            pads = [0] * (Max_sentence_length - len(row))
            df["indexed_text"][i] = row + pads
    # Update the Global Variable
    N_FEATURES = Max_sentence_length

    return df


def one_hot_output(data):
    '''
    Transform the labels to one-hot notation
    :param data: Input data in form of a pandas dataframe.
    :return: Updated training data with labels in one-hot encoding format
    '''
    for i, row in enumerate(data["author"]):
        one_hot_encoding = [0] * 3
        if row == "EAP":
            one_hot_encoding[0] = 1
        elif row == "HPL":
            one_hot_encoding[1] = 1
        else:
            one_hot_encoding[2] = 1
        data["author"][i] = one_hot_encoding
    return data


def convert_output_to_number(out):
    '''

    :param out: A list of output labels in string. (The dataset we are using has labels in string)
    :return: list of labels in a numberic form where 0 -> EAP, 1 -> HPL and 2 -> MWS
    '''
    for index, author in enumerate(out):
        if author == "EAP":
            out[index] = 0
        elif author == "HPL":
            out[index] = 1
        else:
            out[index] = 2
    return out


class Cnn_Magic:
    # Class that deals with the Convoution Neural
    def _init_(self):
        '''

        :return:
        '''
        pass

    def experiment(self, train_X, train_y, test_X, test_y, num_train_sample, iterations, lr, bs):
        '''

        The computational graph for tensor is define in this function. and then later computed

        :param train_X: Training data of shape (Number_of_train_samples, number_featues, 1, 1)
        :param train_y: Training labels of shape (Number_of_train_samples, number_class=3)
        :param test_X: Testing data of shape (Number_of_test_samples, number_featues, 1, 1)
        :param test_y: Training labels of shape (Number_of_test_samples, number_class=3)
        :param num_train_sample: Number of training samples.
        :param iterations: Number of Epoch for training
        :param lr: Learning rate of training
        :param bs: Batch size used during training.

        :return: a tuple of average Training Loss, Training Accuracy, Testing Accuracy.

        '''

        self.n_iterations = iterations
        self.learning_rate = lr
        self.batch_size = bs

        # initialize the weight and bias tensor for the Filter (kernal)

        # 16 filters of size 3 by 3 by 1
        w0 = tf.get_variable('W0', shape=(3, 3, 1, 16), initializer=tf.contrib.layers.xavier_initializer())
        # 32 filters of size 3 by 3 by 16
        w1 = tf.get_variable('W1', shape=(3, 3, 16, 32), initializer=tf.contrib.layers.xavier_initializer())
        # weights for the fully connected layer where the output from the previous CNN later is flattened
        w2 = tf.get_variable('W2', shape=(Max_sentence_length * 32, 64),
                             initializer=tf.contrib.layers.xavier_initializer())
        # weights for the output layer
        w3 = tf.get_variable('W3', shape=(64, N_CLASSES), initializer=tf.contrib.layers.xavier_initializer())

        # biases for all layers
        b0 = tf.get_variable('B0', shape=(16), initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable('B3', shape=(3), initializer=tf.contrib.layers.xavier_initializer())

        weights = {
            'wc1': w0,
            'wc2': w1,
            'wd1': w2,
            'out': w3,
        }

        biases = {
            'bc1': b0,
            'bc2': b1,
            'bd1': b2,
            'out': b3,
        }

        # init Regularization constant
        beta = 0.1

        # define place holder for both training data and output labels
        x = tf.placeholder("float", [None, Max_sentence_length, 1, 1])
        y = tf.placeholder("float", [None, 3])

        # Forward pass of CNN
        pred = self.conv_net(x, weights, biases)

        # Compute the loss
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

        # include regularization to avoid overfitting
        regularizer = tf.nn.l2_loss(weights['wc1'] + tf.nn.l2_loss(weights['wc2']) + tf.nn.l2_loss(weights['wd1'])
                                    + tf.nn.l2_loss(weights['out']))

        cost = tf.reduce_mean(cost + beta * regularizer)

        # Tensor for back propagation
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Here you check whether the index of the maximum value of the predicted author is equal to the
        # actual labelled author.
        # y -> real labels
        # pred -> our prediction from the forward pass
        # both 'pred' and 'y' are column vectors.

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        # calculate accuracy across all the given sentences and average them out.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initializing the weights and biases variables using Xavier initialization
        init = tf.global_variables_initializer()

        # Run the session
        with tf.Session() as sess:
            sess.run(init)

            train_loss = []
            train_accuracy = []
            test_accuracy = []
            loss = 0
            acc = 0

            for i in range(self.n_iterations):

                for batch in range(num_train_sample // self.batch_size):
                    batch_x = np.array(
                        train_X[batch * self.batch_size:min((batch + 1) * self.batch_size, num_train_sample)])
                    batch_y = np.array(
                        train_y[batch * self.batch_size:min((batch + 1) * self.batch_size, num_train_sample)])

                    # Run optimization tensor which perform Backpropagation and updates the weights.
                    opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})

                    # Saving the loss and accuracy which would be later used for plotting
                    train_loss.append(loss)
                    train_accuracy.append(acc)

                # Print loss at the end of each Epoch
                print("Epoch " + str(i) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

            # Calculate test accuracy using 5 batches from testing data
            num_test_batch = 5
            test_batch_size = int(len(test_X) / num_test_batch)

            for batch_number in range(num_test_batch):
                test_batch_x = np.array(
                    test_X[batch_number * test_batch_size:min((batch_number + 1) * test_batch_size, len(test_X))])
                test_batch_y = np.array(
                    test_y[batch_number * test_batch_size:min((batch_number + 1) * test_batch_size, len(test_y))])

                test_acc, test_loss = sess.run([accuracy, cost], feed_dict={x: test_batch_x, y: test_batch_y})

                test_accuracy.append(test_acc)
                print("Testing Accuracy:", "{:.5f}".format(test_acc))

        # Plotting training Accuracy and Training loss
        x_axis = [i for i in range(len(train_accuracy))]
        plt.figure(1)
        plt.subplot(211)
        plt.plot(x_axis, train_accuracy, 'ro', )
        plt.axis([0, len(x_axis), 0, 1])
        plt.xlabel('Number of Iterations.')
        plt.ylabel('Training Accuracy')

        plt.subplot(212)
        plt.plot(x_axis, train_loss, 'bo', )
        plt.axis([0, len(x_axis), 0, max(train_loss)])
        plt.xlabel('Number of Iterations.')
        plt.ylabel('Training Loss')
        plt.show()

        # plt.plot(x_axis, train_loss, 'bo')
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_acc = sum(train_accuracy) / len(train_accuracy)
        avg_test_acc = sum(test_accuracy) / len(test_accuracy)

        return avg_train_loss, avg_train_acc, avg_test_acc

    def conv2d(self, x, W, b, strides=1):
        '''
         The function performs on CNN operation.
        :param x: Input data
        :param W: Filter weights
        :param b: Bias values
        :param strides: Strides for convolution
        :return: output of the activation of the cnn layer output
        '''

        x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def conv_net(self, x, weights, biases):

        '''
        This function takes care of the whole forward pass of CNN
        :param x: Input features of size (number_of_samples_in_batch, num_features, 1 ,1)
        :param weights: a dictionary of weights for all layers
        :param biases: a dictionary of biases for all layers
        :return:
        '''
        # here we call the conv2d function we had defined above and pass the input sentences, weights
        # wc1 and bias bc1.

        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])

        # second layer of CNN
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input with weights in 2 Dimension.
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Output, class prediction
        # For the output layer, we multiply the fully connected layer output with output weights
        # and add a bias term.
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

        return out


def get_data(filename):
    '''
    Read and Pre process the data
    :param filename: Input file name
    :return: preprocessed data
    '''
    data = read_data(filename)
    # create dictionaries , tokenize ,etc
    index_word_dict, word_index_dict = create_word_dict(data)
    tokenize_data = tokenize_pad_sentences(data, word_index_dict)
    tokenize_data = one_hot_output(tokenize_data)

    # extract features
    data_x = list(tokenize_data['indexed_text'])
    # extract labels
    data_y = np.array(list(tokenize_data['author']))

    data_x = np.array(data_x)
    # reshape to fit tensors dimensions
    data_x = data_x.reshape(data_x.shape[0], Max_sentence_length, 1, 1)
    return data_x, data_y


'''
    train_and_test function is used by CLI
'''


def train_and_test(data, test_ratio = 0.2):

    print("!!! TRAIN AND TESTING USING CNN MODEL !!!")
    # 1. Reading and processing training data
    data_x, data_y = get_data(data)

    #  2. Breaking data into test and train test
    N = len(data_x)
    indices = list(np.random.permutation(N))
    num_test_sample = int(N * test_ratio)
    test_idx, training_idx = indices[:num_test_sample], indices[num_test_sample:-1]
    train_data_x, test_data_x = data_x[training_idx], data_x[test_idx]
    train_data_y, test_data_y = data_y[training_idx], data_y[test_idx]

    # 3. Create CNN class object
    cnn_model = Cnn_Magic()

    # 4. Create and run model
    train_loss, train_acc, test_acc = cnn_model.experiment(train_X=train_data_x, train_y=train_data_y,
                                                           test_X=test_data_x, test_y=test_data_y,
                                                           num_train_sample=len(train_data_x),
                                                           iterations=training_iterations, lr=learning_rate,
                                                           bs=batch_size)

    # 5. Print final results
    print("For learning rate ", learning_rate, " and batch size ", batch_size, " , the Test accuracy is ", test_acc,
          ".")
    print("Finish")


