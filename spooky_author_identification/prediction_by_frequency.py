import nltk
import pandas as pd
import pickle


class PredictionByFreq:
    AUTHOR = "author"
    WORD = "word"
    PROBABILITY = "probability"
    JOINT_PROBABILITY = "joint_probability"
    TEXT = "text"

    def __init__(self):
        pass

    def generate_model(self, training_data, model_path):

        """
        The function serialize and store term frequency of each word and group by author.
        :param training_data: path to data to calculate term frequency.
        :param model_path: path to dump the serialized model
        """

        training_texts = pd.read_csv(training_data)
        groups_by_author = training_texts.groupby(self.AUTHOR)
        word_freq_by_author = nltk.probability.ConditionalFreqDist()

        for author_name, group_name in groups_by_author:
            sentences = group_name[self.TEXT].str.cat(sep=' ').lower()
            word_tokens = nltk.tokenize.word_tokenize(sentences)
            word_freqs = nltk.FreqDist(word_tokens)
            word_freq_by_author[author_name] = word_freqs

        f = open(model_path, "wb")
        pickle.dump(word_freq_by_author, f)
        f.close()
        print("generated prediction by frequency model successfully")

    def predict_sentence(self, sentence, model):

        """
        The function predict the author for a given sentence.
        :param sentence: the sentence to predict an author.
        :param model: path to the model storing term frequencies
        """

        preprocessed_sentence = nltk.tokenize.word_tokenize(sentence.lower())
        prob_of_word_by_author = pd.DataFrame(columns=[self.AUTHOR, self.WORD, self.PROBABILITY])

        for i in model.keys():
            for j in preprocessed_sentence:
                word_freq = model[i].freq(j) + 0.000001
                output = pd.DataFrame([[i, j, word_freq]], columns=[self.AUTHOR, self.WORD, self.PROBABILITY])

                prob_of_word_by_author = prob_of_word_by_author.append(output, ignore_index=True)

        test_probabilities_by_author = pd.DataFrame(columns=[self.AUTHOR, self.JOINT_PROBABILITY])

        for i in model.keys():
            one_author = prob_of_word_by_author.query(self.AUTHOR + ' == "' + i + '"')
            joint_probability = one_author.product(numeric_only=True)[0]

            output = pd.DataFrame([[i, joint_probability]], columns=[self.AUTHOR, self.JOINT_PROBABILITY])
            test_probabilities_by_author = test_probabilities_by_author.append(output, ignore_index=True, sort=True)

        return test_probabilities_by_author.loc[test_probabilities_by_author[self.JOINT_PROBABILITY].idxmax(),
                                                self.AUTHOR]

    def testing(self, path_to_model, test_data):

        """
        The function calculate the accuracy for given test data and term frequencies
        :param path_to_model: path to model storing the term frequencies
        :param test_data: path to test data
        """

        f = open(path_to_model, "rb")
        word_freq_by_author_model = pickle.load(f)
        f.close()

        t = 0
        f = 0
        test_texts = pd.read_csv(test_data)
        counter = 0
        for sentence, author in zip(test_texts[self.TEXT], test_texts[self.AUTHOR]):
            counter += 1
            predicted_author = self.predict_sentence(sentence, word_freq_by_author_model)
            if predicted_author == author:
                t += 1
            else:
                f += 1

            if counter % 100 == 0:
                print("accuracy of %s sentences is %s percentage " % (str(counter), str((t/(t+f))*100)))
            if counter == 400:
                break
        return t, f


'''
functions in outer scope called by command line interface
'''


def testing(path_to_model, path_to_test_data):
    print("!!! PREDICTION BY FREQUENCY !!!")
    pred_by_freq = PredictionByFreq()

    path_to_model = path_to_model + "/" + "pred_by_freq.pkl"
    t, f = pred_by_freq.testing(path_to_model, path_to_test_data)
    print("accuracy of model is %s " % (str((t/(t+f))*100)))
    print("PREDICTION BY FREQUENCY IS COMPLETED")


def predict_sentence(path_to_model, sentence):
    pred_by_freq = PredictionByFreq()

    f = open(path_to_model, "rb")
    word_freq_by_author_model = pickle.load(f)
    f.close()

    predicted_author = pred_by_freq.predict_sentence(sentence, word_freq_by_author_model)
    print("\"%s\" is most likely said by %s" % (sentence, predicted_author))


def create_model(path_to_train_data, path_to_model):
    path_to_model = path_to_model + "/" + "pred_by_freq.pkl"
    pred_by_freq = PredictionByFreq()
    pred_by_freq.generate_model(path_to_train_data, path_to_model)

