import pandas as pd
import numpy as np

'''
The purpose of this script is to read the whole dataset, shuffle it and divide it into a training and test set using 
random indices.
'''


def separate_data(test_ratio, data_path, train_path, test_path):
    # Reading the input data file.
    data = pd.read_csv(data_path)
    header = list(data.columns.values)
    data = np.array(data)

    N = len(data)
    indices = list(np.random.permutation(N))
    num_test_sample = int(N * test_ratio)

    test_idx, training_idx = indices[:num_test_sample], indices[num_test_sample:-1]
    train_data, test_data = pd.DataFrame(data[training_idx]), pd.DataFrame(data[test_idx])

    train_data.to_csv(train_path, header=header, index=False)
    test_data.to_csv(test_path, header=header, index=False)
