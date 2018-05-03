from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

ENABLE_LOGGING = True


def log(data, message="\n"):
    if ENABLE_LOGGING:
        print("{}:  {}".format(message, data))


def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(range(max(x) + 1))
    b = lb.transform(x)
    return b

# Converting space separated pixels to int array


def string_to_int_array(item):
    return [int(p) for p in item.split()]


def get_X_and_y(df):
    X = [string_to_int_array(item) for item in df.iloc[:, 1].values]
    X = np.array(X) / 255.0
    y = np.array(df.iloc[:, 0].values)
    y = np.array(one_hot_encode(y))
    y = y.astype(np.float32, copy=False)
    return X, y


def get_training_set(data):
    not_class1 = data.loc[data['emotion'] != 1]
    class1 = data.loc[data['emotion'] == 1]
    class1_aug = class1
    # This is done to handle the class imbalance for emotion 1 which has only ~500 pics
    for i in range(11):
        class1_aug = class1_aug.append(class1)
    complete_training_set = not_class1.append(class1_aug)
    return get_X_and_y(complete_training_set)


# In[58]:


def get_test_set(data):
    return get_X_and_y(data)

def get_batch(X, y, current_batch, batch_size):
    X_batch = X[current_batch * batch_size:(
        current_batch * batch_size + batch_size)]
    y_batch = y[current_batch * batch_size:(
        current_batch * batch_size + batch_size)]
    return X_batch, y_batch
