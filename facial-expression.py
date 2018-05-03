
import pandas as pd
import numpy as np
import random

from sklearn.utils import shuffle
import tensorflow as tf
from helpers import one_hot_encode, get_batch, get_training_set, get_test_set, log
from params import HyperParameter
from linear_classifier import LinearClassifer
from cnn import CNN
file = r'data/fer2013.csv'
df = pd.read_csv(file)

train_X, train_y = get_training_set(df.loc[df['Usage'] == 'Training'])
test_X, test_y = get_test_set(df.loc[df['Usage'] == 'PublicTest'])
validate_X, validate_y = get_test_set(df.loc[df['Usage'] == 'PrivateTest'])

labels = 7
image_size = 2304

params = HyperParameter(num_batches=2, batch_size=8,
                        epoch=200, learning_rate=.001, hold_prob=0.5)

log(params)

model = LinearClassifer(params=params, labels=labels, image_size=image_size)
#model = CNN(params=params, labels=labels, image_size=image_size)


log(model.run(train_X=train_X, train_y=train_y, test_X=test_X,
          test_y=test_y, validate_X=validate_X, validate_y=validate_y))
