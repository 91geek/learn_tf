import tensorflow as tf
from tensorflow import keras
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
# 下载 IMDB数据集
imdb = keras.datasets.imdb
# 参数 num_words=10000 会保留训练数据中出现频次在前 10000 位的字词。为确保数据规模处于可管理的水平，罕见字词将被舍弃。
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)
