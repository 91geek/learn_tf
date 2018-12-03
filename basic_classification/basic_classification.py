import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from base import *

# 预处理数据
train_images = train_images / 255.0
test_images = test_images / 255.0

# 构建模型
# 设置层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# 编译模型
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
model.fit(train_images, train_labels, epochs=5)
# 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)
model.save("model", True, True)
print('Test accuracy:', test_acc)
