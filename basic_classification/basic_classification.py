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
	#第一层 tf.keras.layers.Flatten 将图像格式从二维数组（28x28 像素）转换成一维数组（28 * 28 = 784 像素）。
	#可以将该层视为图像中像素未堆叠的行，并排列这些行。该层没有要学习的参数；它只改动数据的格式
    keras.layers.Flatten(input_shape=(28, 28)),
    # 第一个 Dense 层具有 128 个节点（或神经元）
    keras.layers.Dense(128, activation=tf.nn.relu),
    # 第二个（也是最后一个）层是具有 10 个节点的 softmax 层，该层会返回一个具有 10 个概率得分的数组，这些得分的总和为 1
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
