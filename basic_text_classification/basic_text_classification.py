import tensorflow as tf
from tensorflow import keras
import numpy as np
from base import *


print("Training entries: {}, labels: {}".format(
    len(train_data), len(train_labels)))
print(train_data[0])
# 影评的长度可能会有所不同。以下代码显示了第一条和第二条影评中的字词数。由于神经网络的输入必须具有相同长度，因此我们稍后需要解决此问题
print(len(train_data[0]), len(train_data[1]))

# 将整数转换回字词
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))

# 准备数据
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding='post', maxlen=256)
print(len(train_data[0]), len(train_data[1]))
print(train_data[0])
# 构建模型
vocab_size = 10000
model = keras.Sequential()
# 该层会在整数编码的词汇表中查找每个字词-索引的嵌入向量。模型在接受训练时会学习这些向量
model.add(keras.layers.Embedding(vocab_size, 16))
# GlobalAveragePooling1D 层通过对序列维度求平均值，针对每个样本返回一个长度固定的输出向量。这样，模型便能够以尽可能简单的方式处理各种长度的输入
model.add(keras.layers.GlobalAveragePooling1D())
# 该长度固定的输出向量会传入一个全连接 (Dense) 层（包含 16 个隐藏单元）
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# 最后一层与单个输出节点密集连接。应用 sigmoid 激活函数后，结果是介于 0 到 1 之间的浮点值，表示概率或置信水平。
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

# 隐藏单元
# 损失函数和优化器
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy', metrics=['accuracy'])

# 创建验证集
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
# 评估模型
results = model.evaluate(test_data, test_labels)
print(results)

# 创建准确率和损失随时间变化的图
history_dict = history.history
print(history_dict.keys())

import matplotlib.pyplot as plt
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
# 验证损失和准确率的变化情况并非如此，它们似乎在大约 20 个周期后达到峰值。
# 这是一种过拟合现象：模型在训练数据上的表现要优于在从未见过的数据上的表现。
# 在此之后，模型会过度优化和学习特定于训练数据的表示法，而无法泛化到测试数据。
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
