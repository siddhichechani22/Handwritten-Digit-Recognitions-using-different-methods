# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:56:01 2019

@author: siddh
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

### Normalized the data set to converge faster 
x_train = tf.keras.utils.normalize(x_train, axis= 1)
x_test = tf.keras.utils.normalize(x_test, axis= 1)

model = tf.keras.models.Sequential()
### add input layer ########
model.add(tf.keras.layers.Flatten())
### add hidden layer ########
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
## add output layer, Activation function is softmax for probabilistic model.
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train,epochs = 3) 
val_loss, val_acc = model.evaluate(x_test,y_test)
print("validation loss:",val_loss)
print("validation accuracy:",val_acc)
### save the model ###
#model.save('num_classifier.model')
#new_model = tf.keras.models.load_model('num_classifier.model')
predictions = model.predict([x_test])
print(type(predictions))

#predict top 5 element
for index in range(5):
    plt.imshow(x_test[index])
    plt.show()
    print("above digit is: ",np.argmax(predictions[index]))
