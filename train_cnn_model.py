import numpy as np
from random import shuffle
import tensorflow as tf
import keras
from math import ceil
import matplotlib.pyplot as plt 
import math
import cmath
import h5py
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import keras.backend as K
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


from tensorflow.python.framework import graph_io
import pydot
from tensorflow.keras.utils import plot_model


print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

classes = 7
EPOCH = 150

#68%
# BATCH = 128
# dropout_rate = 0.43676722686695313
# kernel_size = 4
# num_filters = 48
# initial_learning_rate = 0.001

#Total params: 729,543 # 68%  
# BATCH = 64
# dropout_rate = 0.44583
# kernel_size = 4
# num_filters = 48
# initial_learning_rate = 0.001

#60%
BATCH = 32
dropout_rate = 0.094323
kernel_size = 3
num_filters = 32
initial_learning_rate = 0.001

#TO RUN
# BATCH = 128
# dropout_rate = 0.45
# kernel_size = 3
# num_filters = 80
# initial_learning_rate = 0.001


data = h5py.File(".\\NewDataset90-120\\WiFiDataset7classes5Candidates80_20_90-120.h5", 'r')
# training_x = np.asarray(data['dataset_rgb'])
x_train = np.asarray(data['x_train'])
x_train = np.expand_dims(x_train, axis=3)
y_train = data['y_train']
y_train = np.asarray(y_train)
y_train = np.expand_dims(y_train, axis=0)
y_train = np.transpose(y_train)

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=20)

# data = h5py.File("C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/TFA-Net-main/TF_sabya/train_cnn/Data/data_unet_testing.h5", 'r')
# training_x = np.asarray(data['dataset_rgb'])
x_test = np.asarray(data['x_test'])
x_test = np.expand_dims(x_test, axis=3)
y_test = data['y_test']
y_test = np.asarray(y_test)
y_test = np.expand_dims(y_test, axis=0)
y_test = np.transpose(y_test)



input_data = tf.keras.Input(shape=x_train.shape[1:])

x = tf.keras.layers.Conv2D(num_filters, (kernel_size,kernel_size))(input_data)
x = tf.keras.layers.Conv2D(num_filters, (kernel_size,kernel_size))(x)
x = tf.keras.layers.MaxPooling2D(2,2)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.Dropout(dropout_rate)(x)

x = tf.keras.layers.Conv2D(num_filters, (kernel_size,kernel_size))(x)
x = tf.keras.layers.Conv2D(num_filters, (kernel_size,kernel_size))(x)
x = tf.keras.layers.MaxPooling2D(2,2)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.Dropout(dropout_rate)(x)

x = tf.keras.layers.Conv2D(2*num_filters, (kernel_size,kernel_size))(x)
x = tf.keras.layers.Conv2D(2*num_filters, (kernel_size,kernel_size))(x)
x = tf.keras.layers.MaxPooling2D(2,2)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.Dropout(dropout_rate)(x)

# x = tf.keras.layers.Conv2D(2*num_filters, (kernel_size,kernel_size))(x)
# x = tf.keras.layers.Conv2D(2*num_filters, (kernel_size,kernel_size))(x)
# x = tf.keras.layers.MaxPooling2D(2,2)(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.activations.relu(x)
# x = tf.keras.layers.Dropout(dropout_rate)(x)


flatten = tf.keras.layers.Flatten()(x)
dense = tf.keras.layers.Dense(256)(flatten)
dropout = tf.keras.layers.Dropout(dropout_rate)(dense)
relu_dense = tf.keras.activations.relu(dropout)


dense2 = tf.keras.layers.Dense(classes)(relu_dense)
output_layer = tf.keras.activations.softmax(dense2)

model = tf.keras.Model(inputs=input_data, outputs=output_layer)
print(model.summary())


# Visualize the model and save it as a PNG image
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate, decay_steps=1000, decay_rate=0.5)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = initial_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"] )

# history=model.fit(x=training_x,
#         y=training_y,
#         batch_size=BATCH,
#         epochs=EPOCH,
#         validation_data=(X_test, y_test),
#         steps_per_epoch=ceil(training_x.shape[0]/BATCH))

history=model.fit(x=x_train,
        y=y_train,
        batch_size=BATCH,
        epochs=EPOCH,
        validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

y_pred = model.predict(x_test)

#plotting confusion_matrix#####################################################

pred_y = []
for i in range(len(y_pred)):
    list_y = list(y_pred[i])
    a = list_y.index(max(list_y))
    pred_y.append(a)
    
cf_matrix = confusion_matrix(y_test, pred_y)

print(cf_matrix)

import seaborn as sns

cf = []
for i in range(len(cf_matrix)):
    cf.append(cf_matrix[i].astype(np.float64)/sum(cf_matrix[i].astype(np.float64)))

##################################################

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(train_acc))

plt.figure(1)
plt.plot(epochs, train_acc, 'r', label='Training acc',linewidth=1)
plt.plot(epochs, val_acc, 'b', label='Validation acc',linewidth=1)
plt.title('Training and Validation Accuracy',fontsize=14)
plt.ylabel('Accuracy',fontsize=14) 
plt.xlabel('Epoch',fontsize=14)
plt.ylim([0, 1.0])
plt.legend()
plt.show()
plt.savefig('.\\NewDataset90-120\\train_vs_val.png')

plt.figure(2)
plt.plot(epochs, train_loss, label='Training loss',linewidth=2)
plt.plot(epochs, val_loss, label='validation Loss',linewidth=2)
plt.title('Training and Validation Losses',fontsize=14)
plt.ylabel('Loss',fontsize=14) 
plt.xlabel('Epoch',fontsize=14)
plt.legend()
plt.show()
plt.savefig('.\\NewDataset90-120\\train_vs_val_loss.png')
#plt.ylim([0, 10])


model.save(".\\NewDataset90-120\\saved_model\\")
model.save_weights('.\\NewDataset90-120\\saved_model\\spect.h5')