import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

sys.path.append('libs')
import libs
sys.path.append('models')
import cnn_1d as model

# set seed
np.random.seed(1)
tf.random.set_seed(1)


# load and read json file
parser = argparse.ArgumentParser()
parser.add_argument('--json_file', type=str, help='path to json file')
args = parser.parse_args()

json_file = args.json_file

with open(json_file) as f:
    settings = json.load(f)

signal_dataset_folder = settings['signal_dataset_folder']
background_dataset_folder = settings['background_dataset_folder']
output_folder = settings['output_folder']

# create output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# load data
signal_data = libs.load_data(signal_dataset_folder)
background_data = libs.load_data(background_dataset_folder)
print('Signal data shape:', signal_data.shape)
print('Background data shape:', background_data.shape)

labels = np.concatenate([np.ones(signal_data.shape[0]), np.zeros(background_data.shape[0])])
data = np.concatenate([signal_data, background_data])

# shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# split data
train_data = data[:int(0.7*data.shape[0])]
train_labels = labels[:int(0.7*data.shape[0])]
val_data = data[int(0.7*data.shape[0]):int(0.85*data.shape[0])]
val_labels = labels[int(0.7*data.shape[0]):int(0.85*data.shape[0])]
test_data = data[int(0.85*data.shape[0]):]
test_labels = labels[int(0.85*data.shape[0]):]

print('Train data', np.unique(train_labels, return_counts=True))
print('Val data', np.unique(val_labels, return_counts=True))
print('Test data', np.unique(test_labels, return_counts=True))


model = model.create_model(train_data.shape)

# train model
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
]


history = model.fit((train_data[:, :, 0], train_data[:, :, 1]), 
                    train_labels, 
                    epochs=100, 
                    batch_size=32, 
                    validation_data=([val_data[:, :, 0], val_data[:, :, 1]], val_labels), 
                    callbacks=callbacks)

# save model
model.save(os.path.join(output_folder, 'model.h5'))

# do some plots
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.savefig(os.path.join(output_folder, 'loss.png'))
plt.close()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig(os.path.join(output_folder, 'accuracy.png'))
plt.close()

# evaluate model, create confusion matrix
test_loss, test_acc = model.evaluate((test_data[:, :, 0], test_data[:, :, 1]), test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

predictions = model.predict((test_data[:, :, 0], test_data[:, :, 1]))
libs.log_metrics(test_labels, predictions, output_folder=output_folder)



