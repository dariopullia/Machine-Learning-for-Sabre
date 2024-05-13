import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
sys.path.append('libs')
import libs


def create_model(input_shape):
    print('Creating model...')
    print('Input shape:', input_shape)
    input_layer_r = tf.keras.layers.Input(shape=[input_shape[1],1])
    input_layer_l = tf.keras.layers.Input(shape=[input_shape[1],1])
    x_r = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(input_layer_r)
    x_l = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(input_layer_l)

    x_r = tf.keras.layers.MaxPooling1D(pool_size=2)(x_r)
    x_l = tf.keras.layers.MaxPooling1D(pool_size=2)(x_l)

    x_r = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x_r)
    x_l = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x_l)

    x_r = tf.keras.layers.MaxPooling1D(pool_size=2)(x_r)
    x_l = tf.keras.layers.MaxPooling1D(pool_size=2)(x_l)

    x_r = tf.keras.layers.Flatten()(x_r)
    x_l = tf.keras.layers.Flatten()(x_l)

    x = tf.keras.layers.Concatenate()([x_r, x_l])

    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input_layer_r, input_layer_l], outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print('Model created')
    return model