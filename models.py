import os

import tensorflow as tf

k = tf.keras

def simple_covid_net():

    covid_net = k.models.Sequential()

    ##################################
    #  FIRST EXPERIMENT - Basic CNN  #
    ##################################

    covid_net.add(k.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(224,224,1)))
    covid_net.add(k.layers.Conv2D(32, kernel_size=(3,3), activation='relu'))
    covid_net.add(k.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
    covid_net.add(k.layers.MaxPooling2D(pool_size=(2,2)))
    covid_net.add(k.layers.Dropout(0.25))
    covid_net.add(k.layers.Flatten())
    covid_net.add(k.layers.Dense(128, activation='relu'))
    covid_net.add(k.layers.Dropout(0.5))
    covid_net.add(k.layers.Dense(4, activation='softmax'))

    return covid_net

def covid_inception():

    base_model = k.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    ##################################
    # SECOND EXPERIMENT - Inception  #
    ##################################

    x = base_model.output
    x = k.layers.GlobalAveragePooling2D()(x)

    x = k.layers.Dense(1024, activation='relu')(x)

    prediction_layer = k.layers.Dense(4, activation='softmax')(x)

    model = k.models.Model(inputs=base_model.input, outputs=prediction_layer)

    for layer in base_model.layers:
        layer.trainable = False

    return model






