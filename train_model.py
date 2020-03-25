import tensorflow as tf

k = tf.keras

from models import simple_covid_net
from data import load_data

model = simple_covid_net()

model.compile(loss=k.losses.categorical_crossentropy,
              optimizer=k.optizers.Adadelta(),
              metrics=['accuracy'])