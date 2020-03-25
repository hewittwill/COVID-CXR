import tensorflow as tf

k = tf.keras

from models import simple_covid_net

train_datagen = k.preprocessing.ImageDataGenerator()
val_datagen = k.preprocessing.ImageDataGenerator()
test_datagen = k.preprocessing.ImageDataGenerator()

model = simple_covid_net()

train_generator = train_datagen.flow_from_directory(
    directory="data/train",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory="data/test",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    directory="data/validation",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

model.compile(loss=k.losses.categorical_crossentropy,
              optimizer=k.optizers.Adadelta(),
              metrics=['accuracy'])


model.summary()

