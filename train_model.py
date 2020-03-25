import tensorflow as tf

k = tf.keras

from models import simple_covid_net

train_datagen = k.preprocessing.image.ImageDataGenerator()
val_datagen = k.preprocessing.image.ImageDataGenerator()
test_datagen = k.preprocessing.image.ImageDataGenerator()

model = simple_covid_net()

train_generator = train_datagen.flow_from_directory(
    directory="data/train",
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory="data/test",
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    directory="data/validation",
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

model.compile(loss=k.losses.categorical_crossentropy,
              optimizer=k.optimizers.Adadelta(),
              metrics=['accuracy'])


model.summary()

print('START MODEL TRAINING')

model.fit_generator(generator=train_generator, validation_split=0.1, epochs=10)

model.save('models/25032020_covid_simple_net.h5')

