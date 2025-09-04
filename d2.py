import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


# Definir diretórios de treino e teste
train_pasta = 'dataset\\train'
test_pasta = 'dataset\\test'


# Pré-processamento de imagens com aumento de dados
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Normaliza para os valoresas cores da imagem estarem entre 0 - 1
test_datagen = ImageDataGenerator(rescale=1.0/255)


# Carregar dados de treino e teste
train_generator = train_datagen.flow_from_directory(
    train_pasta,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_pasta,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)


# Construção do modelo CNN usando add()
model = models.Sequential()

# Primeira camada convolucional + pooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Segunda camada convolucional + pooling
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Terceira camada convolucional + pooling
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


# Quarta camada convolucional + pooling
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# transformar em vetor para a densa
model.add(layers.Flatten())


# Camada totalmente conectada
model.add(layers.Dense(512, activation='relu'))

# Camada de saída com 15 classes
model.add(layers.Dense(15, activation='softmax'))



# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // train_generator.batch_size,
          epochs=10,
          validation_data=test_generator,
          validation_steps=test_generator.samples // test_generator.batch_size)