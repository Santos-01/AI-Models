import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt

#Cargar las imagenes de la base de datos en una variable
fashion_mnist = keras.datasets.fashion_mnist  

#Cargar todos los datos contenidos en la varible separandolo en dos conjuntos.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
len(train_labels)
train_labels
len(test_labels)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#normalización de datos (Se hace para que todas las muestras tengan un valor de 0 a 1)
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


model2 = keras.Sequential()
model2.add(keras.Input(shape=(28,28,1))) #Capa de entrada
model2.add(keras.layers.Conv2D(32,5, strides=2, activation ='relu')) #Capa convolucional
model2.add(keras.layers.MaxPool2D(2)) #Capa de pooling
model2.add(keras.layers.Flatten()) #aplanar matriz para conectar a una capa de 128 neuronas
model2.add(keras.layers.Dense(128,activation='relu'))
model2.add(keras.layers.Dense(10,activation='softmax'))

#Hiper parametros
model2.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy', #función de costo, entropía cruzada para datos categoricos (multiclase)
              metrics=['accuracy'])

# Reshape de las imágenes para añadir el canal de color
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

historial = model2.fit(train_images,
                      train_labels,
                      epochs=10,
                      validation_data=(test_images, test_labels))
#Evaluar el modelo
acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']
loss = historial.history['loss']
val_loss = historial.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#Utilizar los datos de prueba para verificar accuracy
test_loss, test_acc = model2.evaluate(test_images,  test_labels, verbose=2) 
print('\nTest accuracy:', test_acc)

predictions = model2.predict(test_images) #Hacer predicciones

print("Predicción de la muestra 0")
print(predictions[0])
print('ID de la clase predicha')
print(np.argmax(predictions[0]))

# Definir funciones auxiliares para la visualización
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img.reshape(28,28), cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
        
    plt.xlabel(f"{class_names[predicted_label]} ({100*np.max(predictions_array):2.0f}%) "+
               f"[{class_names[true_label]}]", color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Definir la carpeta donde se guardará el modelo
carpeta_modelos = 'modelos_guardados'
nombre_modelo = 'modelo_fashion_mnist.h5'
ruta_completa = os.path.join(carpeta_modelos, nombre_modelo)

# Guardar el modelo localmente
print(f"\nGuardando el modelo en {ruta_completa}...")
model2.save(ruta_completa)
print(f"Modelo guardado como '{ruta_completa}'")

# Cargar el modelo guardado y hacer predicciones para verificar
print("\nCargando el modelo guardado...")
nuevo_modelo = keras.models.load_model(ruta_completa)

# Recompilar el modelo con las mismas configuraciones
nuevo_modelo.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Verificar que el modelo cargado funciona igual
print("\nHaciendo predicciones con el modelo cargado...")
new_predictions = nuevo_modelo.predict(test_images)

# Comparar una predicción del modelo original y el cargado
print("\nComparando predicciones:")
print("Predicción del modelo original:", np.argmax(predictions[0]))
print("Predicción del modelo cargado:", np.argmax(new_predictions[0]))
print("Clase real:", test_labels[0])
print("Nombre de la prenda:", class_names[test_labels[0]])

