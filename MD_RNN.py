import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import os

# Verificar que TensorFlow está instalado correctamente
print("Versión de TensorFlow:", tf.__version__)

# Cargar dataset IMDB
print("\nCargando dataset IMDB...")
vocab_size = 10000  # Usar las 10,000 palabras más frecuentes
max_len = 200  # Longitud máxima de secuencia
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

# Preprocesar: Rellenar secuencias a longitud fija
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Construir modelo RNN con LSTM
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),  # Embedding layer (removed deprecated input_length)
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),  # LSTM layer
    Dense(1, activation='sigmoid')  # Output layer para clasificación binaria
])

# Compilar modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar modelo
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluar modelo
print("\nEvaluando el modelo en datos de prueba...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f'\nPrecisión en prueba: {accuracy:.4f}')
print(f'Pérdida en prueba: {loss:.4f}')

# Ejemplo de predicción (decodificar una reseña de prueba)
word_index = tf.keras.datasets.imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Probar con algunas reseñas de ejemplo
print("\nProbando el modelo con reseñas de ejemplo:")
for i in range(2):  # Probar con 2 reseñas diferentes
    try:
        sample_review = x_test[i]
        true_sentiment = y_test[i]
        print(f"\nReseña #{i+1}:")
        print("Texto:", decode_review(sample_review))
        prediction = model.predict(sample_review.reshape(1, -1), verbose=0)[0][0]
        sentiment = "positiva" if prediction >= 0.5 else "negativa"
        print(f"Predicción: {sentiment} (valor={prediction:.4f})")
        print(f"Sentimiento real: {'positiva' if true_sentiment == 1 else 'negativa'}")
        print(f"¿Predicción correcta? {'Sí' if (prediction >= 0.5) == true_sentiment else 'No'}")
    except Exception as e:
        print(f"Error al procesar la reseña #{i+1}:", str(e))


# Definir la carpeta y nombre del modelo
carpeta_modelos = 'modelos_guardados'
nombre_modelo = 'modelo_sentimientos_rnn.h5'
ruta_completa = os.path.join(carpeta_modelos, nombre_modelo)

# Guardar el modelo
print(f"\nGuardando el modelo en {ruta_completa}...")
model.save(ruta_completa)
print(f"Modelo guardado como '{ruta_completa}'")
print("")



