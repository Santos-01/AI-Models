import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# ================================
# CONFIGURACIÓN DE LA APP
# ================================
st.set_page_config(page_title="Clasificador de prendas - IA", layout="centered")

st.title(" Clasificador de prendas con IA (Usando Fashion MNIST)")
st.write("Sube una imagen de una prenda 28x28 de preferencia, se cambiara a escala de grises y el modelo la clasificará.")

# ================================
# CARGAR EL MODELO
# ================================
@st.cache_resource
def cargar_modelo():
    ruta_modelo = "modelos_guardados/modelo_fashion_mnist.h5"
    modelo = keras.models.load_model(ruta_modelo)
    # Recompilar el modelo con las mismas configuraciones
    modelo.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Crear un pequeño conjunto de datos dummy para inicializar las métricas
    dummy_data = np.zeros((1, 28, 28, 1))
    dummy_labels = np.zeros((1,))
    modelo.evaluate(dummy_data, dummy_labels, verbose=0)
    
    return modelo

modelo = cargar_modelo()

# Clases del dataset Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ================================
# SUBIR UNA IMAGEN
# ================================
archivo = st.file_uploader("Sube una imagen de una prenda (.png o .jpg)", type=["png", "jpg", "jpeg"])

if archivo is not None:
    # Abrir la imagen y convertirla a escala de grises
    imagen = Image.open(archivo).convert('L')

    # Redimensionar a 28x28 píxeles
    imagen = imagen.resize((28, 28))

    # Mostrar imagen original
    st.image(imagen, caption="Imagen cargada", use_container_width=False)

    # Convertir a numpy y normalizar
    img_array = np.array(imagen) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # ================================
    # HACER PREDICCIÓN
    # ================================
    prediccion = modelo.predict(img_array)
    clase_predicha = np.argmax(prediccion)
    confianza = np.max(prediccion) * 100

    st.subheader(" Resultado")
    st.write(f"**Predicción:** {class_names[clase_predicha]}")
    st.write(f"**Confianza:** {confianza:.2f}%")

    # Mostrar barras de probabilidad
    st.bar_chart(prediccion[0])

else:
    st.info("Por favor, sube una imagen para clasificar.")

st.markdown("---")
st.caption("Desarrollado por Iñaki García · Proyecto de IA con TensorFlow y Streamlit")

#Para correr localmente, usar:
# streamlit run Clasificador_de_imagenes.py
