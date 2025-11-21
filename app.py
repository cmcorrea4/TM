import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import paho.mqtt.client as paho
import time
import json
import platform

# Estilos CSS personalizados para tipografía negra y cursiva
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        color: black !important;
        font-family: 'Courier New', cursive !important;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

def on_publish(client, userdata, result):             
    print("el dato ha sido publicado \n")
    pass

def on_message(client, userdata, message):
    global message_received
    time.sleep(2)
    message_received = str(message.payload.decode("utf-8"))
    st.write(message_received)

broker = "broker.mqttdashboard.com"
port = 1883
client1 = paho.Client("grego")
client1.on_message = on_message

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes😁")

with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usarlo en esta app para identificar")

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    print(prediction)

    if prediction[0][0] > 0.5:
        st.header('Enciende luz, con Probabilidad: ' + str(prediction[0][0]))
        act1 = "ON"
        client1 = paho.Client("grego")                           
        client1.on_publish = on_publish                          
        client1.connect(broker, port)  
        message = json.dumps({"Act1": act1})
        ret = client1.publish("gregoriomensaje", message)

    if prediction[0][1] > 0.5:
        st.header('Apaga luz, con Probabilidad: ' + str(prediction[0][1]))
        act1 = "OFF"
        client1 = paho.Client("grego")                           
        client1.on_publish = on_publish                          
        client1.connect(broker, port)  
        message = json.dumps({"Act1": act1})
        ret = client1.publish("gregoriomensaje", message)

    #if prediction[0][2] > 0.5:
    #    st.header('Derecha, con Probabilidad: ' + str(prediction[0][2]))
