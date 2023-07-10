import streamlit as st
import cv2
import numpy as np
#from PIL import Image
from PIL import Image as Imag, ImageOps as ImagOps
from keras.models import load_model

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento óptico de Caracteres")

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    size = (224, 224)
    imag2 = ImagOps.fit(cv2_img, size, Imag.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(imag2)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0]>0.5:
      st.write('Carlos detectado con probabilidad de: '+str( prediction[0][0]) )
    if prediction[0][1]>0.5:
      st.write('Taza detectada con probabilidad de: '+str( prediction[0][1]))
