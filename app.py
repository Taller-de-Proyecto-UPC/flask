from io import BytesIO
from flask import Flask, request, jsonify
import numpy as np
import requests
from tensorflow import keras
import tensorflow_hub as hub
from keras.utils import get_custom_objects
from keras.preprocessing import image
import os
from flask_cors import CORS
from PIL import Image
import cv2

get_custom_objects().update({'KerasLayer': hub.KerasLayer})
modelo = keras.models.load_model(f"efficientnet.h5")

#modelo = keras.models.load_model(
#       (f"efficientnet.h5"),
#       custom_objects={'KerasLayer':hub.KerasLayer}
#)

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    img_url = data['img_url']

    response = requests.get(img_url)

    response.raise_for_status()

    img = Image.open(BytesIO(response.content))
    img = np.array(img).astype(float) / 255
    img = cv2.resize(img, (512, 512))
    img = np.reshape(img, (1, 512, 512, 3))
    prediccion = modelo.predict(img)
    print(prediccion)

    # Realiza la predicción
    categoria = np.argmax(prediccion, axis=-1)
    print(categoria)

    # Determina el resultado
    if categoria[0] <= 0.5:
        result = 'Tiene aneurisma'
    else:
        result = 'No tiene aneurisma'
     # Retorna el resultado como JSON
    return jsonify({'result': result})


@app.route('/testing', methods=['POST'])
def test():
    #data = request.get_data(as_text=True)
    #print(test)
    data = request.get_json()

    img_url = data['img_url']
    print(img_url)
    # Tu lógica aquí
    return jsonify({'result': img_url})
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)