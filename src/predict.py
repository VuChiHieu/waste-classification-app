import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("models/model_deploy_final.h5")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(img_array)
    idx = np.argmax(pred)
    return class_names[idx], float(pred[0][idx]) * 100
