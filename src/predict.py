import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Dựng lại ĐÚNG kiến trúc như lúc train (base MobileNetV2 + GAP + Dense128 + Dropout + Dense6)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Load weights đã lưu (thay vì load cả model)
model.load_weights("models/model_deploy_final.weights.h5")

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(img_array)
    idx = np.argmax(pred)
    return class_names[idx], float(pred[0][idx]) * 100
