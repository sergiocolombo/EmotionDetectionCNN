import os
import numpy as np
from tensorflow.keras.preprocessing import image
from model_creation import model
import evaluate_model

# Carica i pesi salvati nel modello
model.load_weights('model_weights.weights.h5')

# Definisce le classi di emozioni 
emotion_classes = ['angry', 'happy', 'neutral', 'sad']

def predict_emotion(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    emotion_label = emotion_classes[emotion_index]
    return emotion_label

# Testo tutte le immagini nella cartella di test
test_folder = 'test'
for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    emotion = predict_emotion(img_path)
    print(f'{img_name}: {emotion}')










    